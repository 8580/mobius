#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Schmee-Hahn censored Gaussian Process
#
"""
Schmee-Hahn imputation for right-, left- and interval-censored targets.

The Schmee-Hahn algorithm (Schmee & Hahn, *Technometrics* 21:417-432, 1979)
alternates between

  **E-step**  replace each censored sentinel by ``E[Y | Y in censoring interval]``
              under the currently fitted model, and

  **M-step**  refit the model to the completed data set.

:class:`CensoredEMGPModel` wraps any Mobius surrogate that exposes
``fit(X, y, y_noise=None)`` and ``predict(X)``, so it works with
:class:`~mobius.surrogate_models.GPModel`,
:class:`~mobius.surrogate_models.GPLLModel` and friends without modification.

Two departures from a naive transcription of the 1979 paper matter enough to
call out, because without them the algorithm is close to a no-op on a GP:

1. **The E-step is out-of-sample.** A GP with small noise very nearly
   interpolates its training data, so the *in-sample* predictive mean at a
   censored point is essentially the value that was just imputed there. Feeding
   that back through the truncated-normal update gives a fixed point at the
   starting value: the algorithm converges immediately to naive substitution
   and reports success. The E-step therefore uses the exact leave-one-out
   predictive distribution (Rasmussen & Williams, 2006, eq. 5.12), which is
   available in closed form for an exact GP at the cost of one Cholesky
   factorisation, and falls back to k-fold refitting for models that do not
   expose GPyTorch internals.

2. **The imputed points carry their own extra variance.** Schmee-Hahn is known
   to underestimate dispersion because it substitutes a conditional *mean* and
   then treats it as if it were a measurement. On a GP that shows up as the
   fitted noise collapsing towards zero over successive iterations, which in
   turn makes the E-step increments shrink and the marginal likelihood
   optimisation ill-conditioned. Passing the truncated *variance*
   ``Var[Y | Y in interval]`` through as known observation noise restores the
   dispersion the imputation threw away, which is exactly the term a proper EM
   would keep in its Q-function.

When more than roughly a third of the data is censored, no amount of care in
the E-step makes imputation trustworthy: the imputed points start reinforcing
one another and the iteration drifts. Use
:class:`~mobius.surrogate_models.tobit_gaussian_process.TobitGPModel` in that
regime; it optimises the censored likelihood directly and never invents values.
"""
from __future__ import annotations

import copy
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:  # gpytorch is a hard dependency of Mobius, but keep the import defensive
    import gpytorch
except ImportError:  # pragma: no cover
    gpytorch = None

from .censoring import CensoringSpec, censored_normal_logpdf, truncated_normal_moments
from .surrogate_model import _SurrogateModel

__all__ = ["CensoredEMGPModel", "SchmeeHahnGPModel", "EMResult"]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class EMResult:
    """Diagnostics from a Schmee-Hahn run.

    Attributes
    ----------
    imputed_y : ndarray of shape (n_samples,)
        The completed target vector the final model was fitted to.
    spec : `CensoringSpec`
        The censoring pattern that was applied.
    imputed_variance : ndarray of shape (n_samples,)
        ``Var[Y | Y in interval]`` at convergence; zero for observed rows. This
        is the extra observation noise handed to the surrogate when
        ``variance_correction`` is enabled.
    converged : bool
        Whether the maximum change in the imputations fell below ``tol``.
    n_iter : int
        Number of completed EM iterations.
    max_change : float
        Largest absolute change in any imputed value on the final iteration.
    history : list of dict
        Per-iteration record with keys ``iteration``, ``max_change``,
        ``mean_imputed``, ``censored_loglik`` and ``mean_predictive_sd``.
    e_step : str
        Which E-step was actually used (``'loo'``, ``'kfold'`` or ``'in_sample'``).
    """

    imputed_y: np.ndarray
    spec: CensoringSpec
    imputed_variance: np.ndarray
    converged: bool
    n_iter: int
    max_change: float
    history: List[Dict[str, float]] = field(default_factory=list)
    e_step: str = "loo"

    @property
    def censored_mask(self) -> np.ndarray:
        """Backwards-compatible alias for ``spec.censored``."""
        return self.spec.censored

    @property
    def lower_bounds(self) -> np.ndarray:
        """Backwards-compatible alias for ``spec.lower``."""
        return self.spec.lower


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _take(X: Any, index: np.ndarray) -> Any:
    """Index ``X`` along the sample axis whatever container it is."""
    if isinstance(X, np.ndarray):
        return X[index]
    if torch.is_tensor(X):
        return X[torch.as_tensor(index)]
    if isinstance(X, (list, tuple)):
        idx = np.flatnonzero(index) if np.asarray(index).dtype == bool else np.asarray(index)
        return [X[int(i)] for i in idx]
    return np.asarray(X)[index]


def _accepts_y_noise(model: Any) -> bool:
    """Whether ``model.fit`` has a ``y_noise`` parameter."""
    try:
        return "y_noise" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return False


def _extract_mean_std(prediction: Any, min_std: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise the several shapes a Mobius/GPyTorch ``predict`` can return."""
    if isinstance(prediction, tuple):
        if len(prediction) < 2:
            raise TypeError("predict() returned a 1-tuple; expected (mean, std).")
        mean, std = prediction[0], prediction[1]
    elif hasattr(prediction, "mean"):
        mean = prediction.mean
        if hasattr(prediction, "stddev"):
            std = prediction.stddev
        elif hasattr(prediction, "variance"):
            std = torch.sqrt(torch.as_tensor(prediction.variance).clamp_min(0.0))
        else:
            raise TypeError("Prediction exposes `mean` but neither `stddev` nor `variance`.")
    else:
        raise TypeError(
            f"Unsupported prediction type {type(prediction)!r} from base_model.predict(); "
            "expected a (mean, std) tuple or a distribution object."
        )

    mean = mean.detach().cpu().numpy() if torch.is_tensor(mean) else np.asarray(mean)
    std = std.detach().cpu().numpy() if torch.is_tensor(std) else np.asarray(std)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if len(mean) != len(std):
        raise ValueError("predict() returned mean and std of different lengths.")
    return mean, np.maximum(std, min_std)


def exact_gp_loo_predictive(
    model: Any, jitter: float = 1e-8
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed-form leave-one-out predictive distribution of a fitted exact GP.

    For ``y ~ N(m, K_y)`` with ``K_y = K_f + sigma_n^2 I`` the LOO predictive
    for observation ``i`` is (Rasmussen & Williams, 2006, eq. 5.12)

    .. math::

        \\mu_{-i} = y_i - \\frac{[K_y^{-1}(y - m)]_i}{[K_y^{-1}]_{ii}},
        \\qquad
        \\sigma^2_{-i} = \\frac{1}{[K_y^{-1}]_{ii}}

    which costs a single Cholesky factorisation for all ``n`` folds at once.
    Note ``mu`` does not in fact depend on ``y_i``: expanding the numerator
    cancels the ``i``-th term exactly, which is precisely the property that
    stops the E-step from chasing its own imputation.

    Parameters
    ----------
    model : object
        A Mobius surrogate exposing ``_model`` (a ``gpytorch.models.ExactGP``)
        and ``_likelihood``.
    jitter : float, default : 1e-8
        Diagonal jitter added before the Cholesky factorisation.

    Returns
    -------
    (mu, sigma) or None
        Arrays of shape (n_train,) aligned with the training set, or ``None``
        if ``model`` is not an exact-GP surrogate.
    """
    gp = getattr(model, "_model", None)
    likelihood = getattr(model, "_likelihood", None)
    if gp is None or likelihood is None or gpytorch is None:
        return None
    if not isinstance(gp, gpytorch.models.ExactGP):
        return None
    if getattr(gp, "train_inputs", None) is None or gp.train_targets is None:
        return None

    was_training = gp.training, likelihood.training
    try:
        # In train mode an ExactGP returns the *prior*; pushing it through the
        # likelihood gives exactly K_y and the prior mean.
        gp.train()
        likelihood.train()
        with torch.no_grad():
            prior = likelihood(gp(*gp.train_inputs))
            K_y = prior.covariance_matrix.double()
            mean = prior.mean.double().reshape(-1)
            y = gp.train_targets.double().reshape(-1)
    except Exception as exc:  # pragma: no cover - model-specific failures
        warnings.warn(
            f"Could not build the exact-GP LOO predictive ({exc!r}); "
            "falling back to the k-fold E-step.",
            RuntimeWarning,
        )
        return None
    finally:
        gp.train(was_training[0])
        likelihood.train(was_training[1])

    n = K_y.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device=K_y.device)
    for scale in (1.0, 10.0, 100.0, 1e4):
        try:
            chol = torch.linalg.cholesky(K_y + jitter * scale * eye)
            break
        except Exception:  # pragma: no cover - ill-conditioned kernels
            chol = None
    if chol is None:  # pragma: no cover
        warnings.warn("K_y is not positive definite; skipping the LOO E-step.", RuntimeWarning)
        return None

    K_inv = torch.cholesky_inverse(chol)
    diag = torch.diagonal(K_inv).clamp_min(torch.finfo(torch.float64).tiny)
    mu = y - (K_inv @ (y - mean)) / diag
    sigma = torch.sqrt(1.0 / diag)
    return mu.cpu().numpy(), sigma.cpu().numpy()


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class CensoredEMGPModel(_SurrogateModel):
    """Schmee-Hahn imputation wrapper around any Mobius surrogate model.

    The wrapper is a ``_SurrogateModel`` in its own right, so it drops straight
    into a Mobius acquisition function. Attributes it does not define itself
    are forwarded to ``base_model``, which keeps ``device``, ``score`` and any
    model-specific accessors working.

    Parameters
    ----------
    base_model : `_SurrogateModel`
        The surrogate to wrap. Must expose ``fit(X, y)`` and ``predict(X)``.
        Exact-GP models (``GPModel``, ``GPLLModel``) additionally unlock the
        analytic leave-one-out E-step.
    censoring_limit : float or array-like of shape (n_samples,), default : None
        The assay detection limit. Can also be supplied to :meth:`fit`.
    censored_mask : array-like of bool of shape (n_samples,), default : None
        Explicit censoring mask. If omitted it is inferred by comparing the
        targets against ``censoring_limit``.
    direction : {'right', 'left'}, default : 'right'
        ``'right'`` means the true value lies at or above the limit, which is
        the usual case for a potency reported as ``pIC50 > 4.5``.
    method : {'mean', 'sample'}, default : 'mean'
        ``'mean'`` is the deterministic Schmee-Hahn conditional mean.
        ``'sample'`` draws from the truncated normal instead, giving a
        stochastic-EM variant that propagates more of the imputation
        uncertainty at the cost of a noisy convergence test.
    e_step : {'auto', 'loo', 'kfold', 'in_sample'}, default : 'auto'
        How predictions at censored points are obtained. ``'auto'`` uses the
        analytic LOO predictive when the base model is an exact GP and k-fold
        otherwise. ``'in_sample'`` reproduces the naive behaviour and is
        retained only for comparison - see the module docstring for why it
        collapses onto simple substitution.
    n_splits : int, default : 5
        Number of folds for the k-fold E-step.
    max_iter : int, default : 50
        Maximum number of EM iterations.
    tol : float, default : 1e-4
        Convergence threshold on the largest absolute change in an imputation.
    damping : float, default : 0.7
        Weight given to the new proposal, in ``(0, 1]``. Values below 1 slow
        the positive feedback between mutually correlated censored points.
    variance_correction : bool, default : True
        Pass ``Var[Y | Y in interval]`` to the surrogate as known observation
        noise on the imputed rows. Requires ``base_model.fit`` to accept
        ``y_noise``; silently disabled otherwise.
    max_imputation_sd : float or None, default : 3.0
        Cap each imputation at ``bound + max_imputation_sd * sigma``. Guards
        against a single badly-extrapolated point running away.
    predictive_noise : float, default : 0.0
        Additional observation standard deviation added in quadrature during
        the E-step. Leave at zero for GPyTorch models, whose ``predict``
        already marginalises the likelihood; set it to the assay repeatability
        SD only if the base model returns latent uncertainty.
    min_std : float, default : 1e-6
        Floor on predictive standard deviations.
    detection_atol : float, default : 1e-8
        Tolerance used when inferring the censoring mask from the limit.
    refit_attempts : int, default : 3
        How many times to retry a failed ``base_model.fit`` after jittering the
        kernel hyperparameters. Marginal-likelihood optimisation on a data set
        with a spike of identical values at the detection limit fails more
        often than on ordinary data.
    random_state : int, default : None
        Seed for the ``'sample'`` method and the k-fold split.
    verbose : bool, default : False
        Print per-iteration diagnostics.

    Attributes
    ----------
    em_result_ : `EMResult`
        Diagnostics from the last call to :meth:`fit`.

    Examples
    --------
    >>> from mobius import GPModel
    >>> from mobius.surrogate_models import CensoredEMGPModel
    >>> import gpytorch
    >>> gp = GPModel(kernel=gpytorch.kernels.MaternKernel(nu=2.5))       # doctest: +SKIP
    >>> model = CensoredEMGPModel(gp, censoring_limit=4.5)               # doctest: +SKIP
    >>> model.fit(X_train, y_train)                                      # doctest: +SKIP
    >>> mu, sigma = model.predict(X_test)                                # doctest: +SKIP
    """

    def __init__(
        self,
        base_model: Any,
        *,
        censoring_limit: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        direction: str = "right",
        method: str = "mean",
        e_step: str = "auto",
        n_splits: int = 5,
        max_iter: int = 50,
        tol: float = 1e-4,
        damping: float = 0.7,
        variance_correction: bool = True,
        max_imputation_sd: Optional[float] = 3.0,
        predictive_noise: float = 0.0,
        min_std: float = 1e-6,
        detection_atol: float = 1e-8,
        refit_attempts: int = 3,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if method not in {"mean", "sample"}:
            raise ValueError("`method` must be 'mean' or 'sample'.")
        if e_step not in {"auto", "loo", "kfold", "in_sample"}:
            raise ValueError("`e_step` must be 'auto', 'loo', 'kfold' or 'in_sample'.")
        if direction not in {"right", "left"}:
            raise ValueError("`direction` must be 'right' or 'left'.")
        if not 0.0 < damping <= 1.0:
            raise ValueError("`damping` must lie in (0, 1].")
        if max_iter < 1:
            raise ValueError("`max_iter` must be at least 1.")
        if n_splits < 2:
            raise ValueError("`n_splits` must be at least 2.")

        self.base_model = base_model
        self.censoring_limit = censoring_limit
        self.censored_mask = censored_mask
        self.direction = direction
        self.method = method
        self.e_step = e_step
        self.n_splits = int(n_splits)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.variance_correction = bool(variance_correction)
        self.max_imputation_sd = max_imputation_sd
        self.predictive_noise = float(predictive_noise)
        self.min_std = float(min_std)
        self.detection_atol = float(detection_atol)
        self.refit_attempts = max(1, int(refit_attempts))
        self.random_state = random_state
        self.verbose = bool(verbose)

        self._rng = np.random.default_rng(random_state)
        self._X_train = None
        self._y_train = None
        self.em_result_: Optional[EMResult] = None

    # -- attribute forwarding --------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails. Guarding dunders and
        # ``base_model`` itself keeps copy/pickle from recursing while the
        # instance dict is still empty.
        if name.startswith("__") or name == "base_model":
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __repr__(self) -> str:
        return (
            f"CensoredEMGPModel(base_model={self.base_model!r}, "
            f"direction={self.direction!r}, method={self.method!r}, e_step={self.e_step!r})"
        )

    # -- public API -------------------------------------------------------- #
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        y_noise: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
        lower: Optional[Any] = None,
        upper: Optional[Any] = None,
    ) -> "CensoredEMGPModel":
        """Run Schmee-Hahn EM and leave ``base_model`` fitted to the result.

        The positional signature matches the rest of Mobius
        (``fit(X_train, y_train, y_noise)``), so the wrapper can be handed
        straight to an acquisition function.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,) or (n_samples, n_features)
            Sequences (HELM/FASTA) or feature vectors.
        y_train : array-like of shape (n_samples,)
            Target values with censoring sentinels in place.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation *variance* per row. The truncated-variance
            correction is added on top of whatever is supplied here.
        censored_mask : array-like of bool, default : None
            Overrides the mask given to the constructor.
        censoring_limit : float or array-like, default : None
            Overrides the limit given to the constructor.
        lower, upper : array-like, default : None
            Explicit per-row censoring intervals. Use these for interval
            censoring or for mixed left/right censoring in one data set.

        Returns
        -------
        self : `CensoredEMGPModel`
        """
        y = np.asarray(y_train, dtype=float).reshape(-1)
        n = len(y)
        X = X_train
        if hasattr(X, "__len__") and len(X) != n:
            raise ValueError(
                f"X_train has {len(X)} samples but y_train has {n} values."
            )

        spec = CensoringSpec.from_arrays(
            y,
            censored=censored_mask if censored_mask is not None else self.censored_mask,
            censoring_limit=(
                censoring_limit if censoring_limit is not None else self.censoring_limit
            ),
            direction=self.direction,
            lower=lower,
            upper=upper,
            detection_atol=self.detection_atol,
        )

        base_noise = (
            np.asarray(y_noise, dtype=float).reshape(-1) if y_noise is not None else None
        )
        if base_noise is not None and len(base_noise) != n:
            raise ValueError(f"y_noise has length {len(base_noise)}; expected {n}.")

        self._X_train = np.asarray(X_train) if not isinstance(X_train, np.ndarray) else X_train.copy()
        self._y_train = y.copy()

        if not spec.any_censored:
            self._fit_base(X, y, base_noise)
            self.em_result_ = EMResult(
                imputed_y=y.copy(),
                spec=spec,
                imputed_variance=np.zeros(n),
                converged=True,
                n_iter=0,
                max_change=0.0,
                history=[],
                e_step="none",
            )
            return self

        e_step = self._resolve_e_step()
        y_imp = self._initialise(y, spec)
        imputed_var = np.zeros(n)
        history: List[Dict[str, float]] = []
        max_change = float("inf")
        converged = False
        n_iter = 0

        for iteration in range(1, self.max_iter + 1):
            n_iter = iteration
            noise = self._assemble_noise(base_noise, imputed_var, spec)
            if not self._fit_base(X, y_imp, noise):
                warnings.warn(
                    f"base_model.fit failed at EM iteration {iteration} after "
                    f"{self.refit_attempts} attempts; stopping early and keeping the "
                    "last successful fit.",
                    RuntimeWarning,
                )
                n_iter = iteration - 1
                break

            mu, sd = self._e_step_predictions(X, y_imp, noise, spec, e_step)
            proposal, variance = self._truncated_update(mu, sd, spec)

            previous = y_imp[spec.censored].copy()
            y_imp[spec.censored] = (
                1.0 - self.damping
            ) * previous + self.damping * proposal
            imputed_var = np.zeros(n)
            imputed_var[spec.censored] = variance
            max_change = float(np.max(np.abs(y_imp[spec.censored] - previous)))

            loglik = float(
                np.sum(
                    censored_normal_logpdf(
                        y, mu, sd, spec.lower, spec.upper, observed=~spec.censored
                    )
                )
            )
            record = {
                "iteration": iteration,
                "max_change": max_change,
                "mean_imputed": float(y_imp[spec.censored].mean()),
                "censored_loglik": loglik,
                "mean_predictive_sd": float(sd[spec.censored].mean()),
            }
            history.append(record)
            if self.verbose:
                print(
                    f"[Schmee-Hahn] iter {iteration:3d}  max_change={max_change:.3e}  "
                    f"mean_imputed={record['mean_imputed']:.4f}  "
                    f"censored_loglik={loglik:.3f}"
                )
            if max_change < self.tol:
                converged = True
                break

        # The loop refits *before* updating, so the surrogate is one step stale.
        final_noise = self._assemble_noise(base_noise, imputed_var, spec)
        if not self._fit_base(X, y_imp, final_noise):
            # A BO loop should not die because the last marginal-likelihood
            # optimisation was unlucky. Back off towards progressively simpler
            # targets before giving up.
            fallbacks = (
                ("without the variance correction", y_imp, base_noise),
                ("at the censoring bounds", self._initialise(y, spec), base_noise),
            )
            for description, fallback_y, fallback_noise in fallbacks:
                if self._fit_base(X, fallback_y, fallback_noise):
                    warnings.warn(
                        "base_model.fit failed on the final Schmee-Hahn imputation; "
                        f"refitted {description}. Treat this fit as provisional and "
                        "consider standardising y_train or using TobitGPModel.",
                        RuntimeWarning,
                    )
                    y_imp = fallback_y
                    imputed_var = np.zeros(n)
                    break
            else:
                raise RuntimeError(
                    "base_model.fit failed on the final Schmee-Hahn imputation and on "
                    "every fallback. Try standardising y_train, reducing `damping`, or "
                    "switching to TobitGPModel."
                )

        self._y_train = y_imp.copy()
        self.em_result_ = EMResult(
            imputed_y=y_imp.copy(),
            spec=spec,
            imputed_variance=imputed_var,
            converged=converged,
            n_iter=n_iter,
            max_change=max_change,
            history=history,
            e_step=e_step,
        )

        if not converged and n_iter >= self.max_iter:
            warnings.warn(
                f"Schmee-Hahn did not converge in {self.max_iter} iterations "
                f"(last max_change={max_change:.3e}, {spec.fraction_censored:.0%} of rows "
                "censored). Imputation-based methods drift when censored points are "
                "numerous and mutually correlated; TobitGPModel optimises the censored "
                "likelihood directly and is the better choice here.",
                RuntimeWarning,
            )
        return self

    def predict(self, X_test: Any, y_noise: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with the underlying surrogate fitted to the completed data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples,) or (n_samples, n_features)
            Query points.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation variance at the query points, if the base model
            supports it.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Predictive mean.
        sigma : ndarray of shape (n_samples,)
            Predictive standard deviation.
        """
        if y_noise is not None and _accepts_y_noise_predict(self.base_model):
            return self.base_model.predict(X_test, y_noise=y_noise)
        return self.base_model.predict(X_test)

    def clone(self) -> "CensoredEMGPModel":
        """Return an unfitted deep copy, useful for repeated BO experiments."""
        cloned = copy.deepcopy(self)
        cloned.em_result_ = None
        cloned._X_train = None
        cloned._y_train = None
        cloned._rng = np.random.default_rng(self.random_state)
        return cloned

    # -- internals ---------------------------------------------------------- #
    def _resolve_e_step(self) -> str:
        if self.e_step != "auto":
            return self.e_step
        gp = getattr(self.base_model, "_model", None)
        has_exact_gp = (
            gpytorch is not None
            and gp is not None
            and isinstance(gp, gpytorch.models.ExactGP)
        ) or (
            # Not yet fitted, so ``_model`` is still None: infer from the class.
            gp is None and hasattr(self.base_model, "_likelihood")
        )
        return "loo" if has_exact_gp else "kfold"

    def _initialise(self, y: np.ndarray, spec: CensoringSpec) -> np.ndarray:
        """Start every censored row at its own bound - deterministic and conservative."""
        y_imp = y.copy()
        right = spec.is_right_censored
        left = spec.is_left_censored
        interval = spec.censored & ~right & ~left
        y_imp[right] = spec.lower[right]
        y_imp[left] = spec.upper[left]
        y_imp[interval] = 0.5 * (spec.lower[interval] + spec.upper[interval])
        return y_imp

    def _assemble_noise(
        self,
        base_noise: Optional[np.ndarray],
        imputed_var: np.ndarray,
        spec: CensoringSpec,
    ) -> Optional[np.ndarray]:
        if not self.variance_correction or not _accepts_y_noise(self.base_model):
            return base_noise
        if not np.any(imputed_var > 0.0):
            return base_noise
        noise = np.zeros(len(imputed_var)) if base_noise is None else base_noise.copy()
        noise = noise + imputed_var
        return noise

    def _fit_base(self, X: Any, y: np.ndarray, noise: Optional[np.ndarray]) -> bool:
        """Fit the wrapped model, retrying with jittered hyperparameters on failure."""
        supports_noise = noise is not None and _accepts_y_noise(self.base_model)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.refit_attempts):
            try:
                if supports_noise:
                    self.base_model.fit(X, y, y_noise=noise)
                else:
                    self.base_model.fit(X, y)
                return True
            except Exception as exc:  # botorch raises ModelFittingError, among others
                last_exc = exc
                self._jitter_hyperparameters(scale=0.1 * (attempt + 1))
        if self.verbose and last_exc is not None:
            print(f"[Schmee-Hahn] base_model.fit failed: {last_exc!r}")
        return False

    def _jitter_hyperparameters(self, scale: float) -> None:
        """Perturb kernel parameters so a retry starts from a different point."""
        kernel = getattr(self.base_model, "_kernel", None)
        if kernel is None or not hasattr(kernel, "parameters"):
            return
        with torch.no_grad():
            for parameter in kernel.parameters():
                if parameter.numel():
                    parameter.add_(torch.randn_like(parameter) * scale)

    def _e_step_predictions(
        self,
        X: Any,
        y_imp: np.ndarray,
        noise: Optional[np.ndarray],
        spec: CensoringSpec,
        e_step: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean/sd at every training row, out of sample where possible."""
        n = len(y_imp)
        if e_step == "loo":
            loo = exact_gp_loo_predictive(self.base_model)
            if loo is not None and len(loo[0]) == n:
                mu, sd = loo
                return mu, self._inflate(sd)
            e_step = "kfold"  # graceful degradation

        if e_step == "kfold":
            return self._kfold_predictions(X, y_imp, noise, n)

        prediction = self.base_model.predict(X)
        mu, sd = _extract_mean_std(prediction, self.min_std)
        return mu, self._inflate(sd)

    def _inflate(self, sd: np.ndarray) -> np.ndarray:
        sd = np.maximum(np.asarray(sd, dtype=float), 0.0)
        if self.predictive_noise:
            sd = np.sqrt(sd**2 + self.predictive_noise**2)
        return np.maximum(sd, self.min_std)

    def _kfold_predictions(
        self, X: Any, y_imp: np.ndarray, noise: Optional[np.ndarray], n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Out-of-fold predictions for models that hide their GP internals.

        The wrapped model is refitted on each training fold, so it is left in an
        arbitrary state; :meth:`fit` always refits on the full data afterwards.
        """
        order = self._rng.permutation(n)
        folds = np.array_split(order, min(self.n_splits, n))
        mu = np.empty(n)
        sd = np.empty(n)
        for fold in folds:
            if len(fold) == 0 or len(fold) == n:
                continue
            keep = np.setdiff1d(np.arange(n), fold, assume_unique=False)
            fold_noise = noise[keep] if noise is not None else None
            if not self._fit_base(_take(X, keep), y_imp[keep], fold_noise):
                raise RuntimeError("base_model.fit failed inside the k-fold E-step.")
            prediction = self.base_model.predict(_take(X, fold))
            fold_mu, fold_sd = _extract_mean_std(prediction, self.min_std)
            mu[fold] = fold_mu
            sd[fold] = fold_sd
        return mu, self._inflate(sd)

    def _truncated_update(
        self, mu: np.ndarray, sd: np.ndarray, spec: CensoringSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The E-step proper: conditional moments on the censoring interval."""
        mask = spec.censored
        mu_c, sd_c = mu[mask], sd[mask]
        lower, upper = spec.lower[mask], spec.upper[mask]

        mean, variance = truncated_normal_moments(
            mu_c, sd_c, lower=lower, upper=upper, min_sigma=self.min_std
        )

        if self.method == "sample":
            mean = _truncated_normal_sample(mu_c, sd_c, lower, upper, self._rng)

        if self.max_imputation_sd is not None:
            ceiling = np.where(
                np.isfinite(upper), upper, lower + self.max_imputation_sd * sd_c
            )
            floor = np.where(
                np.isfinite(lower), lower, upper - self.max_imputation_sd * sd_c
            )
            mean = np.clip(mean, floor, ceiling)

        mean = np.where(np.isfinite(lower), np.maximum(mean, lower), mean)
        mean = np.where(np.isfinite(upper), np.minimum(mean, upper), mean)
        return mean, np.maximum(variance, 0.0)


def _accepts_y_noise_predict(model: Any) -> bool:
    try:
        return "y_noise" in inspect.signature(model.predict).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _truncated_normal_sample(
    mu: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse-CDF draw from ``N(mu, sd^2)`` truncated to ``[lower, upper]``."""
    m = torch.as_tensor(np.asarray(mu, dtype=np.float64))
    s = torch.as_tensor(np.asarray(sd, dtype=np.float64)).clamp_min(1e-12)
    lo = torch.as_tensor(np.asarray(lower, dtype=np.float64))
    hi = torch.as_tensor(np.asarray(upper, dtype=np.float64))

    cdf_lo = torch.where(
        torch.isfinite(lo), torch.special.ndtr((lo - m) / s), torch.zeros_like(m)
    )
    cdf_hi = torch.where(
        torch.isfinite(hi), torch.special.ndtr((hi - m) / s), torch.ones_like(m)
    )
    u = torch.as_tensor(rng.uniform(size=len(mu)))
    p = cdf_lo + u * (cdf_hi - cdf_lo)
    eps = float(np.finfo(np.float64).eps)
    p = p.clamp(eps, 1.0 - eps)
    z = torch.erfinv(2.0 * p - 1.0) * float(np.sqrt(2.0))
    draw = (m + s * z).numpy()
    # Round-off can push a draw a hair outside its own support.
    draw = np.where(np.isfinite(lower), np.maximum(draw, lower), draw)
    draw = np.where(np.isfinite(upper), np.minimum(draw, upper), draw)
    return draw


#: Descriptive alias - the algorithm is Schmee & Hahn's, the container is a GP.
SchmeeHahnGPModel = CensoredEMGPModel
