#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Schmee-Hahn EM for right-censored targets
#
"""Schmee-Hahn EM imputation for right-censored targets.

Right censoring means the true value is known only to lie at or above a
detection limit: ``y_true >= limit``. This is the case for a potency reported
as ``pIC50 > 4.5`` because the compound was inactive at the top test
concentration.

The algorithm (Schmee & Hahn, *Technometrics* 21:417-432, 1979) alternates:

    **E-step**  replace each censored sentinel by ``E[Y | Y >= limit]`` under
                the current model,
    **M-step**  refit the model to the completed data.

Two details are load-bearing, and a naive transcription of the 1979 paper gets
both wrong on a Gaussian Process. Both are described in ``NOTES.md``; in short:

1. **The E-step must be out-of-sample.** A GP nearly interpolates its training
   data, so an *in-sample* prediction at a censored row returns the value just
   imputed there. The update then has a fixed point at the detection limit and
   the algorithm converges immediately to naive substitution while reporting
   success. This implementation uses the closed-form leave-one-out predictive
   (Rasmussen & Williams 2006, eq. 5.12), which costs one Cholesky
   factorisation for all *n* folds and, crucially, does not depend on the
   left-out target.

2. **Imputed rows must carry their own extra variance.** Substituting a
   conditional *mean* and treating it as a measurement throws away
   ``Var[Y | Y >= limit]``. On a GP the fitted noise then shrinks each
   iteration, which shrinks the E-step increments, which makes the marginal
   likelihood ill-conditioned - and in practice the fit eventually fails
   outright. Passing the truncated variance through as known observation noise
   restores the dispersion a proper EM would have kept.

Scope: right censoring only, and exact-GP surrogates only (``GPModel``,
``GPLLModel``, ``CachedGPLLModel``). Both restrictions are deliberate - the
leave-one-out identity above is what makes the algorithm work, and it needs an
exact GP.

Examples
--------
>>> import gpytorch
>>> from mobius.surrogate_models import CensoredEMGPModel, GPModel
>>> gp = GPModel(kernel=gpytorch.kernels.MaternKernel(nu=2.5))      # doctest: +SKIP
>>> model = CensoredEMGPModel(gp, censoring_limit=4.5)              # doctest: +SKIP
>>> model.fit(X_train, y_train)                                     # doctest: +SKIP
>>> mu, sigma = model.predict(X_test)                               # doctest: +SKIP
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .surrogate_model import _SurrogateModel

__all__ = ["CensoredEMGPModel", "EMResult", "truncated_normal_moments", "gp_loo_predictive"]

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Truncated-normal moments
# --------------------------------------------------------------------------- #
def truncated_normal_moments(
    mu: Any, sigma: Any, lower: Any, min_sigma: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and variance of ``N(mu, sigma^2)`` truncated to ``[lower, inf)``.

    Parameters
    ----------
    mu, sigma : array-like of shape (n,)
        Location and scale of the untruncated normal.
    lower : array-like of shape (n,)
        Truncation point.
    min_sigma : float, default : 1e-8
        Floor applied to ``sigma`` before dividing by it.

    Returns
    -------
    mean : ndarray of shape (n,)
        ``E[Y | Y >= lower]``, never below ``lower``.
    variance : ndarray of shape (n,)
        ``Var[Y | Y >= lower]``, always strictly positive.

    Notes
    -----
    Everything is evaluated in log space and in float64, so the inverse Mills
    ratio stays accurate far into the tail where ``phi(alpha)`` and
    ``1 - Phi(alpha)`` have both underflowed individually. Computing it as
    ``log(exp(-alpha^2 / 2) / sqrt(2 pi))`` instead - which is what the previous
    implementation did - returns ``-inf`` for ``|alpha| > 38``.
    """
    m = torch.as_tensor(np.asarray(mu, dtype=np.float64).reshape(-1))
    s = torch.as_tensor(np.asarray(sigma, dtype=np.float64).reshape(-1)).clamp_min(min_sigma)
    lo = torch.as_tensor(np.asarray(lower, dtype=np.float64).reshape(-1))

    alpha = (lo - m) / s
    # lambda = phi(alpha) / (1 - Phi(alpha)), via log phi and log Phi(-alpha).
    log_phi = -0.5 * alpha * alpha - _LOG_SQRT_2PI
    lam = torch.exp(log_phi - torch.special.log_ndtr(-alpha))

    mean = m + s * lam
    # 1 + alpha*lambda - lambda^2 -> alpha^-2 as alpha grows, and loses about
    # 2*log10(alpha) digits to cancellation on the way. float64 absorbs that
    # comfortably for any alpha a GP will produce.
    variance = (s * s * (1.0 + alpha * lam - lam * lam)).clamp_min(1e-300)

    # Round-off can nudge the mean a hair below its own support.
    mean = torch.maximum(mean, lo)
    return mean.numpy(), variance.numpy()


# --------------------------------------------------------------------------- #
# Leave-one-out predictive
# --------------------------------------------------------------------------- #
def gp_loo_predictive(model: Any, jitter: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form leave-one-out predictive of a fitted exact GP.

    For ``y ~ N(m, K_y)`` with ``K_y = K_f + sigma_n^2 I``, the LOO predictive
    for observation ``i`` is (Rasmussen & Williams 2006, eq. 5.12)

    .. math::

        \\mu_{-i} = y_i - \\frac{[K_y^{-1}(y - m)]_i}{[K_y^{-1}]_{ii}},
        \\qquad
        \\sigma^2_{-i} = \\frac{1}{[K_y^{-1}]_{ii}}

    One Cholesky factorisation gives all ``n`` folds. Expanding the numerator
    cancels the ``i``-th term exactly, so ``mu_{-i}`` does not depend on
    ``y_i`` - which is precisely what stops the E-step chasing its own
    imputation.

    Parameters
    ----------
    model : object
        Surrogate exposing ``_model`` (a ``gpytorch.models.ExactGP``) and
        ``_likelihood``, both already fitted.
    jitter : float, default : 1e-8
        Diagonal jitter added before factorising.

    Returns
    -------
    mean, sigma : ndarray of shape (n_train,)
        Leave-one-out predictive mean and standard deviation, aligned with the
        training set.

    Raises
    ------
    TypeError
        If ``model`` is not a fitted exact-GP surrogate.
    """
    import gpytorch

    gp = getattr(model, "_model", None)
    likelihood = getattr(model, "_likelihood", None)
    if gp is None or likelihood is None or not isinstance(gp, gpytorch.models.ExactGP):
        raise TypeError(
            f"{type(model).__name__} is not an exact-GP surrogate. CensoredEMGPModel "
            "needs the leave-one-out predictive, which requires GPModel, GPLLModel or "
            "CachedGPLLModel."
        )
    if getattr(gp, "train_inputs", None) is None or gp.train_targets is None:
        raise TypeError("The wrapped model must be fitted before its LOO predictive is taken.")

    was_training = gp.training, likelihood.training
    try:
        # In train mode an ExactGP returns the prior; through the likelihood
        # that is exactly K_y and the prior mean.
        gp.train()
        likelihood.train()
        with torch.no_grad():
            prior = likelihood(gp(*gp.train_inputs))
            K_y = prior.covariance_matrix.double()
            prior_mean = prior.mean.double().reshape(-1)
            y = gp.train_targets.double().reshape(-1)
    finally:
        gp.train(was_training[0])
        likelihood.train(was_training[1])

    n = K_y.shape[-1]
    eye = torch.eye(n, dtype=torch.float64, device=K_y.device)
    for scale in (1.0, 1e2, 1e4, 1e6):
        try:
            chol = torch.linalg.cholesky(K_y + jitter * scale * eye)
            break
        except Exception:  # pragma: no cover - only for pathological kernels
            chol = None
    if chol is None:  # pragma: no cover
        raise torch.linalg.LinAlgError("K_y is not positive definite even with jitter.")

    K_inv = torch.cholesky_inverse(chol)
    diag = torch.diagonal(K_inv).clamp_min(1e-300)
    mean = y - (K_inv @ (y - prior_mean)) / diag
    sigma = torch.sqrt(1.0 / diag)
    return mean.cpu().numpy(), sigma.cpu().numpy()


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class EMResult:
    """Diagnostics from one Schmee-Hahn run.

    Attributes
    ----------
    imputed_y : ndarray of shape (n_samples,)
        The completed target vector the surrogate was finally fitted to.
    censored_mask : ndarray of bool of shape (n_samples,)
        Which rows were treated as censored.
    lower_bounds : ndarray of shape (n_samples,)
        Per-row detection limit. Meaningful only where ``censored_mask``.
    imputed_variance : ndarray of shape (n_samples,)
        The extra observation variance applied to each row -
        ``Var[Y | Y >= limit]`` on censored rows at convergence, zero
        elsewhere. All zeros when ``variance_correction`` is disabled.
    converged : bool
        Whether the largest change in an imputation fell below ``tol``.
    n_iter : int
        Completed EM iterations.
    max_change : float
        Largest absolute change on the final iteration.
    history : list of dict
        One record per iteration with keys ``iteration``, ``max_change``,
        ``mean_imputed`` and ``noise``.
    """

    imputed_y: np.ndarray
    censored_mask: np.ndarray
    lower_bounds: np.ndarray
    imputed_variance: np.ndarray
    converged: bool
    n_iter: int
    max_change: float
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def n_censored(self) -> int:
        return int(self.censored_mask.sum())

    @property
    def fraction_censored(self) -> float:
        return float(self.censored_mask.mean()) if len(self.censored_mask) else 0.0


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class CensoredEMGPModel(_SurrogateModel):
    """Schmee-Hahn EM imputation wrapper for right-censored targets.

    A ``_SurrogateModel`` in its own right, with the positional signature the
    rest of Mobius uses, so it drops straight into an acquisition function.
    Attributes it does not define are forwarded to ``base_model``.

    Parameters
    ----------
    base_model : `_SurrogateModel`
        An exact-GP surrogate: ``GPModel``, ``GPLLModel`` or ``CachedGPLLModel``.
    censoring_limit : float or array-like of shape (n_samples,), default : None
        Detection limit, scalar or per row. Can instead be given to :meth:`fit`.
    censored_mask : array-like of bool of shape (n_samples,), default : None
        Explicit mask. When omitted, rows with ``y >= limit - detection_atol``
        are treated as censored.
    max_iter : int, default : 20
        Maximum EM iterations. Deliberately modest: imputations keep drifting
        upwards well after they have stopped improving, so running to a tight
        tolerance is worse than stopping early. See ``NOTES.md``.
    tol : float, default : 1e-4
        Convergence threshold on the largest change in an imputation.
    damping : float, default : 0.7
        Weight on the new proposal, in ``(0, 1]``. Below 1 this slows the
        positive feedback between censored points that are near neighbours.
    variance_correction : bool, default : True
        Pass ``Var[Y | Y >= limit]`` to the surrogate as known observation
        noise on imputed rows. Disable only to reproduce textbook Schmee-Hahn.
    max_imputation_sd : float or None, default : 3.0
        Cap each imputation at ``limit + max_imputation_sd * sigma``, so one
        badly extrapolated point cannot run away.
    refit_attempts : int, default : 3
        Retries, with jittered kernel hyperparameters, when a fit fails.
        Marginal-likelihood optimisation fails more often on data carrying a
        spike of identical values at the detection limit than on ordinary data.
    detection_atol : float, default : 1e-10
        Tolerance when inferring the censoring mask from the limit.
    verbose : bool, default : False
        Print per-iteration diagnostics.

    Attributes
    ----------
    em_result_ : `EMResult`
        Diagnostics from the last :meth:`fit`.

    Notes
    -----
    Targets should be standardised before fitting. ``GPModel`` optimises in
    float32 via L-BFGS and fails on raw targets carrying a sentinel pile-up
    often enough to matter; the retry logic covers the rest.

    Imputation is reliable up to roughly a third of rows censored. Past that the
    imputed points begin reinforcing one another and the iteration drifts; a
    warning is raised. Prefer a censored likelihood in that regime.
    """

    def __init__(
        self,
        base_model: Any,
        censoring_limit: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        max_iter: int = 20,
        tol: float = 1e-4,
        damping: float = 0.7,
        variance_correction: bool = True,
        max_imputation_sd: Optional[float] = 3.0,
        refit_attempts: int = 3,
        detection_atol: float = 1e-10,
        verbose: bool = False,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError("`damping` must be in (0, 1].")
        if max_iter < 1:
            raise ValueError("`max_iter` must be at least 1.")

        self.base_model = base_model
        self.censoring_limit = censoring_limit
        self.censored_mask = censored_mask
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.variance_correction = bool(variance_correction)
        self.max_imputation_sd = max_imputation_sd
        self.refit_attempts = max(1, int(refit_attempts))
        self.detection_atol = float(detection_atol)
        self.verbose = bool(verbose)

        self._X_train = None
        self._y_train = None
        self.em_result_: Optional[EMResult] = None

    # -- attribute forwarding ---------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. Guarding dunders and
        # `base_model` keeps copy and pickle from recursing while the instance
        # dictionary is still empty.
        if name.startswith("__") or name == "base_model":
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __repr__(self) -> str:
        return f"CensoredEMGPModel(base_model={self.base_model!r})"

    # -- public API --------------------------------------------------------- #
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        y_noise: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
    ) -> "CensoredEMGPModel":
        """Run Schmee-Hahn EM, leaving ``base_model`` fitted to the result.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,) or (n_samples, n_features)
            Sequences (HELM/FASTA) or feature vectors.
        y_train : array-like of shape (n_samples,)
            Targets, with the sentinel value in place on censored rows.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation *variance* per row. The truncated-variance
            correction is added on top of whatever is given here.
        censored_mask : array-like of bool, default : None
            Overrides the constructor mask.
        censoring_limit : float or array-like, default : None
            Overrides the constructor limit.

        Returns
        -------
        self : `CensoredEMGPModel`
        """
        y = np.asarray(y_train, dtype=float).reshape(-1)
        n = len(y)
        if hasattr(X_train, "__len__") and len(X_train) != n:
            raise ValueError(
                f"X_train has {len(X_train)} samples but y_train has {n} values."
            )

        mask, limits = self._resolve_censoring(y, censored_mask, censoring_limit)
        base_noise = None
        if y_noise is not None:
            base_noise = np.asarray(y_noise, dtype=float).reshape(-1)
            if len(base_noise) != n:
                raise ValueError(f"y_noise has length {len(base_noise)}; expected {n}.")

        self._X_train = X_train
        self._y_train = y.copy()

        if not mask.any():
            self._fit_base(X_train, y, base_noise, required=True)
            self.em_result_ = EMResult(
                y.copy(), mask, limits, np.zeros(n), True, 0, 0.0, []
            )
            return self

        # Start every censored row at its own bound: deterministic, conservative,
        # and identical to naive substitution, so any movement is EM's doing.
        y_imp = y.copy()
        y_imp[mask] = limits[mask]
        imputed_var = np.zeros(n)
        history: List[Dict[str, float]] = []
        max_change = float("inf")
        converged = False
        n_iter = 0

        for iteration in range(1, self.max_iter + 1):
            noise = self._assemble_noise(base_noise, imputed_var)
            if not self._fit_base(X_train, y_imp, noise, required=False):
                warnings.warn(
                    f"base_model.fit failed at EM iteration {iteration}; stopping early "
                    "and keeping the last successful fit.",
                    RuntimeWarning,
                )
                break
            n_iter = iteration

            # E-step, out of sample.
            mu, sigma = gp_loo_predictive(self.base_model)
            proposal, variance = self._e_step(mu[mask], sigma[mask], limits[mask])

            previous = y_imp[mask].copy()
            y_imp[mask] = (1.0 - self.damping) * previous + self.damping * proposal
            imputed_var = np.zeros(n)
            if self.variance_correction:
                # Only record what is actually applied, so `imputed_variance`
                # always describes the fit that was performed.
                imputed_var[mask] = variance
            max_change = float(np.max(np.abs(y_imp[mask] - previous)))

            record = {
                "iteration": iteration,
                "max_change": max_change,
                "mean_imputed": float(y_imp[mask].mean()),
                "noise": float(self.base_model._likelihood.noise.mean().detach().cpu()),
            }
            history.append(record)
            if self.verbose:
                print(
                    f"[Schmee-Hahn] iter {iteration:3d}  max_change={max_change:.3e}  "
                    f"mean_imputed={record['mean_imputed']:.4f}"
                )
            if max_change < self.tol:
                converged = True
                break

        # The loop fits *before* updating, so the surrogate is one step stale.
        self._fit_base(
            X_train, y_imp, self._assemble_noise(base_noise, imputed_var), required=True
        )

        self._y_train = y_imp.copy()
        self.em_result_ = EMResult(
            y_imp.copy(), mask, limits, imputed_var, converged, n_iter, max_change, history
        )

        if not converged and n_iter >= self.max_iter and mask.mean() > 1 / 3:
            warnings.warn(
                f"Schmee-Hahn did not converge in {self.max_iter} iterations "
                f"(last max_change={max_change:.3e}) with {mask.mean():.0%} of rows "
                "censored. Imputation drifts when censored points are numerous and "
                "mutually correlated; treat this fit as provisional.",
                RuntimeWarning,
            )
        return self

    def predict(self, X_test: Any, y_noise: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with the surrogate fitted to the completed data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples,) or (n_samples, n_features)
            Query points.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation variance at the query points.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Predictive mean.
        sigma : ndarray of shape (n_samples,)
            Predictive standard deviation.
        """
        if y_noise is not None:
            return self.base_model.predict(X_test, y_noise=y_noise)
        return self.base_model.predict(X_test)

    def clone(self) -> "CensoredEMGPModel":
        """Return an unfitted deep copy, useful in repeated BO experiments."""
        cloned = copy.deepcopy(self)
        cloned.em_result_ = None
        cloned._X_train = None
        cloned._y_train = None
        return cloned

    # -- internals ---------------------------------------------------------- #
    def _resolve_censoring(
        self, y: np.ndarray, censored_mask: Optional[Any], censoring_limit: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        limit = self.censoring_limit if censoring_limit is None else censoring_limit
        mask = self.censored_mask if censored_mask is None else censored_mask
        if limit is None:
            raise ValueError("Supply `censoring_limit` to the constructor or to fit().")

        limits = _broadcast(limit, n, "censoring_limit").astype(float)
        if mask is None:
            mask = np.isfinite(limits) & (y >= limits - self.detection_atol)
        else:
            mask = _broadcast(mask, n, "censored_mask").astype(bool)
        if np.any(mask & ~np.isfinite(limits)):
            raise ValueError("Every censored row needs a finite censoring_limit.")
        return mask, limits

    def _e_step(
        self, mu: np.ndarray, sigma: np.ndarray, lower: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Conditional moments of the censored rows, with a runaway cap."""
        mean, variance = truncated_normal_moments(mu, sigma, lower)
        if self.max_imputation_sd is not None:
            mean = np.minimum(mean, lower + self.max_imputation_sd * sigma)
        return np.maximum(mean, lower), variance

    def _assemble_noise(
        self, base_noise: Optional[np.ndarray], imputed_var: np.ndarray
    ) -> Optional[np.ndarray]:
        if not self.variance_correction or not np.any(imputed_var > 0.0):
            return base_noise
        noise = np.zeros(len(imputed_var)) if base_noise is None else base_noise.copy()
        return noise + imputed_var

    def _fit_base(
        self, X: Any, y: np.ndarray, noise: Optional[np.ndarray], required: bool
    ) -> bool:
        """Fit the wrapped model, retrying with jittered hyperparameters."""
        last: Optional[BaseException] = None
        for attempt in range(self.refit_attempts):
            try:
                if noise is not None:
                    self.base_model.fit(X, y, y_noise=noise)
                else:
                    self.base_model.fit(X, y)
                return True
            except Exception as exc:  # botorch raises ModelFittingError, among others
                last = exc
                self._jitter_hyperparameters(0.1 * (attempt + 1))
        if required:
            raise RuntimeError(
                f"base_model.fit failed after {self.refit_attempts} attempts ({last!r}). "
                "Standardise y_train, or reduce `damping`."
            )
        return False

    def _jitter_hyperparameters(self, scale: float) -> None:
        kernel = getattr(self.base_model, "_kernel", None)
        if kernel is None or not hasattr(kernel, "parameters"):
            return
        with torch.no_grad():
            for parameter in kernel.parameters():
                if parameter.numel():
                    parameter.add_(torch.randn_like(parameter) * scale)


def _broadcast(value: Any, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or check the length of an array."""
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(n, array.item())
    array = array.reshape(-1)
    if len(array) != n:
        raise ValueError(f"`{name}` has length {len(array)}; expected {n}.")
    return array
