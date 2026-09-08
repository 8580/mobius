#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Schmee-Hahn EM imputation wrapper for right-censored observations
#
"""Schmee-Hahn EM imputation wrapper for Mobius/GPyTorch surrogate models.

The wrapper is deliberately model-agnostic: the wrapped model must provide
``fit(X, y, y_noise=None)`` and ``predict(X)``. Predictions may be any of:

  * ``(mean, std)``
  * a GPyTorch ``MultivariateNormal`` with ``mean`` and ``variance``
  * an object with ``mean`` plus ``stddev`` or ``variance``

Right censoring means ``y_true >= censoring_limit``. This is appropriate for
log(IC50) values reported as, for example, "> log10(30 uM)".

Notes
-----
Schmee-Hahn EM alternates between (i) refitting the surrogate on the current
imputations and (ii) replacing each censored target by the conditional
expectation of a lower-truncated normal under the current posterior. When the
surrogate refits its hyperparameters at every step the iteration is not
guaranteed to be contractive: imputing upward raises the posterior mean, which
raises the next imputation. ``damping``, ``max_imputation_sd`` and
``max_total_shift`` bound that feedback loop, and non-convergence is reported
through a ``UserWarning`` rather than passing silently.
"""
from __future__ import annotations

import copy
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .surrogate_model import _SurrogateModel

__all__ = ["CensoredEMGPModel", "EMResult"]

# The exact log-ratio form of the inverse Mills ratio is accurate in float64
# until the standard normal pdf underflows, which happens at alpha ~= 38.6.
# Above that we fall back to the asymptotic expansion of lambda(alpha).
_MILLS_ASYMPTOTIC_ALPHA = 37.0

# Fitted state cleared by ``clone``; covers every Mobius surrogate model.
_FITTED_STATE_ATTRS = (
    "_model", "_likelihood", "_X_train", "_y_train", "_y_noise",
)


@dataclass
class EMResult:
    """Outcome of one Schmee-Hahn EM run.

    Attributes
    ----------
    imputed_y : ndarray of shape (n_samples,)
        Targets the wrapped model was finally fitted on.
    observed_y : ndarray of shape (n_samples,)
        The raw targets as supplied, before any imputation.
    censored_mask : ndarray of bool of shape (n_samples,)
        True for rows treated as right-censored.
    lower_bounds : ndarray of shape (n_samples,)
        Per-row censoring limit (finite only where required).
    converged : bool
        Whether the undamped EM step fell below ``tol``.
    n_iter : int
        Number of EM iterations executed.
    max_change : float
        Largest undamped step on the final iteration.
    observed_y : ndarray of shape (n_samples,)
        Alias kept for clarity; see above.
    fit_failure : str or None
        Repr of the exception that stopped EM early, if any.
    """

    imputed_y: np.ndarray
    censored_mask: np.ndarray
    lower_bounds: np.ndarray
    converged: bool
    n_iter: int
    max_change: float
    observed_y: np.ndarray = field(default=None)
    fit_failure: Optional[str] = field(default=None)

    @property
    def n_censored(self) -> int:
        return int(np.count_nonzero(self.censored_mask))


class CensoredEMGPModel(_SurrogateModel):
    """Drop-in wrapper adding right-censored EM imputation to a surrogate model.

    Parameters
    ----------
    base_model : object
        A Mobius ``GPModel`` or compatible surrogate exposing ``fit`` and
        ``predict``.
    censoring_limit : float or array-like, default : None
        Scalar or per-row lower bound. If provided and ``censored_mask`` is
        not, rows satisfying ``y >= censoring_limit - detection_atol`` are
        treated as right-censored sentinel observations.
    censored_mask : array-like of bool, default : None
        Optional explicit mask. Can instead be passed to ``fit``.
    method : {'mean', 'sample'}, default : 'mean'
        ``'mean'`` for deterministic Schmee-Hahn conditional means,
        ``'sample'`` for stochastic truncated-normal draws.
    max_iter : int, default : 30
        Maximum number of EM iterations. Must be >= 1.
    tol : float, default : 1e-4
        Convergence threshold on the largest *undamped* imputation step, so
        that its meaning does not change with ``damping``.
    damping : float, default : 0.7
        Update weight in (0, 1]. Values around 0.5-0.8 stabilise EM.
    min_std : float, default : 1e-6
        Floor on the predictive standard deviation. Must be > 0.
    predictive_noise : float, default : 0.
        Extra standard deviation added in quadrature. Set this to the assay
        residual SD on the same log scale if the wrapped model returns latent
        rather than observation uncertainty.
    detection_atol : float, default : 1e-10
        Tolerance used when auto-detecting sentinel values at the limit.
    max_imputation_sd : float or None, default : 6.0
        Cap each imputation at ``limit + max_imputation_sd * sd``.
    max_total_shift : float or None, default : 10.0
        Cap the cumulative displacement of any imputation from its limit, in
        units of the initial predictive sd. Bounds the EM feedback loop even
        when the predictive sd itself grows. Set to None to disable.
    restart_hyperparameters : bool, default : True
        Restore the wrapped model's kernel/likelihood hyperparameters to their
        pre-EM values before every refit. Required for `GPModel`, whose kernel
        instance persists across ``fit`` calls: restarting ``fit_gpytorch_mll``
        from an already-converged optimum makes ``scipy_minimize`` terminate
        ABNORMAL and botorch raise ``ModelFittingError``. Because EM changes
        the targets less and less as it converges, that failure becomes *more*
        likely the closer EM gets to its fixed point.
    on_non_convergence : {'warn', 'raise', 'ignore'}, default : 'warn'
        What to do when ``max_iter`` is exhausted without meeting ``tol``.
    on_fit_failure : {'stop', 'raise'}, default : 'stop'
        What to do when the wrapped model raises during an EM refit.
        ``'stop'`` ends EM at the last imputation that fitted successfully and
        warns; ``'raise'`` propagates. A failure on the very first iteration
        always propagates, because there is no good state to fall back to.
        This matters in practice: ``fit_gpytorch_mll`` does fail on some
        imputed target vectors (the censored rows collapse toward a common
        value, which is a hard target for an RBF GP), and without containment
        one bad iteration destroys the whole run.
    random_state : int or None, default : None
        Seed for ``method='sample'``.
    verbose : bool, default : False
        Print per-iteration progress.

    Attributes
    ----------
    em_result_ : `EMResult` or None
        Result of the last successful ``fit``. None before fitting, and reset
        to None if a ``fit`` call fails part way through.

    Examples
    --------
    >>> model = CensoredEMGPModel(GPModel(kernel), censoring_limit=np.log10(30e-6))
    >>> model.fit(sequences, pic50_values)
    >>> mu, sigma = model.predict(new_sequences)
    """

    def __init__(
        self,
        base_model: Any,
        *,
        censoring_limit: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        method: str = "mean",
        max_iter: int = 30,
        tol: float = 1e-4,
        damping: float = 0.7,
        min_std: float = 1e-6,
        predictive_noise: float = 0.0,
        detection_atol: float = 1e-10,
        max_imputation_sd: Optional[float] = 6.0,
        max_total_shift: Optional[float] = 10.0,
        on_non_convergence: str = "warn",
        on_fit_failure: str = "stop",
        restart_hyperparameters: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if method not in {"mean", "sample"}:
            raise ValueError("method must be 'mean' or 'sample'")
        if on_non_convergence not in {"warn", "raise", "ignore"}:
            raise ValueError("on_non_convergence must be 'warn', 'raise' or 'ignore'")
        if on_fit_failure not in {"stop", "raise"}:
            raise ValueError("on_fit_failure must be 'stop' or 'raise'")
        damping = float(damping)
        if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
            raise ValueError("damping must be a finite value in (0, 1]")
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        min_std = float(min_std)
        if not np.isfinite(min_std) or min_std <= 0.0:
            raise ValueError("min_std must be a finite value > 0")
        if float(predictive_noise) < 0.0:
            raise ValueError("predictive_noise must be >= 0")

        self.base_model = base_model
        self.censoring_limit = censoring_limit
        self.censored_mask = censored_mask
        self.method = method
        self.max_iter = max_iter
        self.tol = float(tol)
        self.damping = damping
        self.min_std = min_std
        self.predictive_noise = float(predictive_noise)
        self.detection_atol = float(detection_atol)
        self.max_imputation_sd = max_imputation_sd
        self.max_total_shift = max_total_shift
        self.on_non_convergence = on_non_convergence
        self.on_fit_failure = on_fit_failure
        self.restart_hyperparameters = bool(restart_hyperparameters)
        self._hp_snapshot = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.em_result_: Optional[EMResult] = None

    # ------------------------------------------------------------------
    # attribute delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Preserve access to wrapped-model attributes expected by Mobius
        # (``_pretrained_model``, ``device``, ``score``, ...). Guard against
        # recursion when ``base_model`` itself is absent, which happens during
        # unpickling before ``__dict__`` is restored.
        if name.startswith("__") or name in {"base_model", "em_result_"}:
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base_model")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "unfitted"
        if self.em_result_ is not None:
            state = (
                f"n_censored={self.em_result_.n_censored}, "
                f"n_iter={self.em_result_.n_iter}, "
                f"converged={self.em_result_.converged}"
            )
        return (
            f"CensoredEMGPModel(base_model={self.base_model!r}, "
            f"method='{self.method}', {state})"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_1d(x: Any, n: Optional[int] = None, name: str = "array") -> np.ndarray:
        a = np.asarray(x)
        if a.ndim == 0:
            if n is None:
                return a.reshape(1)
            return np.full(n, a.item())
        a = a.reshape(-1)
        if n is not None and len(a) != n:
            raise ValueError(f"{name} has length {len(a)}; expected {n}")
        return a

    @staticmethod
    def _validate_targets(y: Any) -> Tuple[np.ndarray, tuple]:
        """Flatten y to 1-D, rejecting genuinely multi-output targets."""
        arr = np.asarray(y)
        shape = arr.shape
        if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[1] != 1):
            raise ValueError(
                "CensoredEMGPModel handles a single target column; got y with "
                f"shape {shape}. Wrap one CensoredEMGPModel per objective for "
                "multi-objective runs."
            )
        return arr.reshape(-1).astype(float), shape

    def _resolve_censoring(
        self, y: np.ndarray, censored_mask: Optional[Any], censoring_limit: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        lim_arg = self.censoring_limit if censoring_limit is None else censoring_limit
        mask_arg = self.censored_mask if censored_mask is None else censored_mask
        if lim_arg is None:
            raise ValueError("Supply censoring_limit in the constructor or fit().")
        limits = self._as_1d(lim_arg, n, "censoring_limit").astype(float)

        if mask_arg is None:
            finite = np.isfinite(limits)
            mask = finite & (y >= limits - self.detection_atol)
            strictly_above = finite & (y > limits + self.detection_atol)
            if np.any(strictly_above):
                warnings.warn(
                    f"{int(np.count_nonzero(strictly_above))} observation(s) lie "
                    "strictly above the censoring limit and were auto-detected as "
                    "right-censored; their values will be replaced by imputations. "
                    "Pass an explicit censored_mask if they are real measurements.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            mask = self._as_1d(mask_arg, n, "censored_mask").astype(bool)

        if np.any(mask & ~np.isfinite(limits)):
            raise ValueError("Every censored row requires a finite lower bound.")
        return mask, limits

    def _base_accepts_y_noise(self) -> bool:
        """Not every Mobius surrogate takes y_noise (RFModel does not)."""
        cached = self.__dict__.get("_accepts_y_noise")
        if cached is None:
            try:
                params = inspect.signature(self.base_model.fit).parameters
            except (TypeError, ValueError):  # builtins / C extensions
                cached = False
            else:
                cached = "y_noise" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            self.__dict__["_accepts_y_noise"] = bool(cached)
        return self.__dict__["_accepts_y_noise"]

    def _hp_modules(self):
        """Sub-modules whose hyperparameters EM must restart from."""
        for attr in ("_kernel", "_likelihood", "covar_module", "mean_module"):
            mod = getattr(self.base_model, attr, None)
            if mod is not None and hasattr(mod, "state_dict") and hasattr(mod, "load_state_dict"):
                yield attr, mod

    def _snapshot_hyperparameters(self) -> None:
        if not self.restart_hyperparameters:
            return
        snap = {}
        for attr, mod in self._hp_modules():
            try:
                snap[attr] = copy.deepcopy(mod.state_dict())
            except Exception:  # noqa: BLE001 - snapshotting is best effort
                continue
        self._hp_snapshot = snap or None

    def _restore_hyperparameters(self) -> None:
        if not self.restart_hyperparameters or not self._hp_snapshot:
            return
        for attr, state in self._hp_snapshot.items():
            mod = getattr(self.base_model, attr, None)
            if mod is None:
                continue
            try:
                mod.load_state_dict(copy.deepcopy(state))
            except Exception:  # noqa: BLE001 - restoring is best effort
                continue

    def _fit_base(self, X: Any, y: np.ndarray, y_noise: Optional[Any],
                  fit_kwargs: dict) -> None:
        self._restore_hyperparameters()
        if y_noise is None and not self._base_accepts_y_noise():
            self.base_model.fit(X, y, **fit_kwargs)
        elif y_noise is None:
            self.base_model.fit(X, y, None, **fit_kwargs)
        else:
            if not self._base_accepts_y_noise():
                raise TypeError(
                    f"{type(self.base_model).__name__}.fit does not accept "
                    "y_noise, but y_noise was supplied."
                )
            self.base_model.fit(X, y, y_noise, **fit_kwargs)

    @staticmethod
    def _select_rows(X: Any, mask: np.ndarray) -> Any:
        """Row-select from X for any container Mobius may hand us."""
        idx = np.flatnonzero(mask)
        if hasattr(X, "iloc"):                       # pandas
            return X.iloc[idx]
        if torch.is_tensor(X):
            return X[torch.as_tensor(idx, device=X.device)]
        try:
            arr = np.asarray(X)
        except Exception as exc:                      # ragged / exotic input
            raise TypeError(
                "X must be array-like with one row per observation; could not "
                f"convert it to an array ({exc.__class__.__name__})."
            ) from exc
        if arr.dtype == object and arr.ndim == 1 and not isinstance(X, np.ndarray):
            # ragged nested sequences become a 1-D object array
            if any(isinstance(v, (list, tuple, np.ndarray)) for v in arr):
                raise TypeError(
                    "X appears to be a ragged nested sequence; supply a "
                    "rectangular array or a list of sequence strings."
                )
        if arr.ndim == 0:
            raise ValueError("X must have at least one dimension.")
        return arr[idx]

    # ------------------------------------------------------------------
    # truncated normal maths
    # ------------------------------------------------------------------
    @staticmethod
    def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x.square()) / np.sqrt(2.0 * np.pi)

    @staticmethod
    def _normal_log_pdf(x: torch.Tensor) -> torch.Tensor:
        # Closed form; does not underflow the way log(exp(-x**2/2)) does.
        return -0.5 * x.square() - 0.5 * float(np.log(2.0 * np.pi))

    @staticmethod
    def _normal_log_survival(x: torch.Tensor) -> torch.Tensor:
        # log Phi(-x), stable even when x is large and positive.
        return torch.special.log_ndtr(-x)

    def _inverse_mills(self, alpha: torch.Tensor) -> torch.Tensor:
        """lambda(alpha) = phi(alpha) / (1 - Phi(alpha)), stable everywhere."""
        log_lambda = self._normal_log_pdf(alpha) - self._normal_log_survival(alpha)
        mills = torch.exp(log_lambda)
        # Beyond the float64 pdf underflow point, use the asymptotic series
        # lambda ~ a + 1/a - 2/a**3, which is accurate to ~1e-9 by then.
        a = alpha.clamp_min(1.0)
        asymptotic = a + torch.reciprocal(a) - 2.0 * torch.reciprocal(a.pow(3))
        return torch.where(alpha > _MILLS_ASYMPTOTIC_ALPHA, asymptotic, mills)

    def _truncated_mean(self, mu: np.ndarray, sd: np.ndarray, lower: np.ndarray) -> np.ndarray:
        dtype = torch.float64
        m = torch.as_tensor(np.asarray(mu), dtype=dtype)
        s = torch.as_tensor(np.asarray(sd), dtype=dtype).clamp_min(self.min_std)
        low = torch.as_tensor(np.asarray(lower), dtype=dtype)
        alpha = (low - m) / s
        return (m + s * self._inverse_mills(alpha)).cpu().numpy()

    def _truncated_sample(self, mu: np.ndarray, sd: np.ndarray, lower: np.ndarray) -> np.ndarray:
        # Inverse-CDF sampling implemented with torch, avoiding a SciPy dependency.
        dtype = torch.float64
        m = torch.as_tensor(np.asarray(mu), dtype=dtype)
        s = torch.as_tensor(np.asarray(sd), dtype=dtype).clamp_min(self.min_std)
        low = torch.as_tensor(np.asarray(lower), dtype=dtype)
        alpha = (low - m) / s
        cdf_lo = 0.5 * (1.0 + torch.erf(alpha / np.sqrt(2.0)))
        u = torch.as_tensor(self.rng.uniform(size=int(m.numel())), dtype=dtype)
        p = cdf_lo + u * (1.0 - cdf_lo)
        p = p.clamp(torch.finfo(dtype).eps, 1.0 - torch.finfo(dtype).eps)
        z = np.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
        out = (m + s * z).cpu().numpy()
        # Guard the extreme tail, where the inverse CDF loses resolution.
        return np.maximum(out, np.asarray(lower, dtype=float))

    # ------------------------------------------------------------------
    # prediction unpacking
    # ------------------------------------------------------------------
    def _extract_mean_std(self, prediction: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(prediction, tuple) and len(prediction) >= 2:
            # Mobius/sklearn-style predict convention is (mean, std).
            mean, std = prediction[0], prediction[1]
        elif hasattr(prediction, "mean"):
            mean = prediction.mean
            if hasattr(prediction, "stddev"):
                std = prediction.stddev
            elif hasattr(prediction, "variance"):
                var = prediction.variance
                std = torch.sqrt(var.clamp_min(0.0)) if torch.is_tensor(var) \
                    else np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0))
            else:
                raise TypeError("Prediction has mean but no stddev/variance.")
        else:
            raise TypeError("Unsupported prediction type from base_model.predict().")

        if torch.is_tensor(mean):
            mean = mean.detach().cpu().numpy()
        if torch.is_tensor(std):
            std = std.detach().cpu().numpy()
        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.asarray(std, dtype=float).reshape(-1)
        if len(mean) != len(std):
            raise ValueError("Prediction mean and std lengths differ.")
        std = np.sqrt(np.maximum(std, 0.0) ** 2 + self.predictive_noise**2)
        return mean, np.maximum(std, self.min_std)

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        y_noise: Optional[Any] = None,
        *,
        censored_mask: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> "CensoredEMGPModel":
        """Fit the wrapped surrogate under right-censored observations.

        ``y_noise`` is accepted positionally so the signature matches
        ``mobius._AcquisitionFunction.fit``, which always passes it.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,) or (n_samples, n_features)
            Sequences (HELM/FASTA) or feature vectors.
        y_train : array-like of shape (n_samples,) or (n_samples, 1)
            Target values, with censored rows reported at their limit.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation noise (variance), forwarded to the base model.
        censored_mask : array-like of bool, default : None
            Overrides the constructor value for this call.
        censoring_limit : float or array-like, default : None
            Overrides the constructor value for this call.

        Returns
        -------
        self : `CensoredEMGPModel`
        """
        # Invalidate first: a failure part way through must not leave a stale
        # result that no longer matches the state of the wrapped model.
        self.em_result_ = None

        y1, original_shape = self._validate_targets(y_train)
        if not np.all(np.isfinite(y1)):
            raise ValueError("y_train contains non-finite values.")
        mask, limits = self._resolve_censoring(y1, censored_mask, censoring_limit)

        def _shaped(values: np.ndarray) -> np.ndarray:
            return values.reshape(original_shape) if len(original_shape) == 2 else values

        if not np.any(mask):
            self._fit_base(X_train, _shaped(y1), y_noise, fit_kwargs)
            self.em_result_ = EMResult(y1.copy(), mask, limits, True, 0, 0.0, y1.copy())
            return self

        self._snapshot_hyperparameters()
        X_cens = self._select_rows(X_train, mask)
        lower = limits[mask]

        # Initialisation at thresholds is conservative and deterministic.
        y_imp = y1.copy()
        y_imp[mask] = lower
        max_change = np.inf
        converged = False
        iteration = 0
        initial_sd: Optional[np.ndarray] = None

        last_good: Optional[np.ndarray] = None
        fit_failure: Optional[BaseException] = None

        for iteration in range(1, self.max_iter + 1):
            try:
                self._fit_base(X_train, _shaped(y_imp), y_noise, fit_kwargs)
            except Exception as exc:  # noqa: BLE001 - surrogate-specific
                if last_good is None or self.on_fit_failure == "raise":
                    raise
                fit_failure = exc
                iteration -= 1
                y_imp = last_good
                break
            last_good = y_imp.copy()
            mu, sd = self._extract_mean_std(self.base_model.predict(X_cens))
            if len(mu) != len(lower):
                raise ValueError(
                    f"base_model.predict returned {len(mu)} values for "
                    f"{len(lower)} censored rows."
                )
            if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))):
                raise ValueError(
                    "base_model.predict returned non-finite mean or standard "
                    f"deviation at EM iteration {iteration}."
                )
            if initial_sd is None:
                initial_sd = sd.copy()

            if self.method == "mean":
                proposed = self._truncated_mean(mu, sd, lower)
            else:
                proposed = self._truncated_sample(mu, sd, lower)

            proposed = np.maximum(proposed, lower)
            if self.max_imputation_sd is not None:
                proposed = np.minimum(proposed, lower + self.max_imputation_sd * sd)
            if self.max_total_shift is not None:
                proposed = np.minimum(proposed, lower + self.max_total_shift * initial_sd)

            old = y_imp[mask].copy()
            # Convergence is judged on the undamped step so that `tol` means
            # the same thing at every damping value.
            max_change = float(np.max(np.abs(proposed - old))) if len(old) else 0.0
            y_imp[mask] = (1.0 - self.damping) * old + self.damping * proposed

            if self.verbose:
                print(f"Censored EM iteration {iteration}: max change={max_change:.6g}")
            if max_change < self.tol:
                converged = True
                break

        # Essential: leave the wrapped surrogate fitted to the final imputation.
        # Skipped when the last damped step moved nothing (the in-loop fit
        # already used these targets) or when EM stopped on a fit failure (the
        # model is already fitted to `last_good`).
        if fit_failure is None and np.any(y_imp[mask] != old):
            try:
                self._fit_base(X_train, _shaped(y_imp), y_noise, fit_kwargs)
            except Exception as exc:  # noqa: BLE001
                if self.on_fit_failure == "raise":
                    raise
                fit_failure = exc
                y_imp = last_good

        if fit_failure is not None:
            converged = False
            warnings.warn(
                f"Censored EM stopped at iteration {iteration}: the wrapped "
                f"model raised {type(fit_failure).__name__}({fit_failure}). The "
                "surrogate is left fitted to the last imputation that succeeded. "
                "Try a smaller `damping`, `method='sample'`, or a different "
                "kernel.",
                UserWarning,
                stacklevel=2,
            )

        if not converged and fit_failure is None:
            msg = (
                f"Censored EM did not converge in {iteration} iterations "
                f"(max change {max_change:.3g} > tol {self.tol:.3g}). The "
                "surrogate is fitted to a non-stationary imputation; consider "
                "lowering `damping`, raising `max_iter`, or tightening "
                "`max_total_shift`."
            )
            if self.on_non_convergence == "raise":
                raise RuntimeError(msg)
            if self.on_non_convergence == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

        self.em_result_ = EMResult(
            y_imp.copy(), mask, limits, converged, iteration, max_change, y1.copy(),
            None if fit_failure is None else f"{type(fit_failure).__name__}: {fit_failure}",
        )
        return self

    def predict(self, X_test: Any, *args: Any, **kwargs: Any) -> Any:
        """Predict with the wrapped surrogate. See the base model's ``predict``."""
        return self.base_model.predict(X_test, *args, **kwargs)

    # ------------------------------------------------------------------
    # surrogate surface expected by mobius acquisition functions
    # ------------------------------------------------------------------
    @property
    def X_train(self):
        """Training inputs held by the wrapped model."""
        return self.base_model.X_train

    @property
    def y_train(self):
        """Targets the wrapped model was fitted on, i.e. the *imputed* values.

        mobius acquisition functions read this to form ``best_f``; see
        ``incumbent`` for the observation-only alternative.
        """
        return self.base_model.y_train

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    @property
    def observed_y_(self) -> Optional[np.ndarray]:
        """Raw targets as supplied to the last ``fit`` (None before fitting)."""
        return None if self.em_result_ is None else self.em_result_.observed_y

    def incumbent(self, maximize: bool = True, use_imputed: bool = False) -> float:
        """Incumbent value for an improvement-based acquisition function.

        By default this uses the *observed* targets, so a censored sentinel can
        never inflate ``best_f`` above anything actually measured. Note that
        ``self.y_train`` delegates to the wrapped model and therefore returns
        the imputed targets; mobius acquisition functions read that attribute
        directly, so pass ``use_imputed=True`` to reproduce their behaviour.
        """
        if self.em_result_ is None:
            raise ValueError("This model instance is not fitted yet.")
        y = self.em_result_.imputed_y if use_imputed else self.em_result_.observed_y
        return float(np.max(y) if maximize else np.min(y))

    def clone(self) -> "CensoredEMGPModel":
        """Return an unfitted deep copy, useful in repeated BO experiments.

        The wrapped model is rebuilt from a pre-fit copy so no GPyTorch state
        is carried over, and the RNG is re-seeded independently so replicate
        runs in ``method='sample'`` are not perfectly correlated.
        """
        base_copy = copy.deepcopy(self.base_model)
        for attr in _FITTED_STATE_ATTRS:
            if hasattr(base_copy, attr):
                try:
                    setattr(base_copy, attr, None)
                except AttributeError:  # read-only property
                    pass
        reset = getattr(base_copy, "reset", None)
        if callable(reset):
            reset()
        # A deep-copied kernel keeps its fitted hyperparameters, which is
        # exactly the state that makes the next fit_gpytorch_mll fail.
        if self._hp_snapshot:
            for attr, state in self._hp_snapshot.items():
                mod = getattr(base_copy, attr, None)
                if mod is not None:
                    try:
                        mod.load_state_dict(copy.deepcopy(state))
                    except Exception:  # noqa: BLE001
                        continue
        cloned = CensoredEMGPModel(
            base_copy,
            censoring_limit=copy.deepcopy(self.censoring_limit),
            censored_mask=copy.deepcopy(self.censored_mask),
            method=self.method,
            max_iter=self.max_iter,
            tol=self.tol,
            damping=self.damping,
            min_std=self.min_std,
            predictive_noise=self.predictive_noise,
            detection_atol=self.detection_atol,
            max_imputation_sd=self.max_imputation_sd,
            max_total_shift=self.max_total_shift,
            on_non_convergence=self.on_non_convergence,
            on_fit_failure=self.on_fit_failure,
            restart_hyperparameters=self.restart_hyperparameters,
            random_state=None,
            verbose=self.verbose,
        )
        cloned._hp_snapshot = None
        cloned.rng = np.random.default_rng(self.rng.integers(0, 2**63 - 1))
        return cloned
