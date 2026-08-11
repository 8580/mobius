"""Schmee-Hahn EM imputation wrapper for Mobius/GPyTorch surrogate models.

The wrapper is deliberately model-agnostic: the wrapped model must provide
``fit(X, y)`` and ``predict(X)``. Predictions may be any of:
  * (mean, std)
  * a GPyTorch MultivariateNormal with ``mean`` and ``variance``
  * an object with ``mean`` plus ``stddev`` or ``variance``

Right censoring means y_true >= censoring_limit. This is appropriate for
log(IC50) values reported as, for example, "> log10(30 uM)".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import copy
import numpy as np
import torch


@dataclass
class EMResult:
    imputed_y: np.ndarray
    censored_mask: np.ndarray
    lower_bounds: np.ndarray
    converged: bool
    n_iter: int
    max_change: float


class CensoredEMGPModel:
    """Drop-in wrapper adding right-censored EM imputation to a GP model.

    Parameters
    ----------
    base_model:
        A Mobius ``GPModel`` or compatible GPyTorch-backed surrogate.
    censoring_limit:
        Scalar or per-row lower bound. If provided and ``censored_mask`` is not,
        rows satisfying ``y >= censoring_limit - detection_atol`` are treated as
        right-censored sentinel observations.
    censored_mask:
        Optional boolean mask. Can instead be passed to ``fit``.
    method:
        ``"mean"`` for deterministic Schmee-Hahn conditional means, or
        ``"sample"`` for stochastic truncated-normal draws.
    predictive_noise:
        Extra standard deviation added in quadrature. Set this to the assay
        residual SD on the same log scale if the wrapped model returns latent
        rather than observation uncertainty.
    damping:
        Update weight in (0, 1]. Values around 0.5-0.8 can stabilise EM.
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
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if method not in {"mean", "sample"}:
            raise ValueError("method must be 'mean' or 'sample'")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.base_model = base_model
        self.censoring_limit = censoring_limit
        self.censored_mask = censored_mask
        self.method = method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.min_std = float(min_std)
        self.predictive_noise = float(predictive_noise)
        self.detection_atol = float(detection_atol)
        self.max_imputation_sd = max_imputation_sd
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.em_result_: Optional[EMResult] = None

    def __getattr__(self, name: str) -> Any:
        # Preserve access to wrapped-model attributes expected by Mobius.
        if name == "base_model":
            raise AttributeError(name)
        return getattr(self.base_model, name)

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
            mask = np.isfinite(limits) & (y >= limits - self.detection_atol)
        else:
            mask = self._as_1d(mask_arg, n, "censored_mask").astype(bool)
        if np.any(mask & ~np.isfinite(limits)):
            raise ValueError("Every censored row requires a finite lower bound.")
        return mask, limits

    @staticmethod
    def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * x.square()) / np.sqrt(2.0 * np.pi)

    @staticmethod
    def _normal_log_survival(x: torch.Tensor) -> torch.Tensor:
        # log Phi(-x), stable even when x is large and positive.
        return torch.special.log_ndtr(-x)

    def _truncated_mean(self, mu: np.ndarray, sd: np.ndarray, lower: np.ndarray) -> np.ndarray:
        dtype = torch.float64
        m = torch.as_tensor(mu, dtype=dtype)
        s = torch.as_tensor(sd, dtype=dtype).clamp_min(self.min_std)
        l = torch.as_tensor(lower, dtype=dtype)
        alpha = (l - m) / s
        log_lambda = torch.log(self._normal_pdf(alpha)) - self._normal_log_survival(alpha)
        # The ratio can overflow only for pathological scales; alpha + 1/alpha
        # is the leading asymptotic form of the inverse Mills ratio.
        mills = torch.exp(torch.clamp(log_lambda, max=700.0))
        asymptotic = alpha + torch.reciprocal(alpha.clamp_min(1e-12))
        mills = torch.where(alpha > 8.0, asymptotic, mills)
        out = m + s * mills
        return out.cpu().numpy()

    def _truncated_sample(self, mu: np.ndarray, sd: np.ndarray, lower: np.ndarray) -> np.ndarray:
        # Inverse-CDF sampling implemented with torch, avoiding a SciPy dependency.
        dtype = torch.float64
        m = torch.as_tensor(mu, dtype=dtype)
        s = torch.as_tensor(sd, dtype=dtype).clamp_min(self.min_std)
        l = torch.as_tensor(lower, dtype=dtype)
        alpha = (l - m) / s
        cdf_lo = 0.5 * (1.0 + torch.erf(alpha / np.sqrt(2.0)))
        u_np = self.rng.uniform(size=len(mu))
        u = torch.as_tensor(u_np, dtype=dtype)
        p = cdf_lo + u * (1.0 - cdf_lo)
        p = p.clamp(torch.finfo(dtype).eps, 1.0 - torch.finfo(dtype).eps)
        z = np.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
        return (m + s * z).cpu().numpy()

    def _extract_mean_std(self, prediction: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(prediction, tuple) and len(prediction) >= 2:
            mean, uncertainty = prediction[0], prediction[1]
            # Mobius/sklearn-style predict convention is (mean, std).
            std = uncertainty
        elif hasattr(prediction, "mean"):
            mean = prediction.mean
            if hasattr(prediction, "stddev"):
                std = prediction.stddev
            elif hasattr(prediction, "variance"):
                std = torch.sqrt(prediction.variance.clamp_min(0.0))
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

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        censored_mask: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
        **fit_kwargs: Any,
    ) -> "CensoredEMGPModel":
        original_shape = np.asarray(y).shape
        y1 = self._as_1d(y, name="y").astype(float)
        mask, limits = self._resolve_censoring(y1, censored_mask, censoring_limit)
        if not np.any(mask):
            self.base_model.fit(X, np.asarray(y), **fit_kwargs)
            self.em_result_ = EMResult(y1.copy(), mask, limits, True, 0, 0.0)
            return self

        # Initialisation at thresholds is conservative and deterministic.
        y_imp = y1.copy()
        y_imp[mask] = limits[mask]
        max_change = np.inf
        converged = False

        for iteration in range(1, self.max_iter + 1):
            fit_y = y_imp.reshape(original_shape) if len(original_shape) == 2 else y_imp
            self.base_model.fit(X, fit_y, **fit_kwargs)
            prediction = self.base_model.predict(np.asarray(X)[mask])
            mu, sd = self._extract_mean_std(prediction)
            lower = limits[mask]
            if self.method == "mean":
                proposed = self._truncated_mean(mu, sd, lower)
            else:
                proposed = self._truncated_sample(mu, sd, lower)
            proposed = np.maximum(proposed, lower)
            if self.max_imputation_sd is not None:
                proposed = np.minimum(proposed, lower + self.max_imputation_sd * sd)
            old = y_imp[mask].copy()
            y_imp[mask] = (1.0 - self.damping) * old + self.damping * proposed
            max_change = float(np.max(np.abs(y_imp[mask] - old)))
            if self.verbose:
                print(f"Censored EM iteration {iteration}: max change={max_change:.6g}")
            if max_change < self.tol:
                converged = True
                break

        # Essential: leave the wrapped surrogate fitted to the final imputation.
        fit_y = y_imp.reshape(original_shape) if len(original_shape) == 2 else y_imp
        self.base_model.fit(X, fit_y, **fit_kwargs)
        self.em_result_ = EMResult(y_imp.copy(), mask, limits, converged, iteration, max_change)
        return self

    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.predict(X, *args, **kwargs)

    def clone(self) -> "CensoredEMGPModel":
        """Return an unfitted deep copy, useful in repeated BO experiments."""
        cloned = copy.deepcopy(self)
        cloned.em_result_ = None
        return cloned
