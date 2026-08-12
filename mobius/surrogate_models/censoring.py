#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - censoring utilities
#
"""
Shared machinery for censored regression in Mobius.

This module holds everything that is *pure maths* and therefore easy to unit
test in isolation from GPyTorch:

* :class:`CensoringSpec` - a validated description of which observations are
  censored and in which direction.
* :func:`truncated_normal_moments` - numerically stable first and second
  moments of a normal distribution truncated to an interval.
* :func:`censored_normal_logpdf` - the Tobit log-likelihood contribution of a
  single observation under a normal error model.

Conventions
-----------
An observation is described by an interval ``[lower, upper]`` that is known to
contain the true value:

===================  ===========  ===========  =============================
kind                 ``lower``    ``upper``    meaning
===================  ===========  ===========  =============================
observed             ``y``        ``y``        the value is known exactly
right-censored       ``L``        ``+inf``     ``y >= L`` ("> 30 uM")
left-censored        ``-inf``     ``U``        ``y <= U`` ("< 1 nM")
interval-censored    ``L``        ``U``        ``L <= y <= U``
===================  ===========  ===========  =============================

All functions work in ``float64`` regardless of the dtype used by the
surrogate model, because the tail arithmetic is where precision is lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "CensoringSpec",
    "log_ndtr",
    "normal_logpdf",
    "inverse_mills_ratio",
    "truncated_normal_moments",
    "censored_normal_logpdf",
]

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))

# Because everything below is evaluated in log space, the direct forms stay
# accurate to ~1e-9 relative out to alpha ~ 300 in float64 (verified against
# scipy.stats.truncnorm). Only beyond that does the 1 + alpha*lambda - lambda^2
# cancellation eat the mantissa, so the asymptotic series takes over there.
_ASYMPTOTIC_THRESHOLD = 200.0


# --------------------------------------------------------------------------- #
# Low level normal-distribution helpers (torch, float64)
# --------------------------------------------------------------------------- #
def _t(x: Any) -> torch.Tensor:
    """Coerce to a 1-D float64 torch tensor."""
    if torch.is_tensor(x):
        return x.detach().to(dtype=torch.float64).reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1))


def log_ndtr(x: torch.Tensor) -> torch.Tensor:
    """``log Phi(x)``, accurate deep into both tails."""
    return torch.special.log_ndtr(x)


def normal_logpdf(x: torch.Tensor) -> torch.Tensor:
    """``log phi(x)`` computed directly rather than as ``log(exp(...))``.

    Computing ``torch.log(torch.exp(-0.5 * x ** 2) / sqrt(2 pi))`` silently
    underflows to ``-inf`` for ``|x| > ~38`` in float64; this form does not.
    """
    return -0.5 * x * x - _LOG_SQRT_2PI


def inverse_mills_ratio(alpha: torch.Tensor) -> torch.Tensor:
    """``phi(alpha) / (1 - Phi(alpha))``, the right-tail inverse Mills ratio.

    Evaluated as ``exp(log phi(alpha) - log Phi(-alpha))`` so that both factors
    stay representable however far into the tail ``alpha`` sits. Beyond
    :data:`_ASYMPTOTIC_THRESHOLD` the asymptotic series
    ``alpha + 1/alpha - 2/alpha**3 + 10/alpha**5`` is substituted.
    """
    direct = torch.exp(normal_logpdf(alpha) - log_ndtr(-alpha))
    safe = alpha.clamp_min(1.0)
    asymptotic = safe + 1.0 / safe - 2.0 / safe.pow(3) + 10.0 / safe.pow(5)
    return torch.where(alpha > _ASYMPTOTIC_THRESHOLD, asymptotic, direct)


def _one_sided_truncated_variance(
    alpha: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    """Variance factor of a standard normal truncated to ``x > alpha``.

    The textbook form ``1 + alpha * lambda - lambda**2`` is used directly; it
    loses roughly ``2 * log10(alpha)`` digits to cancellation, which float64
    absorbs comfortably until ``alpha`` is in the hundreds. Past that the
    series ``alpha**-2 * (1 - 6 * alpha**-2 + 50 * alpha**-4)`` takes over.
    """
    direct = 1.0 + alpha * lam - lam.pow(2)
    safe = alpha.clamp_min(1.0)
    inv2 = 1.0 / safe.pow(2)
    tail = inv2 * (1.0 - 6.0 * inv2 + 50.0 * inv2.pow(2))
    return torch.where(alpha > _ASYMPTOTIC_THRESHOLD, tail, direct)


def _log_ndtr_diff(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``log(Phi(beta) - Phi(alpha))`` for ``beta >= alpha``, tail-stable.

    The naive difference cancels catastrophically when both bounds sit in the
    same tail, so the computation is rearranged into ``log`` space and the
    smaller term is factored out.
    """
    # Work with whichever tail keeps both terms representable.
    both_left = beta <= 0.0
    both_right = alpha >= 0.0

    log_hi_l, log_lo_l = log_ndtr(beta), log_ndtr(alpha)
    # Phi(beta) - Phi(alpha) == Phi(-alpha) - Phi(-beta) (mirror the right tail)
    log_hi_r, log_lo_r = log_ndtr(-alpha), log_ndtr(-beta)

    def _stable(log_hi: torch.Tensor, log_lo: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp((log_lo - log_hi).clamp(max=-1e-12))
        return log_hi + torch.log1p(-ratio)

    left = _stable(log_hi_l, log_lo_l)
    right = _stable(log_hi_r, log_lo_r)
    # Straddling zero: the mass is O(1) and the plain difference is fine.
    middle = torch.log(
        (torch.special.ndtr(beta) - torch.special.ndtr(alpha)).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    out = torch.where(both_left, left, torch.where(both_right, right, middle))
    return out


# --------------------------------------------------------------------------- #
# Truncated moments
# --------------------------------------------------------------------------- #
def truncated_normal_moments(
    mu: Any,
    sigma: Any,
    lower: Any = None,
    upper: Any = None,
    min_sigma: float = 1e-8,
    as_numpy: bool = True,
) -> Tuple[Any, Any]:
    """First two central moments of ``N(mu, sigma^2)`` truncated to ``[lower, upper]``.

    Parameters
    ----------
    mu, sigma : array-like
        Location and scale of the untruncated normal.
    lower, upper : array-like or None
        Truncation bounds. ``None``, ``-inf`` and ``+inf`` all mean "unbounded
        on that side".
    min_sigma : float, default : 1e-8
        Floor applied to ``sigma`` before any division.
    as_numpy : bool, default : True
        Return numpy arrays rather than torch tensors.

    Returns
    -------
    mean : array of shape (n,)
        ``E[Y | lower <= Y <= upper]``.
    variance : array of shape (n,)
        ``Var[Y | lower <= Y <= upper]``. Always strictly positive.

    Notes
    -----
    For a purely right-truncated normal the variance factor
    ``1 + alpha * lambda - lambda**2`` suffers catastrophic cancellation once
    ``alpha`` exceeds roughly 8 (it tends to ``alpha**-2``, a difference of two
    quantities each of order ``alpha**2``). The asymptotic form is substituted
    in that regime.
    """
    m = _t(mu)
    s = _t(sigma).clamp_min(min_sigma)
    n = m.numel()

    neg_inf = torch.full((n,), -float("inf"), dtype=torch.float64)
    pos_inf = torch.full((n,), float("inf"), dtype=torch.float64)
    lo = neg_inf if lower is None else _t(lower).expand(n).clone()
    hi = pos_inf if upper is None else _t(upper).expand(n).clone()

    if torch.any(hi < lo):
        raise ValueError("`upper` must be >= `lower` for every observation.")

    alpha = (lo - m) / s
    beta = (hi - m) / s
    lo_finite = torch.isfinite(alpha)
    hi_finite = torch.isfinite(beta)

    # ---- generic (interval) branch -------------------------------------- #
    log_z = _log_ndtr_diff(
        torch.where(lo_finite, alpha, torch.full_like(alpha, -40.0)),
        torch.where(hi_finite, beta, torch.full_like(beta, 40.0)),
    )
    zero = torch.zeros_like(m)
    phi_a = torch.where(lo_finite, torch.exp(normal_logpdf(alpha) - log_z), zero)
    phi_b = torch.where(hi_finite, torch.exp(normal_logpdf(beta) - log_z), zero)
    ratio = phi_a - phi_b

    a_phi_a = torch.where(lo_finite, alpha * phi_a, zero)
    b_phi_b = torch.where(hi_finite, beta * phi_b, zero)
    var_factor = 1.0 + a_phi_a - b_phi_b - ratio.pow(2)

    # ---- pure right-truncation branch (upper == +inf) -------------------- #
    lam = inverse_mills_ratio(alpha)
    right_var = _one_sided_truncated_variance(alpha, lam)
    right_only = lo_finite & ~hi_finite
    ratio = torch.where(right_only, lam, ratio)
    var_factor = torch.where(right_only, right_var, var_factor)

    # ---- pure left-truncation branch (lower == -inf) --------------------- #
    # By symmetry: Y | Y <= u  ==  -( (-Y) | (-Y) >= -u ).
    lam_l = inverse_mills_ratio(-beta)
    left_var = _one_sided_truncated_variance(-beta, lam_l)
    left_only = hi_finite & ~lo_finite
    ratio = torch.where(left_only, -lam_l, ratio)
    var_factor = torch.where(left_only, left_var, var_factor)

    # ---- fully unbounded ------------------------------------------------- #
    unbounded = ~lo_finite & ~hi_finite
    ratio = torch.where(unbounded, zero, ratio)
    var_factor = torch.where(unbounded, torch.ones_like(var_factor), var_factor)

    mean = m + s * ratio
    variance = (s.pow(2) * var_factor).clamp_min(torch.finfo(torch.float64).tiny)

    # A truncated mean can never leave its own support; clamp away round-off.
    mean = torch.where(lo_finite, torch.maximum(mean, lo), mean)
    mean = torch.where(hi_finite, torch.minimum(mean, hi), mean)

    if as_numpy:
        return mean.numpy(), variance.numpy()
    return mean, variance


def censored_normal_logpdf(
    y: Any,
    mu: Any,
    sigma: Any,
    lower: Any = None,
    upper: Any = None,
    observed: Any = None,
    min_sigma: float = 1e-8,
) -> np.ndarray:
    """Per-observation Tobit log-likelihood under ``Y ~ N(mu, sigma^2)``.

    Observed rows contribute ``log phi((y - mu) / sigma) - log sigma``;
    censored rows contribute ``log P(lower <= Y <= upper)``.

    This is the quantity a censored model actually maximises, and it is the
    only fair way to compare a Tobit fit against an imputation-based fit.
    """
    m = _t(mu)
    s = _t(sigma).clamp_min(min_sigma)
    yy = _t(y)
    n = m.numel()

    obs = (
        torch.ones(n, dtype=torch.bool)
        if observed is None
        else torch.as_tensor(np.asarray(observed).reshape(-1)).bool().expand(n)
    )

    lo = _t(lower).expand(n) if lower is not None else torch.full((n,), -float("inf"), dtype=torch.float64)
    hi = _t(upper).expand(n) if upper is not None else torch.full((n,), float("inf"), dtype=torch.float64)

    alpha = torch.where(torch.isfinite(lo), (lo - m) / s, torch.full_like(m, -40.0))
    beta = torch.where(torch.isfinite(hi), (hi - m) / s, torch.full_like(m, 40.0))

    ll_obs = normal_logpdf((yy - m) / s) - torch.log(s)
    ll_cens = _log_ndtr_diff(alpha, beta)
    return torch.where(obs, ll_obs, ll_cens).numpy()


# --------------------------------------------------------------------------- #
# Censoring specification
# --------------------------------------------------------------------------- #
@dataclass
class CensoringSpec:
    """Validated description of the censoring pattern of a training set.

    Attributes
    ----------
    censored : ndarray of bool of shape (n,)
        ``True`` where the corresponding target is a censoring sentinel rather
        than a measurement.
    lower : ndarray of float of shape (n,)
        Lower bound of the interval known to contain the true value.
        ``-inf`` for left-censored rows.
    upper : ndarray of float of shape (n,)
        Upper bound. ``+inf`` for right-censored rows.
    """

    censored: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.censored = np.asarray(self.censored, dtype=bool).reshape(-1)
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)
        if not (len(self.censored) == len(self.lower) == len(self.upper)):
            raise ValueError("censored / lower / upper must all have the same length.")
        if np.any(self.upper < self.lower):
            raise ValueError("Every censoring interval must satisfy upper >= lower.")
        bad = self.censored & ~np.isfinite(self.lower) & ~np.isfinite(self.upper)
        if np.any(bad):
            raise ValueError(
                "A censored observation needs at least one finite bound; rows "
                f"{np.flatnonzero(bad).tolist()} have neither."
            )

    # -- convenience views ------------------------------------------------- #
    @property
    def n_censored(self) -> int:
        return int(self.censored.sum())

    @property
    def fraction_censored(self) -> float:
        return float(self.censored.mean()) if len(self.censored) else 0.0

    @property
    def any_censored(self) -> bool:
        return bool(self.censored.any())

    @property
    def is_right_censored(self) -> np.ndarray:
        return self.censored & np.isfinite(self.lower) & ~np.isfinite(self.upper)

    @property
    def is_left_censored(self) -> np.ndarray:
        return self.censored & ~np.isfinite(self.lower) & np.isfinite(self.upper)

    def subset(self, index: np.ndarray) -> "CensoringSpec":
        return CensoringSpec(self.censored[index], self.lower[index], self.upper[index])

    def __len__(self) -> int:
        return len(self.censored)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        y: Any,
        censored: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
        direction: str = "right",
        lower: Optional[Any] = None,
        upper: Optional[Any] = None,
        detection_atol: float = 1e-8,
    ) -> "CensoringSpec":
        """Build a spec from the several shorthands Mobius users reach for.

        Parameters
        ----------
        y : array-like of shape (n,)
            Target values. Censored entries are expected to hold the sentinel
            (usually the assay limit itself).
        censored : array-like of bool, default : None
            Explicit censoring mask. If omitted it is inferred from
            ``censoring_limit`` by comparing ``y`` against the limit.
        censoring_limit : float or array-like, default : None
            The assay detection limit, scalar or per-row.
        direction : {'right', 'left'}, default : 'right'
            Which side ``censoring_limit`` truncates. ``'right'`` means the
            true value is at or above the limit (the ``pIC50 > 4.5`` case);
            ``'left'`` means at or below it.
        lower, upper : array-like, default : None
            Explicit per-row interval bounds. When given they take precedence
            and support interval censoring.
        detection_atol : float, default : 1e-8
            Tolerance used when inferring the mask from ``censoring_limit``.

        Returns
        -------
        spec : `CensoringSpec`
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)

        if lower is not None or upper is not None:
            lo = _broadcast(lower, n, -np.inf, "lower")
            up = _broadcast(upper, n, np.inf, "upper")
            if censored is None:
                mask = ~np.isclose(lo, up, rtol=0.0, atol=detection_atol)
            else:
                mask = _broadcast(censored, n, False, "censored").astype(bool)
            return cls(mask, lo, up)

        if direction not in {"right", "left"}:
            raise ValueError("direction must be 'right' or 'left'.")

        if censoring_limit is None and censored is None:
            return cls(np.zeros(n, dtype=bool), y.copy(), y.copy())

        if censoring_limit is None:
            raise ValueError(
                "A censoring_limit is required when a censored mask is supplied "
                "without explicit lower/upper bounds."
            )

        limits = _broadcast(censoring_limit, n, np.nan, "censoring_limit")
        if censored is None:
            if direction == "right":
                mask = np.isfinite(limits) & (y >= limits - detection_atol)
            else:
                mask = np.isfinite(limits) & (y <= limits + detection_atol)
        else:
            mask = _broadcast(censored, n, False, "censored").astype(bool)

        if np.any(mask & ~np.isfinite(limits)):
            raise ValueError("Every censored row needs a finite censoring_limit.")

        lo = y.copy()
        up = y.copy()
        if direction == "right":
            lo[mask] = limits[mask]
            up[mask] = np.inf
        else:
            lo[mask] = -np.inf
            up[mask] = limits[mask]
        return cls(mask, lo, up)


def _broadcast(value: Any, n: int, fill: float, name: str) -> np.ndarray:
    """Broadcast ``value`` to length ``n``; ``None`` becomes ``fill``."""
    if value is None:
        return np.full(n, fill)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(n, arr.item())
    arr = arr.reshape(-1)
    if len(arr) != n:
        raise ValueError(f"`{name}` has length {len(arr)}; expected {n}.")
    return arr.astype(float) if arr.dtype != bool else arr
