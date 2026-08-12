#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Tobit (censored) Gaussian Process Regressor
#
"""
A Tobit Gaussian Process: censored likelihood, no imputation.

Where :class:`~mobius.surrogate_models.censored_gaussian_process.CensoredEMGPModel`
invents a plausible number for every ``> 30 uM`` and then fits an ordinary GP to
it, the Tobit GP keeps the censored observations as what they actually are -
*inequalities* - and writes them into the likelihood:

.. math::

    p(y_i \\mid f_i) =
    \\begin{cases}
        \\mathcal{N}(y_i;\\, f_i,\\, \\sigma^2) & \\text{observed} \\\\[4pt]
        \\Phi\\!\\left(\\frac{f_i - L_i}{\\sigma}\\right) & y_i \\ge L_i
          \\;\\text{(right-censored)} \\\\[4pt]
        \\Phi\\!\\left(\\frac{U_i - f_i}{\\sigma}\\right) & y_i \\le U_i
          \\;\\text{(left-censored)} \\\\[4pt]
        \\Phi\\!\\left(\\frac{U_i - f_i}{\\sigma}\\right)
        - \\Phi\\!\\left(\\frac{L_i - f_i}{\\sigma}\\right) & \\text{interval}
    \\end{cases}

That likelihood is not conjugate to a GP prior, so exact inference is
unavailable. Both factors are log-concave in ``f``, however, so the posterior
is unimodal and a Gaussian variational approximation is well behaved. This
implementation uses a variational GP (Hensman et al., 2015) whose inducing
points default to the training inputs themselves, which makes the
approximation as tight as the Gaussian family allows; the intractable
expectation ``E_q[log Phi(.)]`` is handled by Gauss-Hermite quadrature.

Why it is worth the extra machinery
-----------------------------------
* **Hyperparameters see the censoring.** The kernel lengthscale, outputscale
  and noise are chosen to maximise the *observed-data* likelihood, so a run of
  inactives at the assay ceiling no longer looks like a suspiciously flat, noiseless
  region of chemical space.
* **Predictions are of the latent potency**, not of the truncated readout, so a
  design predicted at ``pIC50 = 7`` in a region censored at 4.5 is reported as
  such and the acquisition function can act on it.
* **Uncertainty stays honest.** A censored point contributes information but not
  a pinned value, so the posterior stays appropriately wide over the censored
  region rather than collapsing onto invented data.
* It degrades gracefully: with no censored rows it reduces to a standard
  variational GP with a Gaussian likelihood.

The cost is that fitting is gradient-based rather than closed-form, so it is
slower than :class:`~mobius.surrogate_models.GPModel` and has an optimiser to
tune. For lightly censored data sets Schmee-Hahn is cheaper and close enough;
past roughly a third censored, this model is the one to reach for.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import gpytorch
import numpy as np
import torch

from .censoring import CensoringSpec, censored_normal_logpdf
from .surrogate_model import _SurrogateModel

__all__ = ["TobitGPModel", "CensoredGaussianLikelihood"]

_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_EPS = -700.0
#: Stand-in for an infinite standardised bound. Phi(+/-60) is 0 or 1 to well
#: within float32 resolution, and it keeps every intermediate finite.
_BOUND_SENTINEL = 60.0


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
class CensoredGaussianLikelihood(gpytorch.likelihoods.Likelihood):
    """Gaussian likelihood in which some observations are only bounded.

    Parameters
    ----------
    noise_prior : `gpytorch.priors.Prior`, default : None
        Prior on the observation noise variance.
    noise_constraint : `gpytorch.constraints.Interval`, default : None
        Constraint on the noise variance. Defaults to
        ``GreaterThan(1e-6)``, which stops the noise collapsing onto a
        censored region.
    num_quadrature_points : int, default : 32
        Number of Gauss-Hermite nodes used for ``E_q[log p(y | f)]`` on the
        censored rows. 32 is ample for a log-concave integrand; drop it to 16
        if fitting is the bottleneck.

    Notes
    -----
    ``expected_log_prob`` is the quantity a variational ELBO needs;
    ``log_marginal`` is the quantity a *predictive* likelihood needs and has a
    closed form here, because ``int Phi((f - L)/sigma) N(f; m, v) df =
    Phi((m - L)/sqrt(sigma^2 + v))``.
    """

    def __init__(
        self,
        noise_prior: Optional[Any] = None,
        noise_constraint: Optional[Any] = None,
        num_quadrature_points: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
        self.noise_covar = gpytorch.likelihoods.noise_models.HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint
        )
        self.num_quadrature_points = int(num_quadrature_points)
        nodes, weights = np.polynomial.hermite.hermgauss(self.num_quadrature_points)
        self.register_buffer("_gh_nodes", torch.as_tensor(nodes, dtype=torch.float32))
        self.register_buffer(
            "_gh_weights", torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float32)
        )

    # -- noise accessors ---------------------------------------------------- #
    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Any) -> None:
        self.noise_covar.initialize(noise=value)

    # -- core log-likelihood ------------------------------------------------ #
    def _log_prob(
        self,
        f: torch.Tensor,
        target: torch.Tensor,
        censored: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        """``log p(y | f)`` broadcast over any leading sample dimensions."""
        sigma = self.noise.sqrt().reshape(-1)[0]
        gaussian = -0.5 * _LOG_2PI - torch.log(sigma) - 0.5 * ((target - f) / sigma) ** 2

        # log[ Phi((upper - f)/sigma) - Phi((lower - f)/sigma) ], tail-stable.
        lo, hi, lo_finite, hi_finite = _safe_bounds(lower, upper, f)
        big = torch.full_like(f, _BOUND_SENTINEL)
        beta = torch.where(hi_finite, (hi - f) / sigma, big)
        alpha = torch.where(lo_finite, (lo - f) / sigma, -big)
        censored_ll = _log_ndtr_diff(alpha, beta)

        return torch.where(censored, censored_ll, gaussian)

    def expected_log_prob(
        self,
        target: torch.Tensor,
        input: gpytorch.distributions.MultivariateNormal,
        censored: Optional[torch.Tensor] = None,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """``E_{q(f)}[log p(y | f)]`` for each observation.

        The uncensored term is closed form; the censored term is evaluated by
        Gauss-Hermite quadrature over the marginal ``q(f_i)``.
        """
        mean = input.mean.reshape(-1)
        variance = input.variance.reshape(-1).clamp_min(1e-12)
        n = mean.numel()

        if censored is None:
            censored = torch.zeros(n, dtype=torch.bool, device=mean.device)
        if lower is None:
            lower = torch.full((n,), -float("inf"), device=mean.device)
        if upper is None:
            upper = torch.full((n,), float("inf"), device=mean.device)

        sigma2 = self.noise.reshape(-1)[0]
        # Closed form for the Gaussian rows: E[(y - f)^2] = (y - m)^2 + v.
        finite_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        gaussian = -0.5 * (
            _LOG_2PI + torch.log(sigma2) + ((finite_target - mean) ** 2 + variance) / sigma2
        )

        # Quadrature for the censored rows.
        nodes = self._gh_nodes.to(mean.dtype).to(mean.device).unsqueeze(-1)
        weights = self._gh_weights.to(mean.dtype).to(mean.device).unsqueeze(-1)
        f = mean.unsqueeze(0) + torch.sqrt(2.0 * variance).unsqueeze(0) * nodes
        sigma = sigma2.sqrt()
        lo, hi, _, _ = _safe_bounds(lower, upper, lower)
        lo_finite = torch.isfinite(lower).unsqueeze(0)
        hi_finite = torch.isfinite(upper).unsqueeze(0)
        big = torch.full_like(f, _BOUND_SENTINEL)
        beta = torch.where(hi_finite, (hi.unsqueeze(0) - f) / sigma, big)
        alpha = torch.where(lo_finite, (lo.unsqueeze(0) - f) / sigma, -big)
        censored_ll = (weights * _log_ndtr_diff(alpha, beta)).sum(0)

        return torch.where(censored, censored_ll, gaussian)

    def log_marginal(
        self,
        observations: torch.Tensor,
        function_dist: gpytorch.distributions.MultivariateNormal,
        censored: Optional[torch.Tensor] = None,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """``log int p(y | f) q(f) df`` per observation, in closed form.

        Convolving a probit with a Gaussian gives another probit with an
        inflated scale, so no quadrature is needed here.
        """
        mean = function_dist.mean.reshape(-1)
        variance = function_dist.variance.reshape(-1).clamp_min(0.0)
        n = mean.numel()
        if censored is None:
            censored = torch.zeros(n, dtype=torch.bool, device=mean.device)
        if lower is None:
            lower = torch.full((n,), -float("inf"), device=mean.device)
        if upper is None:
            upper = torch.full((n,), float("inf"), device=mean.device)

        total_var = (variance + self.noise.reshape(-1)[0]).clamp_min(1e-12)
        scale = total_var.sqrt()
        gaussian = -0.5 * (_LOG_2PI + torch.log(total_var) + (observations - mean) ** 2 / total_var)

        lo, hi, lo_finite, hi_finite = _safe_bounds(lower, upper, mean)
        big = torch.full_like(mean, _BOUND_SENTINEL)
        beta = torch.where(hi_finite, (hi - mean) / scale, big)
        alpha = torch.where(lo_finite, (lo - mean) / scale, -big)
        censored_ll = _log_ndtr_diff(alpha, beta)
        return torch.where(censored, censored_ll, gaussian)

    def forward(self, function_samples: torch.Tensor, **kwargs: Any) -> torch.distributions.Normal:
        """Observation distribution ignoring censoring - used for sampling."""
        return torch.distributions.Normal(function_samples, self.noise.sqrt())

    def marginal(
        self, function_dist: gpytorch.distributions.MultivariateNormal, **kwargs: Any
    ) -> gpytorch.distributions.MultivariateNormal:
        """Latent posterior convolved with the observation noise."""
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise = self.noise.reshape(-1)[0].expand(mean.shape[-1])
        return function_dist.__class__(mean, covar.add_diagonal(noise))


def _safe_bounds(
    lower: torch.Tensor, upper: torch.Tensor, like: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace infinite bounds by finite placeholders *before* any arithmetic.

    ``torch.where`` evaluates both branches, so letting ``inf - f`` into the
    discarded branch poisons the backward pass with ``nan`` even though the
    forward value is correct. Sanitising first keeps every element finite.
    """
    lo_finite = torch.isfinite(lower).expand_as(like) if lower.dim() == like.dim() else torch.isfinite(lower)
    hi_finite = torch.isfinite(upper).expand_as(like) if upper.dim() == like.dim() else torch.isfinite(upper)
    lo = torch.where(torch.isfinite(lower), lower, torch.zeros_like(lower))
    hi = torch.where(torch.isfinite(upper), upper, torch.zeros_like(upper))
    return lo, hi, lo_finite, hi_finite


def _log_ndtr_diff(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``log(Phi(beta) - Phi(alpha))`` for ``beta >= alpha``, autograd-safe.

    Written so that no branch ever evaluates ``log(0)`` or ``log1p(-1)``, which
    would put a NaN into the backward pass even for the branch that is
    ultimately discarded by ``torch.where``.
    """
    log_ndtr = torch.special.log_ndtr
    finfo = torch.finfo(alpha.dtype)
    # ``torch.where`` evaluates every branch, so a -inf produced by a branch that
    # is ultimately discarded still puts a nan into the backward pass. Each
    # branch below is therefore clamped to be finite in its own right.
    ceiling = 1.0 - 8.0 * finfo.eps

    both_left = beta <= 0.0
    both_right = alpha >= 0.0

    def _stable(log_hi: torch.Tensor, log_lo: torch.Tensor) -> torch.Tensor:
        # Factor out the larger term: log(A - B) = log A + log1p(-B/A).
        # Clamping after the exponential (rather than before) matters in
        # float32, where exp(-1e-7) rounds to exactly 1.0.
        ratio = torch.exp(log_lo - log_hi).clamp(max=ceiling)
        return log_hi + torch.log1p(-ratio)

    left = _stable(log_ndtr(beta), log_ndtr(alpha))
    right = _stable(log_ndtr(-alpha), log_ndtr(-beta))
    middle = torch.log(
        (torch.special.ndtr(beta) - torch.special.ndtr(alpha)).clamp_min(finfo.tiny)
    )
    out = torch.where(both_left, left, torch.where(both_right, right, middle))
    return out.clamp_min(_LOG_EPS)


# --------------------------------------------------------------------------- #
# Variational GP
# --------------------------------------------------------------------------- #
class _VariationalGPModel(gpytorch.models.ApproximateGP):
    """Sparse variational GP with a constant mean and a scaled kernel."""

    def __init__(self, inducing_points: torch.Tensor, kernel: Any, learn_inducing: bool = False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# --------------------------------------------------------------------------- #
# Surrogate
# --------------------------------------------------------------------------- #
class TobitGPModel(_SurrogateModel):
    """Gaussian Process regressor with a censored (Tobit) likelihood.

    The constructor mirrors :class:`~mobius.surrogate_models.GPModel`, so it can
    be swapped in wherever that model is used.

    Parameters
    ----------
    kernel : `gpytorch.kernels.Kernel`
        Covariance function. It is wrapped in a ``ScaleKernel`` internally, as
        in ``GPModel``.
    transform : callable, default : None
        Object with a ``transform`` method mapping sequences to features.
    noise_prior : `gpytorch.priors.Prior`, default : None
        Prior on the observation noise variance.
    noise_constraint : `gpytorch.constraints.Interval`, default : None
        Constraint on the noise variance. Defaults to ``GreaterThan(1e-6)``.
    direction : {'right', 'left'}, default : 'right'
        Default censoring direction used when only a ``censoring_limit`` is
        given. ``'right'`` means the true value lies at or above the limit.
    censoring_limit : float or array-like, default : None
        Default detection limit; may also be supplied per call to :meth:`fit`.
    standardize : bool, default : True
        Standardise targets (and censoring bounds) using the mean and standard
        deviation of the *uncensored* rows before fitting, then map predictions
        back. Strongly recommended: an unstandardised target with a spike of
        identical values at the detection limit is a common cause of
        marginal-likelihood optimisation failures.
    n_inducing : int, default : None
        Number of inducing points. ``None`` uses every training input when
        ``n_samples <= max_exact_inducing`` and a random subset otherwise.
    max_exact_inducing : int, default : 512
        Threshold above which inducing points are subsampled.
    learn_inducing_locations : bool, default : False
        Whether inducing locations are optimised. Keep this ``False`` for
        discrete inputs such as sequence embeddings, where interpolating
        between inputs is meaningless.
    num_quadrature_points : int, default : 32
        Gauss-Hermite nodes for the censored expectation.
    n_iter : int, default : 500
        Maximum number of Adam steps.
    learning_rate : float, default : 0.05
        Adam learning rate.
    tol : float, default : 1e-5
        Relative change in the loss below which fitting stops early.
    patience : int, default : 30
        Number of consecutive tolerated iterations without improvement.
    jitter : float, default : 1e-4
        Cholesky jitter used during variational inference.
    device : str or torch.device, default : None
        Defaults to CUDA when available.
    show_progression : bool, default : True
        Display a progress bar while fitting.
    random_state : int, default : None
        Seed for inducing-point subsampling and variational initialisation.

    Attributes
    ----------
    loss_history_ : list of float
        Negative ELBO per iteration from the last fit.
    censoring_spec_ : `CensoringSpec`
        The censoring pattern used in the last fit.

    Examples
    --------
    >>> import gpytorch
    >>> from mobius.surrogate_models import TobitGPModel
    >>> model = TobitGPModel(kernel=gpytorch.kernels.MaternKernel(nu=2.5),
    ...                      censoring_limit=4.5)                        # doctest: +SKIP
    >>> model.fit(X_train, y_train)                                      # doctest: +SKIP
    >>> mu, sigma = model.predict(X_test)   # latent potency, not the readout
    """

    def __init__(
        self,
        kernel: Any,
        transform: Optional[Any] = None,
        noise_prior: Optional[Any] = None,
        noise_constraint: Optional[Any] = None,
        direction: str = "right",
        censoring_limit: Optional[Any] = None,
        standardize: bool = True,
        n_inducing: Optional[int] = None,
        max_exact_inducing: int = 512,
        learn_inducing_locations: bool = False,
        num_quadrature_points: int = 32,
        n_iter: int = 500,
        learning_rate: float = 0.05,
        tol: float = 1e-5,
        patience: int = 30,
        jitter: float = 1e-4,
        device: Optional[Any] = None,
        show_progression: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if direction not in {"right", "left"}:
            raise ValueError("`direction` must be 'right' or 'left'.")
        if noise_prior is not None and not isinstance(noise_prior, gpytorch.priors.Prior):
            raise ValueError("`noise_prior` must be an instance of gpytorch.priors.Prior.")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._kernel = kernel
        self._transform = transform
        self._noise_prior = noise_prior
        self._noise_constraint = noise_constraint
        self._direction = direction
        self._censoring_limit = censoring_limit
        self._standardize = bool(standardize)
        self._n_inducing = n_inducing
        self._max_exact_inducing = int(max_exact_inducing)
        self._learn_inducing_locations = bool(learn_inducing_locations)
        self._num_quadrature_points = int(num_quadrature_points)
        self._n_iter = int(n_iter)
        self._learning_rate = float(learning_rate)
        self._tol = float(tol)
        self._patience = int(patience)
        self._jitter = float(jitter)
        self._device = device
        self._show_progression = bool(show_progression)
        self._random_state = random_state

        self._model = None
        self._likelihood = None
        self._X_train = None
        self._y_train = None
        self._y_noise = None
        self._location = 0.0
        self._scale = 1.0
        self.loss_history_ = []
        self.censoring_spec_ = None

    # -- properties --------------------------------------------------------- #
    @property
    def device(self) -> Any:
        """Device the model is running on."""
        return self._device

    @property
    def noise(self) -> float:
        """Fitted observation noise *variance*, on the original target scale."""
        self._check_fitted()
        return float(self._likelihood.noise.detach().cpu().item()) * self._scale**2

    # -- fitting ------------------------------------------------------------ #
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        y_noise: Optional[Any] = None,
        censored_mask: Optional[Any] = None,
        censoring_limit: Optional[Any] = None,
        lower: Optional[Any] = None,
        upper: Optional[Any] = None,
    ) -> "TobitGPModel":
        """Fit the Tobit GP by maximising the variational ELBO.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,) or (n_samples, n_features)
            Sequences (if ``transform`` is set) or feature vectors.
        y_train : array-like of shape (n_samples,)
            Target values, with censoring sentinels in place for censored rows.
        y_noise : array-like of shape (n_samples,), default : None
            Accepted for interface compatibility with the rest of Mobius and
            currently ignored: the Tobit likelihood estimates a single
            homoscedastic noise term. A warning is raised if it is supplied.
        censored_mask : array-like of bool, default : None
            Explicit censoring mask; inferred from ``censoring_limit`` if absent.
        censoring_limit : float or array-like, default : None
            Detection limit, overriding the constructor value.
        lower, upper : array-like, default : None
            Explicit per-row censoring intervals, for interval censoring or
            mixed left/right censoring.

        Returns
        -------
        self : `TobitGPModel`
        """
        if y_noise is not None:
            warnings.warn(
                "TobitGPModel estimates a single homoscedastic noise term and ignores "
                "`y_noise`. Use GPModel or CensoredEMGPModel if per-observation noise "
                "matters more to you than exact censoring.",
                UserWarning,
            )

        self._X_train = np.asarray(X_train).copy()
        y = np.asarray(y_train, dtype=float).reshape(-1)
        self._y_train = y.copy()
        n = len(y)
        if self._X_train.shape[0] != n:
            raise ValueError(
                f"X_train has {self._X_train.shape[0]} samples but y_train has {n} values."
            )

        spec = CensoringSpec.from_arrays(
            y,
            censored=censored_mask,
            censoring_limit=censoring_limit if censoring_limit is not None else self._censoring_limit,
            direction=self._direction,
            lower=lower,
            upper=upper,
        )
        self.censoring_spec_ = spec

        self._location, self._scale = self._fit_scaler(y, spec)
        y_s = (y - self._location) / self._scale
        lower_s = (spec.lower - self._location) / self._scale
        upper_s = (spec.upper - self._location) / self._scale

        features = self._featurise(self._X_train)
        generator = None
        if self._random_state is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(self._random_state))
            torch.manual_seed(int(self._random_state))

        inducing = self._select_inducing_points(features, generator)
        self._model = _VariationalGPModel(
            inducing, self._kernel, learn_inducing=self._learn_inducing_locations
        ).to(self._device)
        self._likelihood = CensoredGaussianLikelihood(
            noise_prior=self._noise_prior,
            noise_constraint=self._noise_constraint,
            num_quadrature_points=self._num_quadrature_points,
        ).to(self._device)

        # A sensible starting point: the variational mean at the prior mean and
        # the noise at a fraction of the (standardised) target variance.
        with torch.no_grad():
            self._likelihood.noise = torch.tensor(0.1, device=self._device)

        targets = torch.as_tensor(y_s, dtype=torch.float32, device=self._device)
        censored = torch.as_tensor(spec.censored, dtype=torch.bool, device=self._device)
        lower_t = torch.as_tensor(lower_s, dtype=torch.float32, device=self._device)
        upper_t = torch.as_tensor(upper_s, dtype=torch.float32, device=self._device)
        features = features.to(self._device)

        self._optimise(features, targets, censored, lower_t, upper_t)
        self._model.eval()
        self._likelihood.eval()
        return self

    def _optimise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        censored: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> None:
        self._model.train()
        self._likelihood.train()
        parameters = list(self._model.parameters()) + list(self._likelihood.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self._learning_rate)

        progress = None
        if self._show_progression:
            try:
                from ..utils import ProgressBar

                progress = ProgressBar(desc=f"Fitting Tobit GP ({self._device})")
            except Exception:  # pragma: no cover - utils pulls in heavy deps
                progress = None

        best_loss = float("inf")
        best_state = None
        stale = 0
        self.loss_history_ = []

        for step in range(self._n_iter):
            optimizer.zero_grad(set_to_none=True)
            with gpytorch.settings.cholesky_jitter(
                float_value=self._jitter, double_value=self._jitter
            ):
                output = self._model(features)
                expected_ll = self._likelihood.expected_log_prob(
                    targets, output, censored=censored, lower=lower, upper=upper
                ).sum()
                kl = self._model.variational_strategy.kl_divergence().sum()
                loss = -(expected_ll - kl)

            if not torch.isfinite(loss):
                warnings.warn(
                    f"Non-finite ELBO at iteration {step}; stopping early and keeping "
                    "the best parameters seen so far.",
                    RuntimeWarning,
                )
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()

            value = float(loss.detach().cpu())
            self.loss_history_.append(value)
            if progress is not None:
                progress(None, type("R", (), {"fval": value})())

            if value < best_loss - self._tol * max(1.0, abs(best_loss)):
                best_loss = value
                best_state = {
                    "model": {k: v.detach().clone() for k, v in self._model.state_dict().items()},
                    "likelihood": {
                        k: v.detach().clone() for k, v in self._likelihood.state_dict().items()
                    },
                }
                stale = 0
            else:
                stale += 1
                if value < best_loss:
                    best_loss = value
                if stale >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state["model"])
            self._likelihood.load_state_dict(best_state["likelihood"])

    # -- prediction --------------------------------------------------------- #
    def predict(
        self, X_test: Any, y_noise: Optional[Any] = None, latent: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the *latent* target value at new inputs.

        The returned mean is the posterior over the underlying quantity - the
        potency the assay would have reported with unlimited dynamic range -
        not the censored readout. That is what an acquisition function should
        be ranking on.

        Parameters
        ----------
        X_test : array-like of shape (n_samples,) or (n_samples, n_features)
            Query points.
        y_noise : array-like, default : None
            Ignored; present for interface compatibility.
        latent : bool, default : False
            When ``True`` the returned standard deviation excludes observation
            noise, giving the posterior over ``f`` alone.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Posterior mean, on the original target scale.
        sigma : ndarray of shape (n_samples,)
            Posterior standard deviation, on the original target scale.
        """
        mean, variance = self._posterior(X_test, include_noise=not latent)
        return mean, np.sqrt(variance)

    def predict_censored_probability(
        self,
        X_test: Any,
        censoring_limit: Optional[Any] = None,
        direction: Optional[str] = None,
    ) -> np.ndarray:
        """Probability that a new measurement would come back censored.

        Useful for triaging a proposed batch: a design whose predicted potency
        is high but whose censoring probability is also high is one the current
        assay cannot resolve, and is a candidate for a different readout.

        Parameters
        ----------
        X_test : array-like of shape (n_samples,) or (n_samples, n_features)
            Query points.
        censoring_limit : float or array-like, default : None
            Detection limit. Defaults to the constructor value.
        direction : {'right', 'left'}, default : None
            Defaults to the constructor value.

        Returns
        -------
        probability : ndarray of shape (n_samples,)
            ``P(Y >= limit)`` for right censoring, ``P(Y <= limit)`` for left.
        """
        limit = censoring_limit if censoring_limit is not None else self._censoring_limit
        if limit is None:
            raise ValueError("A censoring_limit is required.")
        direction = direction or self._direction

        mean, variance = self._posterior(X_test, include_noise=True)
        scale = np.sqrt(np.maximum(variance, 1e-24))
        z = (np.asarray(limit, dtype=float) - mean) / scale
        cdf = torch.special.ndtr(torch.as_tensor(z, dtype=torch.float64)).numpy()
        return 1.0 - cdf if direction == "right" else cdf

    def _posterior(self, X_test: Any, include_noise: bool) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        self._model.eval()
        self._likelihood.eval()
        features = self._featurise(np.asarray(X_test)).to(self._device)
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(
            float_value=self._jitter, double_value=self._jitter
        ):
            latent = self._model(features)
            mean = latent.mean.detach().cpu().numpy().astype(float)
            variance = latent.variance.detach().cpu().numpy().astype(float)
            if include_noise:
                variance = variance + float(self._likelihood.noise.detach().cpu().item())
        mean = mean * self._scale + self._location
        variance = variance * self._scale**2
        return mean, np.maximum(variance, 0.0)

    def censored_log_likelihood(
        self, X: Optional[Any] = None, y: Optional[Any] = None, spec: Optional[CensoringSpec] = None
    ) -> float:
        """Observed-data log-likelihood, the fair yardstick for censored fits.

        Observed rows contribute a Gaussian density, censored rows the log
        probability of their interval. Comparing this between a Tobit fit and
        an imputation-based fit is meaningful; comparing R^2 against imputed
        values is not, because each method scores itself against numbers it
        invented.

        Parameters
        ----------
        X : array-like, default : None
            Inputs. Defaults to the training inputs.
        y : array-like, default : None
            Targets. Defaults to the training targets.
        spec : `CensoringSpec`, default : None
            Censoring pattern. Defaults to the one used at fit time.

        Returns
        -------
        loglik : float
            Summed log-likelihood.
        """
        self._check_fitted()
        X = self._X_train if X is None else np.asarray(X)
        y = self._y_train if y is None else np.asarray(y, dtype=float).reshape(-1)
        spec = self.censoring_spec_ if spec is None else spec
        if spec is None:
            spec = CensoringSpec.from_arrays(y)

        mean, variance = self._posterior(X, include_noise=True)
        return float(
            np.sum(
                censored_normal_logpdf(
                    y, mean, np.sqrt(variance), spec.lower, spec.upper, observed=~spec.censored
                )
            )
        )

    def score(self, X_test: Any, y_test: Any) -> float:
        """Coefficient of determination against *uncensored* test values."""
        return super().score(X_test, y_test)

    # -- helpers ------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self._model is None:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                "This model instance is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this estimator."
            )

    @staticmethod
    def _fit_scaler(y: np.ndarray, spec: CensoringSpec) -> Tuple[float, float]:
        """Location/scale from the uncensored rows only.

        Standardising on all rows would use the sentinel pile-up at the limit,
        which is exactly the artefact being modelled away.
        """
        reference = y[~spec.censored] if (~spec.censored).any() else y
        location = float(np.mean(reference))
        scale = float(np.std(reference))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return location, scale

    def _featurise(self, X: Any) -> torch.Tensor:
        if self._transform is not None:
            with torch.no_grad():
                X = self._transform.transform(X)
        if torch.is_tensor(X):
            return X.float()
        return torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float32)

    def _select_inducing_points(
        self, features: torch.Tensor, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        n = features.shape[0]
        target = self._n_inducing or (n if n <= self._max_exact_inducing else self._max_exact_inducing)
        target = int(min(max(target, 1), n))
        if target == n:
            return features.clone()
        index = torch.randperm(n, generator=generator)[:target]
        return features[index].clone()
