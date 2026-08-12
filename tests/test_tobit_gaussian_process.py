"""Tests for :class:`mobius.surrogate_models.TobitGPModel`.

Two layers:

* the likelihood is checked against high-precision numerical integration, which
  is what actually pins down correctness, and
* the surrogate is checked behaviourally - it must reduce to an ordinary GP
  when nothing is censored, and must not be fooled by the pile-up at the
  detection limit when things are.
"""
from __future__ import annotations

import warnings

import gpytorch
import numpy as np
import pytest
import torch
from scipy.stats import norm
from sklearn.exceptions import NotFittedError

from mobius.surrogate_models.gaussian_process import GPModel
from mobius.surrogate_models.surrogate_model import _SurrogateModel
from mobius.surrogate_models.tobit_gaussian_process import (
    CensoredGaussianLikelihood,
    TobitGPModel,
)


def _kernel():
    return gpytorch.kernels.MaternKernel(nu=2.5)


def _tobit(**kwargs):
    kwargs.setdefault("show_progression", False)
    kwargs.setdefault("random_state", 0)
    return TobitGPModel(kernel=_kernel(), **kwargs)


def _reference_expected_log_prob(mean, variance, sigma, lower, upper, target, censored):
    """``E_{q(f)}[log p(y|f)]`` by dense Gauss-Legendre quadrature over +/-12 sd.

    Uses ``scipy``'s log-space CDFs so the integrand stays accurate in the tail,
    which is exactly where a naive reference would disagree with the model for
    the wrong reason.
    """
    sd = np.sqrt(variance)
    nodes, weights = np.polynomial.legendre.leggauss(4001)
    a, b = mean - 12.0 * sd, mean + 12.0 * sd
    f = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    quad_w = 0.5 * (b - a) * weights
    density = norm.pdf(f, mean, sd)
    if not censored:
        g = norm.logpdf(target, f, sigma)
    elif np.isinf(upper):
        g = norm.logsf((lower - f) / sigma)
    elif np.isinf(lower):
        g = norm.logcdf((upper - f) / sigma)
    else:
        # A plain CDF difference underflows to log(0) once both bounds sit in
        # the same tail. Factor out the larger term instead.
        alpha, beta = (lower - f) / sigma, (upper - f) / sigma
        log_hi = np.where(beta <= 0.0, norm.logcdf(beta), norm.logsf(alpha))
        log_lo = np.where(beta <= 0.0, norm.logcdf(alpha), norm.logsf(beta))
        g = log_hi + np.log1p(-np.exp(np.minimum(log_lo - log_hi, -1e-15)))
    return float((quad_w * density * g).sum())


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mean, variance, lower, upper",
    [
        (-1.0, 0.8, -1.0, np.inf),    # right-censored
        (3.0, 2.0, 3.0, np.inf),      # right-censored, far from the mean
        (0.5, 0.4, -np.inf, 0.5),     # left-censored
        (-1.0, 0.8, -1.0, 0.0),       # interval-censored
    ],
)
def test_expected_log_prob_matches_numerical_quadrature(mean, variance, lower, upper):
    likelihood = CensoredGaussianLikelihood(num_quadrature_points=64)
    likelihood.noise = torch.tensor(0.25)
    sigma = float(likelihood.noise.sqrt())

    dist = gpytorch.distributions.MultivariateNormal(
        torch.tensor([mean]), torch.diag(torch.tensor([variance]))
    )
    got = likelihood.expected_log_prob(
        torch.tensor([0.0]),
        dist,
        censored=torch.tensor([True]),
        lower=torch.tensor([lower]),
        upper=torch.tensor([upper]),
    )
    expected = _reference_expected_log_prob(
        mean, variance, sigma, lower, upper, target=0.0, censored=True
    )
    assert float(got[0]) == pytest.approx(expected, abs=1e-5)


def test_expected_log_prob_is_exact_for_uncensored_rows():
    """The Gaussian branch is closed form: E[(y-f)^2] = (y-m)^2 + v."""
    likelihood = CensoredGaussianLikelihood()
    likelihood.noise = torch.tensor(0.25)
    mean, variance, y = 0.3, 0.7, 1.1
    dist = gpytorch.distributions.MultivariateNormal(
        torch.tensor([mean]), torch.diag(torch.tensor([variance]))
    )
    got = float(
        likelihood.expected_log_prob(
            torch.tensor([y]), dist, censored=torch.tensor([False])
        )[0]
    )
    sigma2 = 0.25
    expected = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + ((y - mean) ** 2 + variance) / sigma2)
    assert got == pytest.approx(expected, rel=1e-6)


def test_log_marginal_uses_the_closed_form_probit_convolution():
    """int Phi((f-L)/sigma) N(f; m, v) df == Phi((m-L)/sqrt(sigma^2+v))."""
    likelihood = CensoredGaussianLikelihood()
    likelihood.noise = torch.tensor(0.25)
    sigma2 = 0.25
    mean = torch.tensor([0.0, 1.0, 0.5])
    variance = torch.tensor([0.3, 1.0, 0.4])
    dist = gpytorch.distributions.MultivariateNormal(mean, torch.diag(variance))
    got = likelihood.log_marginal(
        torch.tensor([0.2, 1.0, 0.5]),
        dist,
        censored=torch.tensor([False, True, True]),
        lower=torch.tensor([-np.inf, 1.0, -np.inf]),
        upper=torch.tensor([np.inf, np.inf, 0.5]),
    ).detach().numpy()

    total = np.sqrt(variance.numpy() + sigma2)
    expected = [
        norm.logpdf(0.2, 0.0, total[0]),
        norm.logsf((1.0 - 1.0) / total[1]),
        norm.logcdf((0.5 - 0.5) / total[2]),
    ]
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_infinite_bounds_do_not_produce_nan_gradients():
    """``torch.where`` evaluates both branches, so ``inf - f`` poisons backward.

    This is the trap that makes a naive censored likelihood fail silently: the
    forward value looks right and every gradient is NaN.
    """
    likelihood = CensoredGaussianLikelihood()
    likelihood.noise = torch.tensor(0.25)
    mean = torch.tensor([0.0, 1.0, -2.0, 0.5]).requires_grad_(True)
    variance = torch.tensor([0.3, 1.0, 0.05, 0.4]).requires_grad_(True)
    dist = gpytorch.distributions.MultivariateNormal(mean, torch.diag(variance))
    loss = likelihood.expected_log_prob(
        torch.tensor([0.2, 1.0, -2.0, 0.5]),
        dist,
        censored=torch.tensor([False, True, True, True]),
        lower=torch.tensor([-np.inf, 1.0, -2.0, -np.inf]),
        upper=torch.tensor([np.inf, np.inf, np.inf, 0.5]),
    ).sum()
    loss.backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(variance.grad).all()
    assert torch.isfinite(likelihood.noise_covar.raw_noise.grad).all()


def test_censored_log_prob_is_monotone_in_the_latent_value():
    """A higher latent potency must make a right-censoring event more likely."""
    likelihood = CensoredGaussianLikelihood()
    likelihood.noise = torch.tensor(0.25)
    values = []
    for m in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        dist = gpytorch.distributions.MultivariateNormal(
            torch.tensor([m]), torch.diag(torch.tensor([1e-4]))
        )
        values.append(
            float(
                likelihood.expected_log_prob(
                    torch.tensor([0.0]),
                    dist,
                    censored=torch.tensor([True]),
                    lower=torch.tensor([0.0]),
                    upper=torch.tensor([np.inf]),
                )[0]
            )
        )
    assert all(np.diff(values) > 0.0)


# --------------------------------------------------------------------------- #
# Surrogate: contract
# --------------------------------------------------------------------------- #
def test_is_a_surrogate_model_with_the_mobius_signature(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.25, n=50)
    model = _tobit(censoring_limit=limit, n_iter=120)
    # Exactly the call acquisition_functions._AcquisitionFunction.fit makes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs, None)
    assert isinstance(model, _SurrogateModel)
    mu, sigma = model.predict(X[:5])
    assert mu.shape == (5,) and sigma.shape == (5,)
    assert np.all(sigma > 0.0)


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        _tobit(censoring_limit=0.0).predict(np.zeros((3, 1)))


def test_y_noise_is_ignored_with_a_warning(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.2, n=40)
    model = _tobit(censoring_limit=limit, n_iter=60)
    with pytest.warns(UserWarning, match="homoscedastic"):
        model.fit(X, y_obs, y_noise=np.full(len(y_obs), 0.01))


def test_bad_arguments_are_rejected_early():
    with pytest.raises(ValueError):
        TobitGPModel(kernel=_kernel(), direction="sideways")
    with pytest.raises(ValueError):
        TobitGPModel(kernel=_kernel(), noise_prior="not a prior")


def test_mismatched_lengths_are_rejected(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(n=30)
    with pytest.raises(ValueError):
        _tobit(censoring_limit=limit).fit(X, y_obs[:10])


def test_standardisation_uses_uncensored_rows_only():
    """Scaling on all rows would use the sentinel pile-up being modelled away."""
    y = np.array([0.0, 1.0, 2.0, 5.0, 5.0, 5.0, 5.0])
    spec_limit = 5.0
    from mobius.surrogate_models.censoring import CensoringSpec

    spec = CensoringSpec.from_arrays(y, censoring_limit=spec_limit)
    location, scale = TobitGPModel._fit_scaler(y, spec)
    assert location == pytest.approx(np.mean([0.0, 1.0, 2.0]))
    assert scale == pytest.approx(np.std([0.0, 1.0, 2.0]))


def test_constant_targets_do_not_divide_by_zero():
    from mobius.surrogate_models.censoring import CensoringSpec

    y = np.full(6, 3.0)
    _, scale = TobitGPModel._fit_scaler(y, CensoringSpec.from_arrays(y))
    assert scale == 1.0


# --------------------------------------------------------------------------- #
# Surrogate: behaviour
# --------------------------------------------------------------------------- #
def test_reduces_to_an_ordinary_gp_when_nothing_is_censored(censored_dataset):
    """With no censored rows the ELBO is a plain variational GP objective."""
    X, _, y_true, f, _, _ = censored_dataset(fraction_censored=0.25, n=90)
    X_test = np.linspace(-3.0, 3.0, 80).reshape(-1, 1)
    f_test = 1.5 * np.sin(X_test[:, 0]) + 0.4 * X_test[:, 0]

    exact = GPModel(kernel=_kernel(), show_progression=False)
    exact.fit(X, y_true)
    mu_exact, _ = exact.predict(X_test)

    tobit = _tobit(n_iter=800)
    tobit.fit(X, y_true)  # no censoring_limit -> nothing is censored
    mu_tobit, _ = tobit.predict(X_test)

    assert np.corrcoef(mu_exact, mu_tobit)[0, 1] > 0.99
    rmse_exact = np.sqrt(((mu_exact - f_test) ** 2).mean())
    rmse_tobit = np.sqrt(((mu_tobit - f_test) ** 2).mean())
    assert rmse_tobit < 3.0 * rmse_exact


def test_predicts_above_the_detection_limit(censored_dataset):
    """The point of the model: predictions are of latent potency, not the readout.

    Naive substitution is bounded by the limit almost everywhere; a Tobit fit
    must extrapolate past it.
    """
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.4, n=90)
    model = _tobit(censoring_limit=limit, n_iter=1200, patience=100)
    model.fit(X, y_obs)
    mu, _ = model.predict(X[mask])
    assert mu.mean() > limit
    assert np.mean(mu > limit) > 0.5


def test_beats_naive_substitution_in_the_censored_region(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.35, n=90)
    X_test = np.linspace(-3.0, 3.0, 150).reshape(-1, 1)
    f_test = 1.5 * np.sin(X_test[:, 0]) + 0.4 * X_test[:, 0]
    region = f_test >= limit

    naive = GPModel(kernel=_kernel(), show_progression=False)
    naive.fit(X, y_obs)
    mu_naive, _ = naive.predict(X_test)

    tobit = _tobit(censoring_limit=limit, n_iter=1200, patience=100)
    tobit.fit(X, y_obs)
    mu_tobit, _ = tobit.predict(X_test)

    rmse = lambda mu: np.sqrt(((mu - f_test)[region] ** 2).mean())
    assert rmse(mu_tobit) < rmse(mu_naive)


def test_uncertainty_is_wider_over_the_censored_region(censored_dataset):
    """Censored points inform but do not pin down; the posterior must reflect that."""
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.35, n=90)
    model = _tobit(censoring_limit=limit, n_iter=1000, patience=100)
    model.fit(X, y_obs)
    _, sd_censored = model.predict(X[mask])
    _, sd_observed = model.predict(X[~mask])
    assert sd_censored.mean() > sd_observed.mean()


def test_naive_substitution_is_overconfident_and_tobit_is_not(censored_dataset):
    """Calibration on held-out data measured without a detection limit."""
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.35, n=90)
    rng = np.random.default_rng(99)
    X_test = rng.uniform(-3.0, 3.0, size=(300, 1))
    f_test = 1.5 * np.sin(X_test[:, 0]) + 0.4 * X_test[:, 0]
    y_test = f_test + 0.15 * rng.normal(size=300)
    region = f_test >= limit

    def coverage(mu, sd):
        lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
        return float(np.mean((y_test[region] >= lo[region]) & (y_test[region] <= hi[region])))

    naive = GPModel(kernel=_kernel(), show_progression=False)
    naive.fit(X, y_obs)
    tobit = _tobit(censoring_limit=limit, n_iter=1200, patience=100)
    tobit.fit(X, y_obs)

    assert coverage(*naive.predict(X_test)) < 0.5   # badly overconfident
    assert coverage(*tobit.predict(X_test)) > 0.75


def test_left_censoring_predicts_below_the_limit():
    rng = np.random.default_rng(7)
    X = rng.uniform(-3.0, 3.0, size=(80, 1))
    f = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
    y = f + 0.15 * rng.normal(size=80)
    limit = float(np.quantile(y, 0.3))
    mask = y <= limit
    y_obs = np.where(mask, limit, y)

    model = _tobit(censoring_limit=limit, direction="left", n_iter=1000, patience=100)
    model.fit(X, y_obs)
    mu, _ = model.predict(X[mask])
    assert mu.mean() < limit


def test_interval_censoring_is_accepted():
    rng = np.random.default_rng(8)
    X = rng.uniform(-2.0, 2.0, size=(50, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=50)
    lower, upper = y.copy(), y.copy()
    lower[:12] -= 0.5
    upper[:12] += 0.5

    model = _tobit(n_iter=400)
    model.fit(X, y, lower=lower, upper=upper)
    assert model.censoring_spec_.n_censored == 12
    mu, sd = model.predict(X[:5])
    assert np.all(np.isfinite(mu)) and np.all(sd > 0)


def test_predict_censored_probability_is_a_probability(censored_dataset):
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.35, n=80)
    model = _tobit(censoring_limit=limit, n_iter=800, patience=80)
    model.fit(X, y_obs)
    p = model.predict_censored_probability(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # Rows that came back censored should be the ones most likely to do so again.
    assert p[mask].mean() > p[~mask].mean()


def test_censored_log_likelihood_prefers_the_censored_fit(censored_dataset):
    """The fair yardstick: R^2 against imputed values scores invented numbers."""
    from mobius.surrogate_models.censoring import CensoringSpec, censored_normal_logpdf

    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.35, n=90)
    spec = CensoringSpec.from_arrays(y_obs, censoring_limit=limit)

    tobit = _tobit(censoring_limit=limit, n_iter=1200, patience=100)
    tobit.fit(X, y_obs)
    tobit_ll = tobit.censored_log_likelihood()

    naive = GPModel(kernel=_kernel(), show_progression=False)
    naive.fit(X, y_obs)
    mu, sd = naive.predict(X)
    naive_ll = float(
        np.sum(censored_normal_logpdf(y_obs, mu, sd, spec.lower, spec.upper, observed=~spec.censored))
    )
    assert tobit_ll > naive_ll


def test_latent_flag_removes_observation_noise(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.25, n=60)
    model = _tobit(censoring_limit=limit, n_iter=500)
    model.fit(X, y_obs)
    _, sd_full = model.predict(X, latent=False)
    _, sd_latent = model.predict(X, latent=True)
    assert np.all(sd_latent <= sd_full + 1e-9)
    assert sd_latent.mean() < sd_full.mean()


def test_fit_is_reproducible_under_a_fixed_seed(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.3, n=60)
    predictions = []
    for _ in range(2):
        model = TobitGPModel(
            kernel=_kernel(), censoring_limit=limit, show_progression=False,
            n_iter=200, random_state=42,
        )
        model.fit(X, y_obs)
        predictions.append(model.predict(X[:10])[0])
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-4)


def test_inducing_points_are_subsampled_for_large_inputs():
    rng = np.random.default_rng(11)
    X = rng.uniform(-3.0, 3.0, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=300)
    model = _tobit(n_inducing=40, n_iter=100)
    model.fit(X, y)
    strategy = model._model.variational_strategy
    assert strategy.inducing_points.shape[0] == 40


def test_loss_history_decreases(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.3, n=60)
    model = _tobit(censoring_limit=limit, n_iter=300, patience=300)
    model.fit(X, y_obs)
    history = np.asarray(model.loss_history_)
    assert len(history) > 10
    assert history[-1] < history[0]
    assert np.all(np.isfinite(history))
