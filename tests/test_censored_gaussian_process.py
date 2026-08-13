"""Tests for :mod:`mobius.surrogate_models.censored_gaussian_process`.

Three layers:

* the truncated-normal moments, checked against ``scipy.stats.truncnorm``;
* the leave-one-out predictive, checked against brute-force refit-free
  conditioning, including the independence property the algorithm relies on;
* the estimator itself - contract, behaviour, and regression tests for each of
  the bugs the previous implementation had.

Run with::

    python -m pytest tests/test_censored_gaussian_process.py -q
"""
from __future__ import annotations

import warnings

import gpytorch
import numpy as np
import pytest
import torch
from scipy.stats import norm, truncnorm

from mobius.surrogate_models.censored_gaussian_process import (
    CensoredEMGPModel,
    EMResult,
    gp_loo_predictive,
    truncated_normal_moments,
)
from mobius.surrogate_models.gaussian_process import GPModel
from mobius.surrogate_models.surrogate_model import _SurrogateModel


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _gp(**kwargs):
    return GPModel(kernel=gpytorch.kernels.MaternKernel(nu=2.5), show_progression=False, **kwargs)


def _standardise(y):
    """Centre and scale.

    ``GPModel`` optimises in float32 via L-BFGS and reports ABNORMAL
    termination on unstandardised targets often enough to make tests flaky.
    Standardising is also what the docstring tells users to do.
    """
    y = np.asarray(y, dtype=float)
    return (y - y.mean()) / (y.std() or 1.0)


@pytest.fixture
def censored_dataset():
    """Factory for a 1-D problem with a configurable right-censoring rate.

    Returns ``make(fraction_censored=0.25, n=70, seed=0)`` giving
    ``(X, y_observed, y_true, limit, censored_mask)``, all standardised on the
    uncensored rows so the limit is on the same scale as the targets.
    """

    def make(fraction_censored: float = 0.25, n: int = 70, seed: int = 0, noise: float = 0.15):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-3.0, 3.0, size=(n, 1))
        latent = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
        y_true = latent + noise * rng.normal(size=n)

        limit = float(np.quantile(y_true, 1.0 - fraction_censored))
        mask = y_true >= limit
        y_obs = np.where(mask, limit, y_true)

        location = y_obs[~mask].mean()
        scale = y_obs[~mask].std() or 1.0
        rescale = lambda v: (v - location) / scale
        return X, rescale(y_obs), rescale(y_true), float(rescale(limit)), mask

    return make


# --------------------------------------------------------------------------- #
# Truncated-normal moments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_truncated_moments_match_scipy(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=200) * 2.0
    sigma = np.exp(rng.normal(size=200) * 0.5)
    lower = mu + rng.uniform(-3.0, 4.0, size=200) * sigma

    mean, variance = truncated_normal_moments(mu, sigma, lower)
    alpha = (lower - mu) / sigma
    np.testing.assert_allclose(mean, truncnorm.mean(alpha, np.inf, loc=mu, scale=sigma), rtol=1e-10)
    np.testing.assert_allclose(variance, truncnorm.var(alpha, np.inf, loc=mu, scale=sigma), rtol=1e-9)


def test_truncated_moments_survive_the_far_tail():
    """Regression test: the old code computed ``log(exp(-x^2/2)/sqrt(2 pi))``.

    That underflows to ``-inf`` for ``|alpha| > 38``, so the inverse Mills
    ratio became ``exp(-inf - log Phi(-alpha))``. Evaluating in log space keeps
    every intermediate representable.
    """
    alpha = np.array([10.0, 40.0, 100.0, 300.0])
    mean, variance = truncated_normal_moments(np.zeros_like(alpha), np.ones_like(alpha), alpha)

    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(variance))
    assert np.all(variance > 0.0)
    assert np.all(mean > alpha)                      # the mean stays in its support
    assert np.all(np.diff(variance) < 0.0)           # deeper truncation is tighter

    # scipy is the reference where it is still trustworthy. Past alpha ~ 500 it
    # returns negative variances, so it cannot be used further out than this.
    # rtol is loose on the variance because ``1 + alpha*lambda - lambda^2``
    # cancels - both here and inside scipy - costing roughly 2*log10(alpha)
    # digits. At alpha = 300 the two agree to 2e-6, which is the accuracy
    # genuinely available in float64, not a defect in either.
    np.testing.assert_allclose(mean, truncnorm.mean(alpha, np.inf), rtol=1e-8)
    np.testing.assert_allclose(variance, truncnorm.var(alpha, np.inf), rtol=1e-5)


def test_truncated_mean_never_falls_below_the_bound():
    mean, _ = truncated_normal_moments(
        np.array([-50.0, 0.0, 50.0]), np.ones(3), np.zeros(3)
    )
    assert np.all(mean >= 0.0)


def test_truncated_mean_exceeds_the_bound_and_the_untruncated_mean():
    mu, sigma, lower = np.array([0.0]), np.array([1.0]), np.array([0.0])
    mean, variance = truncated_normal_moments(mu, sigma, lower)
    assert mean[0] > lower[0] and mean[0] > mu[0]
    # Truncation removes mass, so it can only reduce the variance.
    assert variance[0] < sigma[0] ** 2


def test_zero_sigma_is_floored_not_divided_by():
    mean, variance = truncated_normal_moments([1.0], [0.0], [1.0])
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(variance))


# --------------------------------------------------------------------------- #
# Leave-one-out predictive
# --------------------------------------------------------------------------- #
def test_loo_matches_brute_force_conditioning():
    """R&W eq. 5.12 against explicit n-fold conditioning on fixed hyperparameters."""
    rng = np.random.default_rng(0)
    n = 30
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = _standardise(np.sin(1.5 * X[:, 0]) + 0.2 * rng.normal(size=n))
    model = _gp()
    model.fit(X, y)

    mu_loo, sd_loo = gp_loo_predictive(model)
    assert mu_loo.shape == (n,) and sd_loo.shape == (n,)

    gp, likelihood = model._model, model._likelihood
    gp.train()
    likelihood.train()
    with torch.no_grad():
        prior = likelihood(gp(*gp.train_inputs))
        K = prior.covariance_matrix.double()
        prior_mean = prior.mean.double()
    y_t = torch.as_tensor(y, dtype=torch.float64)

    for i in range(n):
        rest = [j for j in range(n) if j != i]
        K_aa = K[np.ix_(rest, rest)]
        K_ia = K[i, rest]
        solve = torch.linalg.solve(K_aa, y_t[rest] - prior_mean[rest])
        mu_ref = float(prior_mean[i] + K_ia @ solve)
        var_ref = float(K[i, i] - K_ia @ torch.linalg.solve(K_aa, K_ia))
        assert mu_loo[i] == pytest.approx(mu_ref, abs=1e-4)
        assert sd_loo[i] == pytest.approx(np.sqrt(var_ref), abs=1e-4)


def test_loo_mean_does_not_depend_on_the_left_out_target():
    """The identity that stops the E-step chasing its own imputation.

    Expanding ``y_i - [K^-1(y - m)]_i / [K^-1]_ii`` cancels the i-th term, so
    perturbing ``y_i`` alone must leave ``mu_-i`` unchanged.
    """
    rng = np.random.default_rng(1)
    n = 25
    X = rng.uniform(-2.0, 2.0, size=(n, 1))
    y = _standardise(np.cos(X[:, 0]) + 0.1 * rng.normal(size=n))
    model = _gp()
    model.fit(X, y)
    mu_before, _ = gp_loo_predictive(model)

    perturbed = y.copy()
    perturbed[0] += 5.0
    model._model.train_targets = torch.as_tensor(
        perturbed, dtype=model._model.train_targets.dtype
    )
    mu_after, _ = gp_loo_predictive(model)
    assert mu_after[0] == pytest.approx(mu_before[0], abs=1e-3)


def test_loo_uncertainty_exceeds_in_sample_uncertainty():
    """A GP nearly interpolates - which is exactly why in-sample is the wrong signal."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-3.0, 3.0, size=(40, 1))
    y = _standardise(np.sin(1.5 * X[:, 0]) + 0.2 * rng.normal(size=40))
    model = _gp()
    model.fit(X, y)
    _, sd_in_sample = model.predict(X)
    _, sd_loo = gp_loo_predictive(model)
    assert sd_loo.mean() > sd_in_sample.mean()


def test_loo_rejects_a_non_gp_model():
    class NotAGP:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X)), np.ones(len(X))

    with pytest.raises(TypeError, match="exact-GP"):
        gp_loo_predictive(NotAGP())


def test_loo_rejects_an_unfitted_model():
    with pytest.raises(TypeError):
        gp_loo_predictive(_gp())


# --------------------------------------------------------------------------- #
# Estimator: behaviour
# --------------------------------------------------------------------------- #
def test_em_recovers_censored_values_better_than_naive_substitution(censored_dataset):
    """The headline: imputations must land closer to the truth than the sentinel."""
    X, y_obs, y_true, limit, mask = censored_dataset(fraction_censored=0.25, n=80)

    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)

    imputed = model.em_result_.imputed_y[mask]
    error_em = np.abs(imputed - y_true[mask]).mean()
    error_naive = np.abs(limit - y_true[mask]).mean()
    assert error_em < error_naive
    # Imputations move up from the bound, towards the truth.
    assert imputed.mean() > limit


def test_em_improves_predictions_over_naive_substitution(censored_dataset):
    """Held-out accuracy above the detection limit, which is what BO acts on."""
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.30, n=80)
    X_test = np.linspace(-3.0, 3.0, 150).reshape(-1, 1)

    rng = np.random.default_rng(0)
    X_ref = rng.uniform(-3.0, 3.0, size=(80, 1))
    latent_ref = 1.5 * np.sin(X_ref[:, 0]) + 0.4 * X_ref[:, 0]
    y_ref = latent_ref + 0.15 * rng.normal(size=80)
    raw_limit = float(np.quantile(y_ref, 0.70))
    location = np.where(y_ref >= raw_limit, raw_limit, y_ref)[y_ref < raw_limit].mean()
    scale = np.where(y_ref >= raw_limit, raw_limit, y_ref)[y_ref < raw_limit].std()
    latent_test = (1.5 * np.sin(X_test[:, 0]) + 0.4 * X_test[:, 0] - location) / scale
    region = latent_test >= limit

    naive = _gp()
    naive.fit(X, y_obs)
    mu_naive, _ = naive.predict(X_test)

    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    mu_em, _ = model.predict(X_test)

    rmse = lambda mu: np.sqrt(np.mean((mu - latent_test)[region] ** 2))
    assert rmse(mu_em) < rmse(mu_naive)


def test_imputations_never_fall_below_the_limit(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.30, n=60)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    result = model.em_result_
    assert np.all(result.imputed_y[mask] >= limit - 1e-9)
    # Uncensored rows must be left exactly alone.
    np.testing.assert_allclose(result.imputed_y[~mask], y_obs[~mask])


def test_max_imputation_sd_caps_runaway_extrapolation(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.30, n=60)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_imputation_sd=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    loose = CensoredEMGPModel(_gp(), censoring_limit=limit, max_imputation_sd=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose.fit(X, y_obs)
    capped_mean = model.em_result_.imputed_y[mask].mean()
    assert capped_mean <= loose.em_result_.imputed_y[mask].mean() + 1e-9


def test_variance_correction_keeps_the_noise_from_collapsing(censored_dataset):
    """Substituting a conditional mean discards dispersion; the correction restores it.

    The two settings end up with different likelihood classes
    (``FixedNoiseGaussianLikelihood`` against ``GaussianLikelihood``), so their
    ``.noise`` attributes are not comparable. What *is* comparable, and is what
    the correction is actually for, is the predictive uncertainty at the
    censored points and the accuracy of the imputations.
    """
    X, y_obs, y_true, limit, mask = censored_dataset(fraction_censored=0.30, n=70)

    predictive_sd, error = {}, {}
    for flag in (False, True):
        model = CensoredEMGPModel(_gp(), censoring_limit=limit, variance_correction=flag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y_obs)
        _, sigma = model.predict(X[mask])
        predictive_sd[flag] = float(sigma.mean())
        error[flag] = float(np.abs(model.em_result_.imputed_y[mask] - y_true[mask]).mean())

        if flag:
            assert np.all(model.em_result_.imputed_variance[mask] > 0.0)
        else:
            np.testing.assert_allclose(model.em_result_.imputed_variance, 0.0)

    assert predictive_sd[True] > predictive_sd[False]
    assert error[True] < error[False]


def test_damping_of_one_still_converges(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.20, n=60)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, damping=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    assert np.all(np.isfinite(model.em_result_.imputed_y))
    assert np.all(model.em_result_.imputed_y[mask] >= limit - 1e-9)


def test_no_censored_rows_is_a_plain_fit(censored_dataset):
    X, _, y_true, _, _ = censored_dataset()
    model = CensoredEMGPModel(_gp(), censoring_limit=1e9)
    model.fit(X, y_true)
    result = model.em_result_
    assert result.n_iter == 0 and result.converged and result.n_censored == 0
    np.testing.assert_allclose(result.imputed_y, y_true)


def test_explicit_mask_overrides_the_inferred_one(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.30, n=60)
    custom = np.zeros(len(y_obs), dtype=bool)
    custom[np.flatnonzero(mask)[:3]] = True
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, censored_mask=custom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    np.testing.assert_array_equal(model.em_result_.censored_mask, custom)
    assert model.em_result_.n_censored == 3


def test_per_row_censoring_limits(censored_dataset):
    """Assay limits can move between plates; the limit may be a vector."""
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.30, n=60)
    limits = np.full(len(y_obs), limit)
    limits[:30] = limit + 0.5
    model = CensoredEMGPModel(_gp(), censoring_limit=limits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    result = model.em_result_
    assert np.all(result.imputed_y[result.censored_mask] >= limits[result.censored_mask] - 1e-9)


def test_history_records_each_iteration(censored_dataset):
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.25, n=60)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    history = model.em_result_.history
    assert 1 <= len(history) <= 5
    assert set(history[0]) == {"iteration", "max_change", "mean_imputed", "noise"}


def test_heavy_censoring_warns_rather_than_failing_silently(censored_dataset):
    """Past roughly a third censored, the fit is provisional and says so."""
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.60, n=70)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=5, tol=1e-12)
    with pytest.warns(RuntimeWarning, match="provisional"):
        model.fit(X, y_obs)
    assert np.all(np.isfinite(model.em_result_.imputed_y))


# --------------------------------------------------------------------------- #
# Estimator: interface contract (regression tests for the previous version)
# --------------------------------------------------------------------------- #
def test_fit_accepts_the_positional_signature_mobius_uses(censored_dataset):
    """``acquisition_functions._AcquisitionFunction.fit`` calls ``fit(X, y, y_noise)``.

    The previous implementation made everything after ``y`` keyword-only, so
    every acquisition function raised TypeError before doing any work.
    """
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.20, n=50)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs, None)          # exactly the acquisition-function call
    mu, sigma = model.predict(X[:5])
    assert mu.shape == (5,) and sigma.shape == (5,) and np.all(sigma > 0)


def test_known_observation_noise_is_combined_with_the_correction(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.25, n=50)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs, np.full(len(y_obs), 0.01))
    assert np.all(model.em_result_.imputed_variance[mask] > 0.0)


def test_is_a_surrogate_model_and_exposes_imputed_targets(censored_dataset):
    """``best_f`` in an acquisition function reads ``y_train``.

    It must see the imputed values, not the pile-up at the detection limit -
    otherwise every censored design is tied and the ranking collapses.
    """
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.25, n=50)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)

    assert isinstance(model, _SurrogateModel)
    assert model.y_train.max() > limit
    assert len(np.unique(model.y_train[mask])) > 1
    assert len(model.X_train) == len(y_obs)


def test_score_is_inherited_and_works(censored_dataset):
    X, y_obs, y_true, limit, mask = censored_dataset(fraction_censored=0.25, n=50)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    assert np.isfinite(model.score(X[~mask], y_true[~mask]))


def test_unknown_attributes_forward_to_the_wrapped_model():
    base = _gp()
    model = CensoredEMGPModel(base, censoring_limit=0.0)
    assert model.device == base.device
    with pytest.raises(AttributeError):
        _ = model.definitely_not_a_real_attribute


def test_clone_returns_an_unfitted_deep_copy(censored_dataset):
    """Regression test: ``__getattr__`` used to forward dunders, breaking deepcopy."""
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.20, n=40)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)

    fresh = model.clone()
    assert fresh.em_result_ is None
    assert fresh is not model and fresh.base_model is not model.base_model
    assert model.em_result_ is not None      # the original is untouched


def test_bad_arguments_are_rejected_at_construction():
    """Regression test: ``max_iter=0`` used to raise UnboundLocalError inside fit."""
    base = _gp()
    for kwargs in (dict(damping=0.0), dict(damping=1.5), dict(max_iter=0)):
        with pytest.raises(ValueError):
            CensoredEMGPModel(base, censoring_limit=0.0, **kwargs)


def test_missing_censoring_limit_is_rejected(censored_dataset):
    X, y_obs, _, _, _ = censored_dataset(n=30)
    with pytest.raises(ValueError, match="censoring_limit"):
        CensoredEMGPModel(_gp()).fit(X, y_obs)


def test_mismatched_lengths_are_rejected(censored_dataset):
    X, y_obs, _, limit, _ = censored_dataset(n=30)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with pytest.raises(ValueError):
        model.fit(X, y_obs[:10])
    with pytest.raises(ValueError):
        model.fit(X, y_obs, np.zeros(5))


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    model = CensoredEMGPModel(_gp(), censoring_limit=0.0)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((3, 1)))


def test_a_failing_fit_is_retried_before_giving_up(censored_dataset):
    """botorch raises ModelFittingError on a sentinel spike more often than one would like."""
    X, y_obs, _, limit, _ = censored_dataset(fraction_censored=0.25, n=50)

    class Flaky(GPModel):
        calls = 0

        def fit(self, X_train, y_train, y_noise=None):
            Flaky.calls += 1
            if Flaky.calls in (2, 3):
                raise RuntimeError("simulated ModelFittingError")
            return super().fit(X_train, y_train, y_noise)

    base = Flaky(kernel=gpytorch.kernels.MaternKernel(nu=2.5), show_progression=False)
    model = CensoredEMGPModel(base, censoring_limit=limit, max_iter=4, refit_attempts=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    mu, sigma = model.predict(X[:3])
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0)


def test_em_result_reports_censoring_summary(censored_dataset):
    X, y_obs, _, limit, mask = censored_dataset(fraction_censored=0.30, n=60)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    result = model.em_result_
    assert isinstance(result, EMResult)
    assert result.n_censored == int(mask.sum())
    assert result.fraction_censored == pytest.approx(mask.mean())
