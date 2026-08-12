"""Unit tests for :mod:`mobius.surrogate_models.censoring`.

These are pure-maths tests: no GP is fitted, so they run in well under a second
and pin down the numerics that everything else is built on.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from mobius.surrogate_models.censoring import (
    CensoringSpec,
    censored_normal_logpdf,
    inverse_mills_ratio,
    normal_logpdf,
    truncated_normal_moments,
)

import torch


# --------------------------------------------------------------------------- #
# Elementary helpers
# --------------------------------------------------------------------------- #
def test_normal_logpdf_survives_the_far_tail():
    """``log(exp(-x^2/2)/sqrt(2 pi))`` underflows; the direct form must not."""
    x = torch.tensor([0.0, 10.0, 50.0, 200.0], dtype=torch.float64)
    got = normal_logpdf(x).numpy()
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, norm.logpdf(x.numpy()), rtol=1e-13)


def test_inverse_mills_ratio_matches_scipy_and_stays_finite():
    alpha = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 100.0])
    got = inverse_mills_ratio(torch.tensor(alpha, dtype=torch.float64)).numpy()
    expected = np.exp(norm.logpdf(alpha) - norm.logsf(alpha))
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    # Far beyond where scipy itself is reliable, the asymptotic branch takes over.
    extreme = inverse_mills_ratio(torch.tensor([1e3, 1e4], dtype=torch.float64)).numpy()
    np.testing.assert_allclose(extreme, [1e3 + 1e-3, 1e4 + 1e-4], rtol=1e-9)


# --------------------------------------------------------------------------- #
# Truncated moments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_right_truncated_moments_match_scipy(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=200) * 2.0
    sigma = np.exp(rng.normal(size=200) * 0.5)
    lower = mu + rng.uniform(-3.0, 4.0, size=200) * sigma
    mean, var = truncated_normal_moments(mu, sigma, lower=lower)
    a = (lower - mu) / sigma
    np.testing.assert_allclose(mean, truncnorm.mean(a, np.inf, loc=mu, scale=sigma), rtol=1e-10)
    np.testing.assert_allclose(var, truncnorm.var(a, np.inf, loc=mu, scale=sigma), rtol=1e-9)


def test_left_truncated_moments_match_scipy():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=200) * 2.0
    sigma = np.exp(rng.normal(size=200) * 0.5)
    upper = mu + rng.uniform(-4.0, 3.0, size=200) * sigma
    mean, var = truncated_normal_moments(mu, sigma, upper=upper)
    b = (upper - mu) / sigma
    np.testing.assert_allclose(mean, truncnorm.mean(-np.inf, b, loc=mu, scale=sigma), rtol=1e-10)
    np.testing.assert_allclose(var, truncnorm.var(-np.inf, b, loc=mu, scale=sigma), rtol=1e-9)


def test_interval_truncated_moments_match_scipy():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=200)
    sigma = np.exp(rng.normal(size=200) * 0.3)
    mean, var = truncated_normal_moments(mu, sigma, lower=mu - sigma, upper=mu + 2.0 * sigma)
    np.testing.assert_allclose(mean, truncnorm.mean(-1, 2, loc=mu, scale=sigma), atol=1e-12)
    np.testing.assert_allclose(var, truncnorm.var(-1, 2, loc=mu, scale=sigma), atol=1e-12)


def test_unbounded_truncation_is_the_identity():
    mu = np.array([-1.0, 0.0, 3.0])
    sigma = np.array([0.5, 1.0, 2.0])
    mean, var = truncated_normal_moments(mu, sigma)
    np.testing.assert_allclose(mean, mu)
    np.testing.assert_allclose(var, sigma**2)


def test_deep_tail_variance_stays_positive_where_scipy_fails():
    """``1 + a*lam - lam^2`` cancels catastrophically; the asymptotic branch rescues it.

    ``scipy.stats.truncnorm.var`` returns a *negative* variance past a ~ 500.
    """
    alpha = np.array([100.0, 300.0, 1000.0, 5000.0])
    _, var = truncated_normal_moments(np.zeros_like(alpha), np.ones_like(alpha), lower=alpha)
    assert np.all(var > 0.0)
    # Leading asymptotic order is alpha**-2.
    np.testing.assert_allclose(var, alpha**-2.0, rtol=1e-3)


def test_truncated_mean_never_leaves_its_support():
    mu = np.array([-50.0, 0.0, 50.0])
    sigma = np.array([1.0, 1.0, 1.0])
    lower = np.array([0.0, 0.0, 0.0])
    mean, _ = truncated_normal_moments(mu, sigma, lower=lower)
    assert np.all(mean >= lower)


def test_upper_below_lower_is_rejected():
    with pytest.raises(ValueError):
        truncated_normal_moments([0.0], [1.0], lower=[1.0], upper=[0.0])


# --------------------------------------------------------------------------- #
# Censored log-likelihood
# --------------------------------------------------------------------------- #
def test_censored_logpdf_covers_all_four_observation_types():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.ones(4)
    sigma = np.ones(4)
    lower = np.array([1.0, 2.0, -np.inf, 0.0])
    upper = np.array([1.0, np.inf, 3.0, 2.0])
    observed = np.array([True, False, False, False])
    got = censored_normal_logpdf(y, mu, sigma, lower, upper, observed=observed)
    expected = [
        norm.logpdf(0.0),                       # exactly observed
        norm.logsf(1.0),                        # right-censored at 2
        norm.logcdf(2.0),                       # left-censored at 3
        np.log(norm.cdf(1.0) - norm.cdf(-1.0)), # interval-censored on [0, 2]
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_censored_logpdf_is_finite_far_into_the_tail():
    got = censored_normal_logpdf(
        np.zeros(3), mu=np.zeros(3), sigma=np.ones(3),
        lower=np.array([30.0, 100.0, 250.0]), upper=np.full(3, np.inf),
        observed=np.zeros(3, dtype=bool),
    )
    assert np.all(np.isfinite(got))
    assert np.all(np.diff(got) < 0.0)  # further out is strictly less likely


# --------------------------------------------------------------------------- #
# CensoringSpec
# --------------------------------------------------------------------------- #
def test_spec_infers_right_censoring_from_a_scalar_limit():
    y = np.array([1.0, 2.0, 4.5, 4.5, 3.0])
    spec = CensoringSpec.from_arrays(y, censoring_limit=4.5)
    np.testing.assert_array_equal(spec.censored, [False, False, True, True, False])
    assert spec.n_censored == 2
    assert spec.fraction_censored == pytest.approx(0.4)
    np.testing.assert_array_equal(spec.is_right_censored, spec.censored)
    assert np.all(np.isinf(spec.upper[spec.censored]))
    np.testing.assert_allclose(spec.lower[spec.censored], 4.5)
    # Uncensored rows are degenerate intervals at the observed value.
    np.testing.assert_allclose(spec.lower[~spec.censored], y[~spec.censored])


def test_spec_supports_left_censoring():
    y = np.array([0.5, 2.0, 0.5, 3.0])
    spec = CensoringSpec.from_arrays(y, censoring_limit=0.5, direction="left")
    np.testing.assert_array_equal(spec.censored, [True, False, True, False])
    assert np.all(np.isneginf(spec.lower[spec.censored]))
    np.testing.assert_array_equal(spec.is_left_censored, spec.censored)


def test_spec_supports_mixed_and_interval_censoring_via_bounds():
    y = np.array([1.0, 9.0, 0.0, 5.0])
    spec = CensoringSpec.from_arrays(
        y,
        lower=[1.0, 9.0, -np.inf, 4.0],
        upper=[1.0, np.inf, 0.0, 6.0],
    )
    np.testing.assert_array_equal(spec.censored, [False, True, True, True])
    assert spec.is_right_censored[1] and spec.is_left_censored[2]
    assert not spec.is_right_censored[3] and not spec.is_left_censored[3]


def test_spec_honours_an_explicit_mask_over_the_inferred_one():
    y = np.array([1.0, 4.5, 4.5])
    spec = CensoringSpec.from_arrays(y, censored=[False, True, False], censoring_limit=4.5)
    np.testing.assert_array_equal(spec.censored, [False, True, False])


def test_spec_rejects_a_censored_row_without_a_finite_bound():
    with pytest.raises(ValueError, match="finite"):
        CensoringSpec([True], [-np.inf], [np.inf])


def test_spec_with_no_limit_is_fully_observed():
    spec = CensoringSpec.from_arrays(np.arange(5.0))
    assert not spec.any_censored
    assert spec.fraction_censored == 0.0


def test_spec_subset_keeps_rows_aligned():
    spec = CensoringSpec.from_arrays(np.array([1.0, 4.5, 2.0, 4.5]), censoring_limit=4.5)
    sub = spec.subset(np.array([1, 3]))
    assert len(sub) == 2 and sub.n_censored == 2
