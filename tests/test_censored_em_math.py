"""Numerical correctness of the truncated-normal machinery.

The reference is ``scipy.stats.truncnorm``, which mobius already depends on.
Everything here is pure maths: no surrogate is fitted.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from ._cgp_import import CensoredEMGPModel


@pytest.fixture
def model():
    """A wrapper instance used only as a namespace for the maths helpers."""
    return CensoredEMGPModel(base_model=object(), censoring_limit=0.0)


def _exact_truncated_mean(mu, sd, lower):
    alpha = (np.asarray(lower) - np.asarray(mu)) / np.asarray(sd)
    return truncnorm.mean(a=alpha, b=np.inf, loc=mu, scale=sd)


# ---------------------------------------------------------------------------
# inverse Mills ratio / conditional mean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [-6.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 7.5])
def test_truncated_mean_matches_scipy_moderate_alpha(model, alpha):
    """Below the asymptotic switchover the implementation must be exact."""
    mu, sd = np.array([0.0]), np.array([1.0])
    lower = np.array([float(alpha)])
    got = model._truncated_mean(mu, sd, lower)
    want = _exact_truncated_mean(mu, sd, lower)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.known_bug
@pytest.mark.parametrize("alpha", [8.001, 8.5, 9.0, 12.0, 20.0, 30.0, 37.0])
def test_truncated_mean_matches_scipy_large_alpha(model, alpha):
    """The exact log-ratio path is valid to alpha ~= 38.

    The implementation switches to the crude one-term asymptotic
    ``alpha + 1/alpha`` at ``alpha > 8``, which is ~4e-4 relative error - far
    looser than necessary and larger than the default convergence tolerance.
    """
    mu, sd = np.array([0.0]), np.array([1.0])
    lower = np.array([float(alpha)])
    got = model._truncated_mean(mu, sd, lower)
    want = _exact_truncated_mean(mu, sd, lower)
    assert got == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("alpha", [40.0, 60.0, 100.0])
def test_truncated_mean_no_underflow_extreme_alpha(model, alpha):
    """Must not collapse to the untruncated mean when the pdf underflows.

    ``log(exp(-a**2/2)/sqrt(2*pi))`` underflows to ``-inf`` for alpha >~ 38;
    the result must still be a finite value above ``lower``.
    """
    mu, sd = np.array([0.0]), np.array([1.0])
    lower = np.array([float(alpha)])
    got = model._truncated_mean(mu, sd, lower)
    assert np.isfinite(got).all()
    assert got[0] >= alpha
    assert got == pytest.approx(alpha + 1.0 / alpha, rel=1e-3)


@pytest.mark.known_bug
def test_truncated_mean_is_continuous_across_switchover(model):
    """No jump discontinuity in the imputation map.

    A discontinuity larger than ``tol`` can stall or oscillate the EM fixed
    point for any censored row sitting near the switchover.
    """
    grid = np.linspace(7.0, 9.0, 4001)
    vals = model._truncated_mean(np.zeros_like(grid), np.ones_like(grid), grid)
    jumps = np.abs(np.diff(vals))
    # lambda(alpha) is smooth here, so every step should be close to the
    # median step; a switchover discontinuity shows up as a lone spike.
    assert jumps.max() < 3.0 * np.median(jumps), (
        f"largest step = {jumps.max():.3e}, median step = {np.median(jumps):.3e}"
    )


def test_truncated_mean_is_monotone_in_lower_bound(model):
    grid = np.linspace(-5.0, 35.0, 2001)
    vals = model._truncated_mean(np.zeros_like(grid), np.ones_like(grid), grid)
    assert np.all(np.diff(vals) > 0)


@pytest.mark.parametrize("alpha", np.linspace(-5.0, 35.0, 41))
def test_truncated_mean_never_below_lower_bound(model, alpha):
    mu, sd = np.array([0.0]), np.array([1.0])
    got = model._truncated_mean(mu, sd, np.array([float(alpha)]))
    assert got[0] >= alpha - 1e-12


def test_truncated_mean_respects_location_and_scale(model):
    """E[Y|Y>=l] for Y~N(mu, sd^2) must scale affinely."""
    mu = np.array([3.0, -2.0, 10.0])
    sd = np.array([0.5, 2.0, 0.1])
    lower = np.array([3.5, -1.0, 10.05])
    got = model._truncated_mean(mu, sd, lower)
    want = _exact_truncated_mean(mu, sd, lower)
    assert got == pytest.approx(want, rel=1e-10)


def test_truncated_mean_far_below_limit_returns_untruncated_mean(model):
    """When the bound is far below the mean, truncation is a no-op."""
    got = model._truncated_mean(np.array([0.0]), np.array([1.0]), np.array([-40.0]))
    assert got == pytest.approx(0.0, abs=1e-9)


def test_truncated_mean_vectorised_matches_elementwise(model):
    rng = np.random.default_rng(0)
    mu = rng.normal(size=50)
    sd = rng.uniform(0.1, 3.0, size=50)
    lower = mu + rng.uniform(-2.0, 5.0, size=50) * sd
    batch = model._truncated_mean(mu, sd, lower)
    single = np.array(
        [model._truncated_mean(mu[i : i + 1], sd[i : i + 1], lower[i : i + 1])[0] for i in range(50)]
    )
    assert batch == pytest.approx(single, rel=1e-12)


def test_min_std_clamp_prevents_division_by_zero(model):
    got = model._truncated_mean(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert np.isfinite(got).all()


@pytest.mark.known_bug
def test_min_std_of_zero_is_rejected():
    """``min_std=0`` silently yields NaN imputations; it must be validated."""
    with pytest.raises(ValueError):
        CensoredEMGPModel(base_model=object(), censoring_limit=0.0, min_std=0.0)


# ---------------------------------------------------------------------------
# log survival helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0])
def test_normal_log_survival_matches_scipy(model, x):
    import torch

    got = float(model._normal_log_survival(torch.tensor([x], dtype=torch.float64))[0])
    assert got == pytest.approx(norm.logsf(x), rel=1e-10, abs=1e-10)


def test_normal_pdf_matches_scipy(model):
    import torch

    xs = np.linspace(-6.0, 6.0, 101)
    got = model._normal_pdf(torch.as_tensor(xs, dtype=torch.float64)).numpy()
    assert got == pytest.approx(norm.pdf(xs), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# truncated sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [-2.0, 0.0, 1.0, 3.0, 5.0])
def test_truncated_sample_respects_lower_bound(alpha):
    m = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=1)
    n = 20000
    s = m._truncated_sample(np.zeros(n), np.ones(n), np.full(n, float(alpha)))
    assert s.min() >= alpha - 1e-9
    assert np.isfinite(s).all()


@pytest.mark.parametrize("alpha", [0.0, 1.5, 3.0])
def test_truncated_sample_moments_match_scipy(alpha):
    m = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=99)
    n = 400000
    s = m._truncated_sample(np.zeros(n), np.ones(n), np.full(n, float(alpha)))
    exact_mean = truncnorm.mean(alpha, np.inf)
    exact_sd = truncnorm.std(alpha, np.inf)
    # 5 standard errors on the mean; generous but still a real check
    assert abs(s.mean() - exact_mean) < 5.0 * exact_sd / np.sqrt(n)
    assert s.std() == pytest.approx(exact_sd, rel=0.02)


def test_truncated_sample_is_seeded():
    a = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=7)
    b = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=7)
    args = (np.zeros(100), np.ones(100), np.zeros(100))
    assert np.allclose(a._truncated_sample(*args), b._truncated_sample(*args))


def test_truncated_sample_different_seeds_differ():
    a = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=1)
    b = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=2)
    args = (np.zeros(100), np.ones(100), np.zeros(100))
    assert not np.allclose(a._truncated_sample(*args), b._truncated_sample(*args))


def test_truncated_sample_respects_location_and_scale():
    m = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, random_state=5)
    n = 200000
    mu, sd, lo = 4.0, 2.5, 5.0
    s = m._truncated_sample(np.full(n, mu), np.full(n, sd), np.full(n, lo))
    assert s.min() >= lo - 1e-9
    exact = truncnorm.mean(a=(lo - mu) / sd, b=np.inf, loc=mu, scale=sd)
    assert s.mean() == pytest.approx(exact, rel=0.01)


# ---------------------------------------------------------------------------
# _as_1d
# ---------------------------------------------------------------------------
def test_as_1d_broadcasts_scalar():
    out = CensoredEMGPModel._as_1d(3.0, 5, "x")
    assert out.shape == (5,)
    assert np.all(out == 3.0)


def test_as_1d_scalar_without_n():
    assert CensoredEMGPModel._as_1d(3.0).shape == (1,)


def test_as_1d_flattens_column_vector():
    out = CensoredEMGPModel._as_1d(np.arange(4).reshape(-1, 1), 4, "x")
    assert out.shape == (4,)


def test_as_1d_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected 4"):
        CensoredEMGPModel._as_1d([1.0, 2.0], 4, "censoring_limit")


# ---------------------------------------------------------------------------
# prediction unpacking
# ---------------------------------------------------------------------------
def test_extract_mean_std_from_tuple(model):
    mu, sd = model._extract_mean_std((np.array([1.0, 2.0]), np.array([0.5, 0.25])))
    assert mu == pytest.approx([1.0, 2.0])
    assert sd == pytest.approx([0.5, 0.25])


def test_extract_mean_std_from_torch_tensors(model):
    import torch

    mu, sd = model._extract_mean_std((torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.25])))
    assert isinstance(mu, np.ndarray) and isinstance(sd, np.ndarray)
    assert mu == pytest.approx([1.0, 2.0])


def test_extract_mean_std_from_distribution_with_stddev(model):
    import torch

    class D:
        mean = torch.tensor([1.0])
        stddev = torch.tensor([2.0])

    mu, sd = model._extract_mean_std(D())
    assert sd == pytest.approx([2.0])


def test_extract_mean_std_from_distribution_with_variance(model):
    import torch

    class D:
        mean = torch.tensor([1.0])
        variance = torch.tensor([9.0])

    mu, sd = model._extract_mean_std(D())
    assert sd == pytest.approx([3.0])


def test_extract_mean_std_rejects_unsupported(model):
    with pytest.raises(TypeError):
        model._extract_mean_std(object())


def test_extract_mean_std_rejects_mean_without_uncertainty(model):
    class D:
        mean = np.array([1.0])

    with pytest.raises(TypeError):
        model._extract_mean_std(D())


def test_extract_mean_std_rejects_length_mismatch(model):
    with pytest.raises(ValueError, match="lengths differ"):
        model._extract_mean_std((np.array([1.0, 2.0]), np.array([0.5])))


def test_extract_mean_std_adds_predictive_noise_in_quadrature():
    m = CensoredEMGPModel(base_model=object(), censoring_limit=0.0, predictive_noise=4.0)
    _, sd = m._extract_mean_std((np.array([0.0]), np.array([3.0])))
    assert sd == pytest.approx([5.0])


def test_extract_mean_std_clamps_negative_std(model):
    _, sd = model._extract_mean_std((np.array([0.0]), np.array([-2.0])))
    assert sd[0] >= model.min_std


def test_extract_mean_std_enforces_min_std(model):
    _, sd = model._extract_mean_std((np.array([0.0]), np.array([0.0])))
    assert sd[0] == pytest.approx(model.min_std)
