"""Integration and statistical-validity tests against a real GPyTorch surrogate.

Marked ``slow``: run with ``pytest --run-slow``. These are the tests that
demonstrate whether the EM actually buys anything on real data, and whether it
stays numerically stable when the surrogate refits its hyperparameters at every
iteration.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from ._cgp_import import CensoredEMGPModel, load_gp_model
from .conftest import RecordingModel

pytestmark = [pytest.mark.slow, pytest.mark.integration]

gpytorch = pytest.importorskip("gpytorch")
pytest.importorskip("botorch")


@pytest.fixture
def gp_factory():
    GPModel = load_gp_model()

    def _make():
        return GPModel(kernel=gpytorch.kernels.RBFKernel(), show_progression=False)

    return _make


def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ===========================================================================
# does it fit at all
# ===========================================================================
def test_fits_and_predicts_with_a_real_gp(gp_factory, censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                          max_iter=10, damping=0.7)
    m.fit(d["X"], d["y_obs"])
    mu, sd = m.predict(d["X"])
    assert mu.shape == (len(d["X"]),)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sd))
    assert np.all(sd > 0)


def test_censoring_mask_matches_the_data(gp_factory, censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"], max_iter=3)
    m.fit(d["X"], d["y_obs"])
    assert m.em_result_.censored_mask.tolist() == d["censored"].tolist()


def test_uncensored_rows_untouched_with_a_real_gp(gp_factory, censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"], max_iter=8)
    m.fit(d["X"], d["y_obs"])
    keep = ~m.em_result_.censored_mask
    assert m.em_result_.imputed_y[keep] == pytest.approx(d["y_obs"][keep])


# ===========================================================================
# does it actually help
# ===========================================================================
def test_em_recovers_latent_values_better_than_threshold_substitution(
    gp_factory, censored_regression_data
):
    """The whole point of Schmee-Hahn: beat naive y = limit substitution."""
    d = censored_regression_data
    cens = d["censored"]

    naive = gp_factory()
    naive.fit(d["X"], d["y_obs"])
    naive_mu, _ = naive.predict(d["X"])

    em = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                           max_iter=20, damping=0.7)
    em.fit(d["X"], d["y_obs"])
    em_mu, _ = em.predict(d["X"])

    naive_imputation_mse = _mse(d["y_obs"][cens], d["y_true"][cens])
    em_imputation_mse = _mse(em.em_result_.imputed_y[cens], d["y_true"][cens])
    assert em_imputation_mse < naive_imputation_mse, (
        f"EM {em_imputation_mse:.4f} did not beat naive {naive_imputation_mse:.4f}"
    )
    assert _mse(em_mu, d["y_true"]) < _mse(naive_mu, d["y_true"])


def test_em_does_not_degrade_uncensored_predictions(gp_factory, censored_regression_data):
    d = censored_regression_data
    keep = ~d["censored"]

    naive = gp_factory()
    naive.fit(d["X"], d["y_obs"])
    naive_mu, _ = naive.predict(d["X"])

    em = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                           max_iter=20, damping=0.7)
    em.fit(d["X"], d["y_obs"])
    em_mu, _ = em.predict(d["X"])

    assert _mse(em_mu[keep], d["y_true"][keep]) <= 1.5 * _mse(
        naive_mu[keep], d["y_true"][keep]
    )


def test_no_censoring_is_equivalent_to_a_plain_fit(gp_factory, censored_regression_data):
    d = censored_regression_data
    plain = gp_factory()
    plain.fit(d["X"], d["y_true"])
    plain_mu, _ = plain.predict(d["X"])

    wrapped = CensoredEMGPModel(gp_factory(), censoring_limit=1e9, max_iter=10)
    wrapped.fit(d["X"], d["y_true"])
    wrapped_mu, _ = wrapped.predict(d["X"])

    assert wrapped.em_result_.n_iter == 0
    assert wrapped_mu == pytest.approx(plain_mu, abs=0.2)


# ===========================================================================
# stability under repeated refitting - the real risk
# ===========================================================================
@pytest.mark.known_bug
@pytest.mark.parametrize("max_iter", [10, 30, 60])
def test_em_survives_many_iterations_on_a_real_gp(gp_factory, censored_regression_data,
                                                  max_iter):
    """EM must not crash as the imputations settle.

    ``GPModel`` keeps one kernel instance across ``fit`` calls, so each refit
    restarts ``fit_gpytorch_mll`` from the previously converged optimum. When
    the targets barely change - which is exactly what happens as EM
    approaches its fixed point - ``scipy_minimize`` terminates ABNORMAL and
    botorch raises ``ModelFittingError``. Longer EM runs are therefore *more*
    likely to fail, not less.
    """
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                          max_iter=max_iter, damping=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(d["X"], d["y_obs"])   # must not raise ModelFittingError
    imputed = m.em_result_.imputed_y[m.em_result_.censored_mask]
    assert np.all(np.isfinite(imputed))
    span = float(np.ptp(d["y_true"]))
    assert imputed.max() < d["limit"] + 5.0 * span, (
        f"imputations reached {imputed.max():.3g} (limit {d['limit']:.3g})"
    )


@pytest.mark.known_bug
def test_gpmodel_can_be_refitted_on_unchanged_targets(gp_factory,
                                                      censored_regression_data):
    """Pre-existing mobius defect, isolated from the EM wrapper.

    ``GPModel.fit`` is effectively single-use: the kernel instance persists,
    so a second fit on the same data starts at the optimum and raises
    ``ModelFittingError``. Any caller that refits a surrogate - EM, a Planner
    reusing surrogate objects across BO rounds - is exposed to this.
    """
    d = censored_regression_data
    gp = gp_factory()
    gp.fit(d["X"], d["y_obs"])
    gp.fit(d["X"], d["y_obs"])   # must not raise


@pytest.mark.known_bug
def test_repeated_gpmodel_fits_are_stable(gp_factory, censored_regression_data):
    """Five refits with barely-changing targets, as EM does near convergence."""
    d = censored_regression_data
    gp = gp_factory()
    y = d["y_obs"].copy()
    for i in range(5):
        y = y + 1e-10 * (i + 1)
        gp.fit(d["X"], y)


@pytest.mark.known_bug
def test_default_settings_converge_on_a_real_gp(gp_factory, censored_regression_data):
    """Defaults (max_iter=30, tol=1e-4, damping=0.7) should reach the fixed point.

    They do not on a real GP, and nothing is emitted to say so.
    """
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"])
    m.fit(d["X"], d["y_obs"])
    assert m.em_result_.converged, (
        f"max_change={m.em_result_.max_change:.3e} after "
        f"{m.em_result_.n_iter} iterations"
    )


@pytest.mark.known_bug
def test_damping_slows_the_approach_to_the_fixed_point(gp_factory,
                                                       censored_regression_data):
    """Lower damping must move the imputations less per iteration.

    Fails on the reviewed implementation for an unrelated reason: an EM refit
    raises ``ModelFittingError`` and nothing contains it, so the run aborts
    before the comparison can be made.

    Note this is asserted on the *applied* displacement, not on
    ``max_change``: once convergence is judged on the undamped step (so that
    ``tol`` is damping-invariant), ``max_change`` is the raw proposal gap and
    is not monotone in damping.
    """
    d = censored_regression_data
    moved = {}
    for damping in (1.0, 0.3):
        m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                              max_iter=4, damping=damping, tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(d["X"], d["y_obs"])
        cens = m.em_result_.censored_mask
        moved[damping] = float(
            np.max(np.abs(m.em_result_.imputed_y[cens] - d["limit"]))
        )
    assert moved[0.3] < moved[1.0]


def test_hyperparameters_are_restarted_between_em_refits(gp_factory,
                                                         censored_regression_data):
    """Each EM refit should start from the same hyperparameters, not the last
    converged ones, so ``fit_gpytorch_mll`` is never restarted at its optimum."""
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"], max_iter=8,
                          damping=1.0)
    if "restart_hyperparameters" not in inspect.signature(CensoredEMGPModel).parameters:
        pytest.skip("wrapper has no hyperparameter-restart option")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(d["X"], d["y_obs"])
    assert np.all(np.isfinite(m.em_result_.imputed_y))


def test_sample_mode_runs_on_a_real_gp(gp_factory, censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                          method="sample", max_iter=6, random_state=0)
    m.fit(d["X"], d["y_obs"])
    cens = m.em_result_.censored_mask
    assert np.all(m.em_result_.imputed_y[cens] >= d["limit"] - 1e-9)


def test_predictive_noise_widens_the_imputations(gp_factory, censored_regression_data):
    d = censored_regression_data
    out = {}
    for noise in (0.0, 1.0):
        m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"],
                              max_iter=1, damping=1.0, predictive_noise=noise,
                              max_imputation_sd=None)
        m.fit(d["X"], d["y_obs"])
        out[noise] = m.em_result_.imputed_y[m.em_result_.censored_mask].mean()
    assert out[1.0] > out[0.0]


# ===========================================================================
# wrapped-model surface with a real GP
# ===========================================================================
def test_delegated_attributes_on_a_real_gp(gp_factory, censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"], max_iter=3)
    m.fit(d["X"], d["y_obs"])
    assert np.asarray(m.X_train).shape == d["X"].shape
    assert len(m.y_train) == len(d["y_obs"])
    assert str(m.device) in {"cpu", "cuda"}
    assert np.isfinite(m.score(d["X"], d["y_obs"]))


@pytest.mark.known_bug
def test_clone_of_a_fitted_gp_wrapper_is_usable(gp_factory, censored_regression_data):
    """A clone must be refittable.

    ``clone`` deep-copies the *fitted* kernel, so the clone's first fit
    restarts ``fit_gpytorch_mll`` at the previous optimum and raises.
    """
    d = censored_regression_data
    m = CensoredEMGPModel(gp_factory(), censoring_limit=d["limit"], max_iter=3)
    m.fit(d["X"], d["y_obs"])
    c = m.clone()
    c.fit(d["X"], d["y_obs"])
    assert np.all(np.isfinite(c.em_result_.imputed_y))
