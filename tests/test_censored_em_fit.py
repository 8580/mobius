"""Behaviour of ``CensoredEMGPModel.fit``: censoring resolution, the EM loop,
invariants, convergence semantics and failure containment.
"""
from __future__ import annotations

import copy
import inspect
import pickle
import warnings

import numpy as np
import pytest
from scipy.stats import truncnorm

from ._cgp_import import CensoredEMGPModel, EMResult
from .conftest import (
    DistributionModel,
    DriftingModel,
    ExplodingModel,
    NaNModel,
    RecordingModel,
    VarianceReturningModel,
)


# ===========================================================================
# constructor validation
# ===========================================================================
def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be"):
        CensoredEMGPModel(object(), method="median")


@pytest.mark.parametrize("damping", [0.0, -0.1, 1.5, np.nan])
def test_rejects_out_of_range_damping(damping):
    with pytest.raises(ValueError, match="damping"):
        CensoredEMGPModel(object(), damping=damping)


@pytest.mark.parametrize("damping", [1e-6, 0.5, 1.0])
def test_accepts_valid_damping(damping):
    CensoredEMGPModel(object(), damping=damping)


@pytest.mark.known_bug
@pytest.mark.parametrize("max_iter", [0, -1])
def test_rejects_non_positive_max_iter(max_iter):
    """``max_iter=0`` currently raises ``UnboundLocalError`` deep inside fit."""
    with pytest.raises(ValueError, match="max_iter"):
        CensoredEMGPModel(object(), censoring_limit=0.0, max_iter=max_iter)


@pytest.mark.known_bug
def test_max_iter_zero_does_not_raise_unbound_local(simple_dataset):
    X, y, limit = simple_dataset
    try:
        m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=0)
        m.fit(X, y)
    except UnboundLocalError as exc:  # pragma: no cover - the defect
        pytest.fail(f"max_iter=0 leaked an implementation error: {exc}")
    except ValueError:
        pass  # rejecting it up front is the acceptable alternative


# ===========================================================================
# censoring resolution
# ===========================================================================
def test_requires_a_censoring_limit():
    m = CensoredEMGPModel(RecordingModel())
    with pytest.raises(ValueError, match="censoring_limit"):
        m.fit(np.zeros((3, 1)), np.zeros(3))


def test_scalar_limit_is_broadcast(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit)
    mask, limits = m._resolve_censoring(y, None, None)
    assert limits.shape == y.shape
    assert mask.tolist() == [False, False, True, True]


def test_per_row_limits_are_honoured():
    m = CensoredEMGPModel(RecordingModel())
    y = np.array([0.0, 1.0, 2.0, 3.0])
    mask, limits = m._resolve_censoring(y, None, [10.0, 1.0, 10.0, 3.0])
    assert mask.tolist() == [False, True, False, True]


def test_explicit_mask_overrides_auto_detection():
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=1.0)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    mask, _ = m._resolve_censoring(y, [True, False, False, False], None)
    assert mask.tolist() == [True, False, False, False]


def test_fit_level_arguments_override_constructor(simple_dataset):
    X, y, _ = simple_dataset
    base = RecordingModel()
    m = CensoredEMGPModel(base, censoring_limit=99.0, max_iter=2)
    m.fit(X, y, censoring_limit=1.0)
    assert m.em_result_.censored_mask.tolist() == [False, False, True, True]


def test_infinite_limit_means_nothing_is_censored():
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=np.inf)
    mask, _ = m._resolve_censoring(np.array([1.0, 2.0]), None, None)
    assert not mask.any()


def test_censored_row_with_non_finite_limit_is_rejected():
    m = CensoredEMGPModel(RecordingModel())
    with pytest.raises(ValueError, match="finite lower bound"):
        m._resolve_censoring(np.array([1.0, 2.0]), [True, False], [np.inf, 1.0])


def test_wrong_length_limit_is_rejected(simple_dataset):
    X, y, _ = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=[1.0, 2.0])
    with pytest.raises(ValueError, match="expected 4"):
        m.fit(X, y)


def test_wrong_length_mask_is_rejected(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit,
                          censored_mask=[True, False])
    with pytest.raises(ValueError, match="expected 4"):
        m.fit(X, y)


@pytest.mark.known_bug
def test_observations_strictly_above_the_limit_are_flagged(recording_model):
    """Auto-detection silently swallows real observations above the limit.

    A right-censored record is reported *at* the limit. A value well above it
    is either a data error or genuinely uncensored; either way overwriting it
    with an imputation without telling the user is wrong.
    """
    m = CensoredEMGPModel(recording_model, censoring_limit=1.0)
    y = np.array([0.0, 0.5, 1.0, 5.0])
    with pytest.warns(UserWarning, match="above the censoring limit"):
        m._resolve_censoring(y, None, None)


def test_detection_atol_captures_float_noise():
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=1.0, detection_atol=1e-6)
    y = np.array([1.0 - 1e-9, 0.0])
    mask, _ = m._resolve_censoring(y, None, None)
    assert mask.tolist() == [True, False]


# ===========================================================================
# EM loop invariants
# ===========================================================================
def test_uncensored_rows_are_never_modified(censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=d["limit"], max_iter=5)
    m.fit(d["X"], d["y_obs"])
    keep = ~m.em_result_.censored_mask
    assert m.em_result_.imputed_y[keep] == pytest.approx(d["y_obs"][keep])


def test_imputations_never_fall_below_the_limit(censored_regression_data):
    d = censored_regression_data
    m = CensoredEMGPModel(RecordingModel(mean=-5.0, sd=0.5),
                          censoring_limit=d["limit"], max_iter=5)
    m.fit(d["X"], d["y_obs"])
    cens = m.em_result_.censored_mask
    assert np.all(m.em_result_.imputed_y[cens] >= d["limit"] - 1e-12)


def test_imputation_is_initialised_at_the_threshold(simple_dataset):
    X, y, limit = simple_dataset
    base = RecordingModel(mean=0.0, sd=1.0)
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=1)
    m.fit(X, y)
    first_fit_y = base.fit_y_history[0]
    assert first_fit_y[2] == pytest.approx(limit)
    assert first_fit_y[3] == pytest.approx(limit)


def test_converges_to_the_analytic_fixed_point(simple_dataset):
    """With a constant posterior the fixed point is E[Y | Y >= l]."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, damping=1.0,
                          max_iter=200, tol=1e-10, max_imputation_sd=None)
    m.fit(X, y)
    expected = truncnorm.mean(a=limit, b=np.inf, loc=0.0, scale=1.0)
    got = m.em_result_.imputed_y[m.em_result_.censored_mask]
    assert got == pytest.approx(expected, rel=1e-6)
    assert m.em_result_.converged


@pytest.mark.parametrize("damping", [1.0, 0.7, 0.3])
def test_fixed_point_is_damping_invariant(simple_dataset, damping):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, damping=damping,
                          max_iter=5000, tol=1e-12, max_imputation_sd=None)
    m.fit(X, y)
    expected = truncnorm.mean(a=limit, b=np.inf, loc=0.0, scale=1.0)
    assert m.em_result_.imputed_y[m.em_result_.censored_mask] == pytest.approx(
        expected, rel=1e-5
    )


@pytest.mark.known_bug
def test_tolerance_is_measured_on_the_undamped_step(simple_dataset):
    """``tol`` must mean the same thing at every damping value.

    Convergence is currently tested against the *damped* increment, so the run
    stops ~1/damping further from the fixed point as damping shrinks.
    """
    X, y, limit = simple_dataset
    expected = truncnorm.mean(a=limit, b=np.inf, loc=0.0, scale=1.0)
    errors = {}
    for damping in (1.0, 0.5, 0.1):
        m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                              censoring_limit=limit, damping=damping,
                              tol=1e-4, max_iter=10000, max_imputation_sd=None)
        m.fit(X, y)
        assert m.em_result_.converged, f"damping={damping} did not converge"
        val = m.em_result_.imputed_y[m.em_result_.censored_mask][0]
        errors[damping] = abs(val - expected)
    # `tol` is a bound on the step, and the map is a mild contraction here, so
    # the distance to the fixed point should stay within a small multiple of
    # `tol` at every damping value.
    assert max(errors.values()) < 10.0 * 1e-4, (
        f"accuracy at fixed tol varies with damping: {errors}"
    )


def test_no_censoring_skips_the_em_loop(simple_dataset):
    X, y, _ = simple_dataset
    base = RecordingModel()
    m = CensoredEMGPModel(base, censoring_limit=1e9, max_iter=30)
    m.fit(X, y)
    assert base.fit_calls == 1
    assert m.em_result_.n_iter == 0
    assert m.em_result_.converged
    assert m.em_result_.imputed_y == pytest.approx(y)


def test_no_censoring_passes_original_targets_through(simple_dataset):
    X, y, _ = simple_dataset
    base = RecordingModel()
    m = CensoredEMGPModel(base, censoring_limit=1e9)
    m.fit(X, y)
    assert base.fit_y_history[0] == pytest.approx(y)


def test_all_rows_censored(simple_dataset):
    X, _, limit = simple_dataset
    y = np.full(4, limit)
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=50, damping=1.0,
                          tol=1e-10, max_imputation_sd=None)
    m.fit(X, y)
    assert m.em_result_.censored_mask.all()
    assert np.all(m.em_result_.imputed_y >= limit)


def test_single_censored_row(simple_dataset):
    X, y, limit = simple_dataset
    y = y.copy()
    y[3] = 0.5
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    assert m.em_result_.censored_mask.sum() == 1


def test_empty_input():
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=1.0)
    m.fit(np.zeros((0, 2)), np.zeros(0))
    assert m.em_result_.n_iter == 0
    assert m.em_result_.imputed_y.shape == (0,)


def test_model_is_left_fitted_on_the_final_imputation(simple_dataset):
    """The surrogate must not lag one EM step behind ``em_result_``."""
    X, y, limit = simple_dataset
    base = RecordingModel(mean=0.0, sd=1.0)
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=6, damping=1.0)
    m.fit(X, y)
    assert base.fit_y_history[-1] == pytest.approx(m.em_result_.imputed_y)


def test_em_result_fields(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=4)
    m.fit(X, y)
    r = m.em_result_
    assert isinstance(r, EMResult)
    assert r.imputed_y.shape == y.shape
    assert r.censored_mask.dtype == bool
    assert r.lower_bounds.shape == y.shape
    assert isinstance(r.converged, (bool, np.bool_))
    assert 0 <= r.n_iter <= 4
    assert np.isfinite(r.max_change)


def test_fit_returns_self(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    assert m.fit(X, y) is m


def test_refit_resets_state(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    first = m.em_result_.imputed_y.copy()
    m.fit(X, np.zeros_like(y), censoring_limit=1e9)
    assert not np.allclose(first, m.em_result_.imputed_y)
    assert m.em_result_.n_iter == 0


# ===========================================================================
# convergence reporting
# ===========================================================================
def test_converged_flag_is_true_when_tolerance_is_met(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=500, tol=1e-8,
                          damping=1.0, max_imputation_sd=None)
    m.fit(X, y)
    assert m.em_result_.converged
    assert m.em_result_.max_change < 1e-8


def test_converged_flag_is_false_when_iterations_run_out(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(DriftingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=3, tol=1e-12,
                          damping=1.0)
    m.fit(X, y)
    assert not m.em_result_.converged
    assert m.em_result_.n_iter == 3


@pytest.mark.known_bug
def test_non_convergence_emits_a_warning(simple_dataset):
    """A silently un-converged surrogate is the dangerous failure mode.

    ``converged=False`` is recorded on ``em_result_`` but nothing is emitted,
    so a caller who never inspects it gets an arbitrary intermediate fit.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(DriftingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=2, tol=1e-14,
                          damping=1.0)
    with pytest.warns(UserWarning, match="did not converge"):
        m.fit(X, y)


def test_verbose_prints_progress(simple_dataset, capsys):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit,
                          max_iter=2, tol=1e-14, verbose=True)
    m.fit(X, y)
    assert "iteration" in capsys.readouterr().out.lower()


def test_number_of_base_fits_is_bounded(simple_dataset):
    """One fit per EM step plus at most one final refit."""
    X, y, limit = simple_dataset
    base = RecordingModel(mean=0.0, sd=1.0)
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=5, tol=0.0)
    m.fit(X, y)
    assert base.fit_calls <= m.em_result_.n_iter + 1


# ===========================================================================
# stability guards
# ===========================================================================
def test_max_imputation_sd_caps_the_step():
    X = np.zeros((2, 1))
    y = np.array([1.0, 1.0])
    # posterior mean far above the limit -> untruncated mean would be used
    m = CensoredEMGPModel(RecordingModel(mean=100.0, sd=1.0), censoring_limit=1.0,
                          max_iter=1, damping=1.0, max_imputation_sd=2.0)
    m.fit(X, y)
    assert np.all(m.em_result_.imputed_y <= 1.0 + 2.0 * 1.0 + 1e-9)


def test_max_imputation_sd_none_disables_the_cap():
    X = np.zeros((2, 1))
    y = np.array([1.0, 1.0])
    kwargs = dict(censoring_limit=1.0, max_iter=1, damping=1.0,
                  max_imputation_sd=None)
    if "max_total_shift" in inspect.signature(CensoredEMGPModel).parameters:
        kwargs["max_total_shift"] = None
    m = CensoredEMGPModel(RecordingModel(mean=100.0, sd=1.0), **kwargs)
    m.fit(X, y)
    assert np.all(m.em_result_.imputed_y > 50.0)


@pytest.mark.known_bug
def test_runaway_feedback_is_contained(simple_dataset):
    """EM must not diverge when the surrogate is refit on its own imputations.

    ``DriftingModel`` reproduces the real ``GPModel`` behaviour (hyperparameters
    refit each iteration), where imputing upward raises the posterior mean,
    which raises the next imputation. On a real GP this eventually makes
    ``fit_gpytorch_mll`` raise ``ModelFittingError``.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(DriftingModel(mean=0.0, sd=1.0), censoring_limit=limit,
                          max_iter=400, damping=1.0)
    m.fit(X, y)
    imputed = m.em_result_.imputed_y[m.em_result_.censored_mask]
    assert np.all(np.isfinite(imputed))
    assert imputed.max() < limit + 50.0, f"imputations ran away to {imputed.max():.3g}"


@pytest.mark.known_bug
def test_non_finite_predictions_are_rejected(simple_dataset):
    """NaN from the surrogate must not silently become a NaN training target."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(NaNModel(mean=0.0, sd=1.0), censoring_limit=limit, max_iter=2)
    with pytest.raises(ValueError, match="non-finite|not finite|NaN"):
        m.fit(X, y)


@pytest.mark.known_bug
def test_failure_inside_the_loop_invalidates_state(simple_dataset):
    """A crashed refit must not leave a stale ``em_result_`` behind."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    assert m.em_result_ is not None

    m.base_model = ExplodingModel(mean=0.0, sd=1.0, fail_on=1)
    with pytest.raises(RuntimeError):
        m.fit(X, y)
    assert m.em_result_ is None, "stale EM result survived a failed fit"


# ===========================================================================
# input shapes and types
# ===========================================================================
def test_column_vector_targets_round_trip(simple_dataset):
    X, y, limit = simple_dataset
    base = RecordingModel(mean=0.0, sd=1.0)
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=2)
    m.fit(X, y.reshape(-1, 1))
    assert base.fit_y_history[-1].shape == (4, 1)


@pytest.mark.known_bug
def test_multi_output_targets_are_rejected_clearly(simple_dataset):
    """(n, k>1) targets are flattened, then blow up with a numpy IndexError."""
    X, _, limit = simple_dataset
    y = np.column_stack([np.array([0.0, 0.25, 1.0, 1.0]),
                         np.array([1.0, 1.0, 0.0, 0.0])])
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    with pytest.raises(ValueError, match="single|1D|one target|multi"):
        m.fit(X, y)


def test_list_targets_are_accepted(simple_dataset):
    X, _, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, [0.0, 0.25, 1.0, 1.0])
    assert m.em_result_.censored_mask.sum() == 2


def test_string_sequence_inputs(helm_sequences):
    """The actual mobius input type is an array of HELM/FASTA strings."""
    y = np.array([0.0, 0.2, 1.0, 1.0])
    base = RecordingModel(mean=0.0, sd=1.0)
    m = CensoredEMGPModel(base, censoring_limit=1.0, max_iter=3)
    m.fit(np.asarray(helm_sequences), y)
    assert m.em_result_.censored_mask.tolist() == [False, False, True, True]
    assert base.fit_calls >= 2


def test_integer_targets_are_cast(simple_dataset):
    X, _, _ = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=1, max_iter=2)
    m.fit(X, np.array([0, 0, 1, 1]))
    assert m.em_result_.imputed_y.dtype.kind == "f"


@pytest.mark.known_bug
def test_ragged_inputs_raise_a_useful_error():
    """A ragged X currently surfaces a raw numpy broadcasting error."""
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=1.0, max_iter=2)
    with pytest.raises((TypeError, ValueError)) as exc:
        m.fit([[1, 2], [3, 4, 5], [6]], np.array([0.0, 1.0, 1.0]))
    assert "inhomogeneous" not in str(exc.value).lower(), (
        "raw numpy error leaked to the caller"
    )


# ===========================================================================
# prediction unpacking end to end
# ===========================================================================
def test_distribution_style_predictions(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(DistributionModel(mean=0.0, sd=1.0, expose="stddev"),
                          censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    assert np.all(np.isfinite(m.em_result_.imputed_y))


@pytest.mark.known_bug
def test_variance_style_distribution(simple_dataset):
    """A numpy ``variance`` attribute hits ``ndarray.clamp_min`` and raises.

    ``_extract_mean_std`` assumes the variance branch is always a torch
    tensor; ``mobius.surrogate_models.RFModel`` and any numpy-backed surrogate
    break it.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(DistributionModel(mean=0.0, sd=2.0, expose="variance"),
                          censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    assert np.all(np.isfinite(m.em_result_.imputed_y))


def test_predict_delegates_to_the_base_model(simple_dataset):
    X, y, limit = simple_dataset
    base = RecordingModel()
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    before = base.predict_calls
    m.predict(X)
    assert base.predict_calls == before + 1


# ===========================================================================
# reproducibility, cloning, serialisation
# ===========================================================================
def test_mean_method_is_deterministic(censored_regression_data):
    d = censored_regression_data
    out = []
    for _ in range(2):
        m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                              censoring_limit=d["limit"], max_iter=6)
        m.fit(d["X"], d["y_obs"])
        out.append(m.em_result_.imputed_y)
    assert out[0] == pytest.approx(out[1])


def test_sample_method_is_reproducible_with_a_seed(censored_regression_data):
    d = censored_regression_data
    out = []
    for _ in range(2):
        m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                              censoring_limit=d["limit"], max_iter=4,
                              method="sample", random_state=1234)
        m.fit(d["X"], d["y_obs"])
        out.append(m.em_result_.imputed_y)
    assert out[0] == pytest.approx(out[1])


def test_sample_method_differs_between_seeds(censored_regression_data):
    d = censored_regression_data
    out = []
    for seed in (1, 2):
        m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                              censoring_limit=d["limit"], max_iter=4,
                              method="sample", random_state=seed)
        m.fit(d["X"], d["y_obs"])
        out.append(m.em_result_.imputed_y)
    assert not np.allclose(out[0], out[1])


def test_pickle_round_trip(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.em_result_.imputed_y == pytest.approx(m.em_result_.imputed_y)


def test_deepcopy_round_trip(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    m2 = copy.deepcopy(m)
    assert m2.em_result_.imputed_y == pytest.approx(m.em_result_.imputed_y)


@pytest.mark.known_bug
def test_clone_returns_an_unfitted_model(simple_dataset):
    """``clone`` is documented as returning an unfitted copy."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    c = m.clone()
    assert c.em_result_ is None
    assert not c.base_model.is_fitted, "clone carries the fitted base model"


@pytest.mark.known_bug
def test_clones_draw_independent_random_streams(simple_dataset):
    """Independent BO replicates must not share an RNG state."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0), censoring_limit=limit,
                          method="sample", max_iter=3)
    m.fit(X, y)
    a, b = m.clone(), m.clone()
    args = (np.zeros(50), np.ones(50), np.zeros(50))
    assert not np.allclose(a._truncated_sample(*args), b._truncated_sample(*args)), (
        "two clones produced identical draws"
    )


def test_attribute_delegation_to_the_base_model(simple_dataset):
    X, y, limit = simple_dataset
    base = RecordingModel()
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    assert m.X_train is base.X_train
    assert np.isfinite(m.score(X, y))
    with pytest.raises(AttributeError):
        _ = m.definitely_not_an_attribute


def test_getattr_does_not_recurse_on_a_bare_instance():
    m = CensoredEMGPModel.__new__(CensoredEMGPModel)
    with pytest.raises(AttributeError):
        _ = m.base_model
