"""Contract between ``CensoredEMGPModel`` and the rest of mobius.

These are the tests that catch "the maths is right but it cannot be plugged
in" failures - which is exactly the state the wrapper is in as reviewed.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from ._cgp_import import CensoredEMGPModel, IMPORTED_VIA_PACKAGE
from .conftest import RecordingModel

pytestmark = pytest.mark.integration


# ===========================================================================
# the exact call signature mobius uses
# ===========================================================================
@pytest.mark.known_bug
def test_fit_accepts_positional_y_noise(simple_dataset):
    """``_AcquisitionFunction.fit`` always passes ``y_noise`` positionally.

        surrogate_model.fit(X_train, y_train[:, i],
                            y_noise[:, i] if y_noise is not None else None)

    The wrapper declares ``fit(self, X, y, *, ...)``, so every acquisition
    function in mobius raises ``TypeError`` on the first call. This is the
    blocking defect: the wrapper cannot currently be used in a Planner.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, y, None)  # must not raise


@pytest.mark.known_bug
def test_fit_forwards_y_noise_to_the_base_model(simple_dataset):
    """Known-noise workflows must keep their noise model.

    ``GPModel.fit`` switches to ``FixedNoiseGaussianLikelihood`` when
    ``y_noise`` is supplied, which changes the predictive sd the EM step
    conditions on. Dropping it silently changes the imputations.
    """
    X, y, limit = simple_dataset
    base = RecordingModel()
    noise = np.full(4, 0.04)
    m = CensoredEMGPModel(base, censoring_limit=limit, max_iter=2)
    m.fit(X, y, noise)
    assert base.last_y_noise is not None, "y_noise was swallowed by the wrapper"
    assert np.asarray(base.last_y_noise) == pytest.approx(noise)


@pytest.mark.known_bug
def test_fit_signature_is_compatible_with_the_acquisition_contract():
    sig = inspect.signature(CensoredEMGPModel.fit)
    params = list(sig.parameters.values())[1:]  # drop self
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) >= 3, (
        "fit must accept (X_train, y_train, y_noise) positionally to match "
        "mobius._AcquisitionFunction.fit"
    )


def test_predict_returns_mean_and_std(simple_dataset):
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    out = m.predict(X)
    assert isinstance(out, tuple) and len(out) == 2
    mu, sd = out
    assert len(mu) == len(X) and len(sd) == len(X)


def test_exposes_the_surrogate_surface_used_by_acquisitions(simple_dataset):
    """EI/LogEI/PI touch ``y_train`` and ``predict``; Pool also touches ``score``."""
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=2)
    m.fit(X, y)
    for attr in ("fit", "predict", "y_train", "X_train", "score"):
        assert hasattr(m, attr), f"missing {attr}"


def test_y_train_reflects_the_imputed_targets(simple_dataset):
    """Documented, deliberate behaviour - pinned so it cannot drift silently.

    ``y_train`` delegates to the base model, which was fitted on the imputed
    targets. EI/LogEI therefore compute ``best_f`` from imputed values. Under
    ``maximize=True`` that pushes the incumbent above anything observed.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=4, damping=1.0)
    m.fit(X, y)
    assert np.max(m.y_train) > np.max(y)
    assert np.max(m.y_train) == pytest.approx(np.max(m.em_result_.imputed_y))


@pytest.mark.known_bug
def test_raw_targets_are_retrievable(simple_dataset):
    """The uncensored observations must remain recoverable after fitting.

    Without this a caller cannot compute an incumbent from real measurements
    only, which is the conservative choice for a maximisation objective.
    """
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(), censoring_limit=limit, max_iter=3)
    m.fit(X, y)
    raw = getattr(m.em_result_, "observed_y", None)
    assert raw is not None, "EMResult does not retain the raw observations"
    assert raw == pytest.approx(y)


@pytest.mark.known_bug
def test_is_a_surrogate_model_subclass():
    """Every other surrogate in mobius subclasses ``_SurrogateModel``."""
    sm = pytest.importorskip("mobius.surrogate_models.surrogate_model")
    assert issubclass(CensoredEMGPModel, sm._SurrogateModel)


# ===========================================================================
# packaging / exports
# ===========================================================================
@pytest.mark.skipif(not IMPORTED_VIA_PACKAGE,
                    reason="full mobius dependency stack not installed")
@pytest.mark.known_bug
def test_exported_from_the_surrogate_models_subpackage():
    import mobius.surrogate_models as sm

    assert hasattr(sm, "CensoredEMGPModel")
    assert "CensoredEMGPModel" in sm.__all__


@pytest.mark.skipif(not IMPORTED_VIA_PACKAGE,
                    reason="full mobius dependency stack not installed")
@pytest.mark.known_bug
def test_exported_from_the_top_level_package():
    import mobius

    assert hasattr(mobius, "CensoredEMGPModel")
    assert "CensoredEMGPModel" in mobius.__all__


@pytest.mark.skipif(not IMPORTED_VIA_PACKAGE,
                    reason="full mobius dependency stack not installed")
@pytest.mark.known_bug
def test_top_level_all_entries_all_resolve():
    """Pre-existing packaging defects in ``mobius/__init__.py``.

    * a missing comma concatenates ``generate_random_linear_polymers`` and
      ``ic50_to_pic50`` into one bogus name;
    * ``plot_results`` and ``VinaScorer`` are exported but never imported,
      so ``from mobius import *`` raises ``AttributeError``.
    """
    import mobius

    missing = [n for n in mobius.__all__ if not hasattr(mobius, n)]
    assert not missing, f"__all__ names with no attribute: {missing}"


@pytest.mark.skipif(not IMPORTED_VIA_PACKAGE,
                    reason="full mobius dependency stack not installed")
def test_no_duplicate_entries_in_top_level_all():
    import mobius

    seen = [n for n in mobius.__all__ if mobius.__all__.count(n) > 1]
    assert not seen, f"duplicated __all__ entries: {sorted(set(seen))}"


# ===========================================================================
# end-to-end through a real mobius acquisition function
# ===========================================================================
@pytest.mark.known_bug
def test_works_inside_a_mobius_acquisition_function(simple_dataset):
    af = pytest.importorskip("mobius.acquisition_functions")
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=3)
    acq = af.ExpectedImprovement(m, maximize=True)
    acq.fit(X, y.reshape(-1, 1))          # <- passes y_noise positionally
    scores = acq.forward(X)
    assert scores.shape == (len(X), 1)
    assert np.all(np.isfinite(scores))


@pytest.mark.known_bug
@pytest.mark.parametrize("acq_name", ["ExpectedImprovement", "LogExpectedImprovement",
                                      "ProbabilityOfImprovement", "PosteriorMean",
                                      "LowerUpperConfidenceBound"])
def test_all_acquisition_functions_accept_the_wrapper(simple_dataset, acq_name):
    af = pytest.importorskip("mobius.acquisition_functions")
    X, y, limit = simple_dataset
    m = CensoredEMGPModel(RecordingModel(mean=0.0, sd=1.0),
                          censoring_limit=limit, max_iter=2)
    acq = getattr(af, acq_name)(m, maximize=True)
    acq.fit(X, y.reshape(-1, 1))
    assert np.all(np.isfinite(acq.forward(X)))


def test_cached_embedder_discovery_sees_through_the_wrapper(simple_dataset):
    """``CachedSequenceGA`` looks for ``surrogate._pretrained_model``.

    Attribute delegation must expose it so pre-warming still works.
    """
    class WithEmbedder(RecordingModel):
        def __init__(self):
            super().__init__()
            self._pretrained_model = object()

    base = WithEmbedder()
    m = CensoredEMGPModel(base, censoring_limit=1.0)
    assert getattr(m, "_pretrained_model", None) is base._pretrained_model
