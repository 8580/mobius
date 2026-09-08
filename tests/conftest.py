"""Shared fixtures and fake surrogate models for the censored-EM test suite.

Marker policy
-------------
``known_bug``
    Asserts the *correct* behaviour for a defect that is present in the
    implementation as reviewed. These fail on the unpatched module and pass
    once the corresponding fix lands. Run ``pytest -m "not known_bug"`` for a
    green regression baseline on the current code.
``slow``
    Fits a real GPyTorch/BoTorch model. Deselected unless ``--run-slow``.
``integration``
    Exercises the contract between the wrapper and the rest of mobius.
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------
# markers / options
# --------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests that fit real GPyTorch models",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "known_bug: asserts correct behaviour for a known defect"
    )
    config.addinivalue_line("markers", "slow: fits a real GP; needs --run-slow")
    config.addinivalue_line("markers", "integration: mobius API contract test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# --------------------------------------------------------------------------
# fake surrogate models
# --------------------------------------------------------------------------
class RecordingModel:
    """Minimal surrogate honouring the mobius fit/predict contract.

    Returns a constant posterior so the EM fixed point is analytic:
    ``E[Y | Y >= l]`` for ``Y ~ N(mean, sd**2)``.
    """

    def __init__(self, mean=0.0, sd=1.0, accepts_y_noise=True):
        self.mean = float(mean)
        self.sd = float(sd)
        self.accepts_y_noise = accepts_y_noise
        self._X_train = None
        self._y_train = None
        self._model = None
        self.fit_calls = 0
        self.predict_calls = 0
        self.fit_y_history = []
        self.fit_X_history = []
        self.last_y_noise = "<never set>"
        self.last_fit_kwargs = None

    def fit(self, X_train, y_train, y_noise=None, **kwargs):
        if not self.accepts_y_noise and y_noise is not None:
            raise TypeError("this model does not accept y_noise")
        self.fit_calls += 1
        self.fit_X_history.append(np.asarray(X_train, dtype=object))
        arr = np.asarray(y_train, dtype=float)
        self.fit_y_history.append(arr.copy())
        self.last_y_noise = y_noise
        self.last_fit_kwargs = dict(kwargs)
        self._y_train = arr.reshape(-1)
        self._X_train = X_train
        self._model = "fitted"

    def predict(self, X_test, y_noise=None):
        self.predict_calls += 1
        n = len(np.atleast_1d(np.asarray(X_test, dtype=object)))
        return np.full(n, self.mean), np.full(n, self.sd)

    # mobius surrogate surface used by the acquisition functions
    @property
    def y_train(self):
        return self._y_train

    @property
    def X_train(self):
        return self._X_train

    @property
    def is_fitted(self):
        return self._y_train is not None

    def score(self, X_test, y_test):
        return 1.0


class DriftingModel(RecordingModel):
    """Posterior mean tracks the mean of the training targets.

    Reproduces the positive-feedback loop that makes Schmee-Hahn EM diverge
    when the surrogate is refit on its own imputations.
    """

    def fit(self, X_train, y_train, y_noise=None, **kwargs):
        super().fit(X_train, y_train, y_noise=y_noise, **kwargs)
        self.mean = float(np.mean(self._y_train))


class VarianceReturningModel(RecordingModel):
    """Returns (mean, variance) rather than (mean, std)."""

    def predict(self, X_test, y_noise=None):
        self.predict_calls += 1
        n = len(np.atleast_1d(np.asarray(X_test, dtype=object)))
        return np.full(n, self.mean), np.full(n, self.sd**2)


class DistributionModel(RecordingModel):
    """Returns an object exposing ``mean``/``stddev`` like a GPyTorch MVN."""

    class _Dist:
        def __init__(self, mean, stddev, expose="stddev"):
            self.mean = mean
            if expose == "stddev":
                self.stddev = stddev
            else:
                self.variance = stddev**2

    def __init__(self, *args, expose="stddev", **kwargs):
        super().__init__(*args, **kwargs)
        self.expose = expose

    def predict(self, X_test, y_noise=None):
        self.predict_calls += 1
        n = len(np.atleast_1d(np.asarray(X_test, dtype=object)))
        return self._Dist(np.full(n, self.mean), np.full(n, self.sd), self.expose)


class ExplodingModel(RecordingModel):
    """Raises on the n-th ``fit`` call, mimicking a Cholesky failure."""

    def __init__(self, *args, fail_on=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on

    def fit(self, X_train, y_train, y_noise=None, **kwargs):
        if self.fit_calls + 1 >= self.fail_on:
            self.fit_calls += 1
            raise RuntimeError("simulated Cholesky failure")
        super().fit(X_train, y_train, y_noise=y_noise, **kwargs)


class NaNModel(RecordingModel):
    def predict(self, X_test, y_noise=None):
        self.predict_calls += 1
        n = len(np.atleast_1d(np.asarray(X_test, dtype=object)))
        mu = np.full(n, self.mean)
        if n:
            mu[-1] = np.nan
        return mu, np.full(n, self.sd)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def recording_model():
    return RecordingModel()


@pytest.fixture
def simple_dataset():
    """4 rows, 2 of them right-censored at 1.0."""
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0.0, 0.25, 1.0, 1.0])
    return X, y, 1.0


@pytest.fixture
def censored_regression_data():
    """Deterministic latent regression with ~35% right censoring."""
    rng = np.random.default_rng(20240612)
    n = 60
    X = rng.uniform(-3.0, 3.0, size=(n, 2))
    latent = 0.9 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 0] * X[:, 1]
    y_true = latent + rng.normal(0.0, 0.15, size=n)
    limit = float(np.quantile(y_true, 0.65))
    y_obs = np.where(y_true >= limit, limit, y_true)
    return {
        "X": X.astype(np.float32),
        "y_true": y_true,
        "y_obs": y_obs.astype(float),
        "limit": limit,
        "censored": y_true >= limit,
    }


@pytest.fixture
def helm_sequences():
    return [
        "PEPTIDE1{A.C.D.E}$$$$",
        "PEPTIDE1{A.C.D.F}$$$$",
        "PEPTIDE1{A.C.D.G}$$$$",
        "PEPTIDE1{A.C.D.H}$$$$",
    ]
