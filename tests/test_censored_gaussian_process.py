"""Tests for :class:`mobius.surrogate_models.CensoredEMGPModel`.

The interesting assertions are the two behavioural ones:

* :func:`test_in_sample_e_step_collapses_onto_naive_substitution` documents the
  failure mode the refactor exists to fix, and
* :func:`test_loo_e_step_recovers_the_censored_values` shows that it is fixed.

Everything else is contract-level: signatures, censoring directions, fault
tolerance and the surrogate-model protocol.
"""
from __future__ import annotations

import warnings

import gpytorch
import numpy as np
import pytest

from mobius.surrogate_models.censored_gaussian_process import (
    CensoredEMGPModel,
    SchmeeHahnGPModel,
    exact_gp_loo_predictive,
)
from mobius.surrogate_models.gaussian_process import GPModel
from mobius.surrogate_models.surrogate_model import _SurrogateModel


def _gp(**kwargs):
    return GPModel(kernel=gpytorch.kernels.MaternKernel(nu=2.5), show_progression=False, **kwargs)


def _standardise(y):
    """Centre and scale targets.

    ``GPModel`` fits in float32 via L-BFGS, which reports ABNORMAL termination
    on unstandardised targets often enough to make tests flaky. Standardising is
    good practice for real use too, and is what the example scripts do.
    """
    y = np.asarray(y, dtype=float)
    return (y - y.mean()) / (y.std() or 1.0)


# --------------------------------------------------------------------------- #
# The leave-one-out machinery
# --------------------------------------------------------------------------- #
def test_analytic_loo_matches_brute_force_refit_free_conditioning():
    """Rasmussen & Williams eq. 5.12 against explicit n-fold conditioning.

    Both use the *same* hyperparameters, so this isolates the linear algebra.
    """
    import torch

    rng = np.random.default_rng(0)
    n = 35
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = _standardise(np.sin(1.5 * X[:, 0]) + 0.2 * rng.normal(size=n))
    model = _gp()
    model.fit(X, y)

    mu_loo, sd_loo = exact_gp_loo_predictive(model)
    assert mu_loo.shape == (n,)

    gp, likelihood = model._model, model._likelihood
    gp.train()
    likelihood.train()
    with torch.no_grad():
        prior = likelihood(gp(*gp.train_inputs))
        K = prior.covariance_matrix.double()
        m = prior.mean.double()
    yv = torch.as_tensor(y, dtype=torch.float64)

    for i in range(n):
        rest = [j for j in range(n) if j != i]
        K_aa = K[np.ix_(rest, rest)]
        K_ia = K[i, rest]
        solve = torch.linalg.solve(K_aa, yv[rest] - m[rest])
        mu_ref = float(m[i] + K_ia @ solve)
        var_ref = float(K[i, i] - K_ia @ torch.linalg.solve(K_aa, K_ia))
        assert mu_loo[i] == pytest.approx(mu_ref, abs=1e-4)
        assert sd_loo[i] == pytest.approx(np.sqrt(var_ref), abs=1e-4)


def test_loo_mean_is_independent_of_the_left_out_target():
    """The property that stops the E-step chasing its own imputation.

    Expanding ``y_i - [K^-1 (y - m)]_i / [K^-1]_ii`` cancels the i-th term
    exactly, so perturbing ``y_i`` alone must not move ``mu_-i``.
    """
    import torch

    rng = np.random.default_rng(1)
    n = 25
    X = rng.uniform(-2.0, 2.0, size=(n, 1))
    y = _standardise(np.cos(X[:, 0]) + 0.1 * rng.normal(size=n))
    model = _gp()
    model.fit(X, y)
    mu_a, _ = exact_gp_loo_predictive(model)

    # Change one training target without refitting hyperparameters.
    model._model.train_targets = torch.as_tensor(
        np.concatenate([[y[0] + 5.0], y[1:]]), dtype=model._model.train_targets.dtype
    )
    mu_b, _ = exact_gp_loo_predictive(model)
    assert mu_b[0] == pytest.approx(mu_a[0], abs=1e-3)


def test_loo_uncertainty_exceeds_in_sample_uncertainty():
    """A GP nearly interpolates; that is precisely why in-sample is the wrong signal."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-3.0, 3.0, size=(40, 1))
    y = _standardise(np.sin(1.5 * X[:, 0]) + 0.2 * rng.normal(size=40))
    model = _gp()
    model.fit(X, y)
    _, sd_in = model.predict(X)
    _, sd_loo = exact_gp_loo_predictive(model)
    assert sd_loo.mean() > sd_in.mean()


def test_loo_returns_none_for_a_non_gp_model():
    class Dummy:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X)), np.ones(len(X))

    assert exact_gp_loo_predictive(Dummy()) is None


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_in_sample_e_step_collapses_onto_naive_substitution(censored_dataset):
    """Regression test for the bug the refactor fixes.

    With an in-sample E-step the predictive mean at a censored row is the value
    just imputed there, so the truncated-normal update barely moves and the
    algorithm reports convergence at (essentially) the detection limit.
    """
    X, y_obs, y_true, _, limit, mask = censored_dataset(fraction_censored=0.4, n=70)
    model = CensoredEMGPModel(
        _gp(), censoring_limit=limit, e_step="in_sample",
        variance_correction=False, max_iter=25,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)

    imputed = model.em_result_.imputed_y[mask]
    gap_naive = np.abs(limit - y_true[mask]).mean()
    gap_in_sample = np.abs(imputed - y_true[mask]).mean()
    # It closes less than half of the distance to the truth.
    assert gap_in_sample > 0.5 * gap_naive


def test_loo_e_step_recovers_the_censored_values(censored_dataset):
    """The refactored E-step must beat both naive substitution and the old one."""
    X, y_obs, y_true, _, limit, mask = censored_dataset(fraction_censored=0.25, n=70)

    naive = np.abs(limit - y_true[mask]).mean()
    errors = {}
    for name, kwargs in (
        ("in_sample", dict(e_step="in_sample", variance_correction=False)),
        ("loo", dict(e_step="loo")),
    ):
        model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=30, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y_obs)
        errors[name] = np.abs(model.em_result_.imputed_y[mask] - y_true[mask]).mean()

    assert errors["loo"] < errors["in_sample"] < naive


def test_imputations_respect_the_censoring_bound(censored_dataset):
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.3, n=50)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    result = model.em_result_
    assert np.all(result.imputed_y[mask] >= limit - 1e-9)
    # Uncensored rows must be left completely alone.
    np.testing.assert_allclose(result.imputed_y[~mask], y_obs[~mask])


def test_variance_correction_stops_the_noise_collapsing(censored_dataset):
    """Substituting a conditional mean discards dispersion; the correction restores it."""
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.35, n=70)
    noises = {}
    for flag in (False, True):
        model = CensoredEMGPModel(
            _gp(), censoring_limit=limit, max_iter=20, variance_correction=flag
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y_obs)
        noises[flag] = float(model.base_model._likelihood.noise.mean().detach().cpu())
        if flag:
            assert np.all(model.em_result_.imputed_variance[model.em_result_.censored_mask] > 0)
    assert noises[True] > noises[False]


def test_left_censoring_imputes_downwards():
    rng = np.random.default_rng(5)
    X = rng.uniform(-3.0, 3.0, size=(60, 1))
    y_true = 1.5 * np.sin(X[:, 0]) + 0.15 * rng.normal(size=60)
    limit = float(np.quantile(y_true, 0.25))
    mask = y_true <= limit
    y_obs = np.where(mask, limit, y_true)

    model = CensoredEMGPModel(_gp(), censoring_limit=limit, direction="left", max_iter=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    imputed = model.em_result_.imputed_y[mask]
    assert np.all(imputed <= limit + 1e-9)
    assert imputed.mean() < limit


def test_interval_censoring_stays_inside_its_interval():
    rng = np.random.default_rng(6)
    X = rng.uniform(-2.0, 2.0, size=(40, 1))
    y = _standardise(np.sin(X[:, 0]) + 0.05 * rng.normal(size=40))
    lower = y.copy()
    upper = y.copy()
    lower[:10] = y[:10] - 0.4
    upper[:10] = y[:10] + 0.4

    model = CensoredEMGPModel(_gp(), max_iter=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, lower=lower, upper=upper)
    imputed = model.em_result_.imputed_y[:10]
    assert np.all(imputed >= lower[:10] - 1e-9)
    assert np.all(imputed <= upper[:10] + 1e-9)


def test_no_censored_rows_is_a_plain_fit(censored_dataset):
    X, _, y_true, _, _, _ = censored_dataset()
    model = CensoredEMGPModel(_gp(), censoring_limit=1e9)
    model.fit(X, y_true)
    assert model.em_result_.n_iter == 0
    assert model.em_result_.converged
    np.testing.assert_allclose(model.em_result_.imputed_y, y_true)


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
def test_fit_accepts_the_positional_signature_mobius_uses(censored_dataset):
    """``acquisition_functions._AcquisitionFunction.fit`` calls fit(X, y, y_noise).

    The original wrapper made everything after ``y`` keyword-only, so every
    acquisition function raised TypeError.
    """
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.2, n=40)
    y2d = y_obs.reshape(-1, 1)
    noise2d = np.full_like(y2d, 0.02)

    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Exactly the call the acquisition function makes.
        model.fit(X, y2d[:, 0], noise2d[:, 0])
    mu, sigma = model.predict(X[:5])
    assert mu.shape == (5,) and sigma.shape == (5,)


def test_is_a_surrogate_model_and_exposes_imputed_targets(censored_dataset):
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.25, n=40)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    assert isinstance(model, _SurrogateModel)
    # ``best_f`` in an acquisition function reads y_train; it must see the
    # imputed values, not the pile-up at the detection limit.
    assert model.y_train.max() > limit
    assert model.X_train.shape[0] == len(y_obs)


def test_unknown_attributes_forward_to_the_wrapped_model(censored_dataset):
    base = _gp()
    model = CensoredEMGPModel(base, censoring_limit=0.0)
    assert model.device == base.device
    with pytest.raises(AttributeError):
        _ = model.definitely_not_a_real_attribute


def test_clone_returns_an_unfitted_copy(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.2, n=30)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    fresh = model.clone()
    assert fresh.em_result_ is None
    assert fresh is not model and fresh.base_model is not model.base_model


def test_alias_points_at_the_same_class():
    assert SchmeeHahnGPModel is CensoredEMGPModel


def test_a_failing_m_step_degrades_instead_of_exploding(censored_dataset):
    """A mid-loop fit failure must not destroy the whole run.

    botorch raises ``ModelFittingError`` more often than you would like on data
    with a spike of identical values at the detection limit.
    """
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.3, n=40)

    class Flaky(GPModel):
        calls = 0

        def fit(self, X_train, y_train, y_noise=None):
            Flaky.calls += 1
            if 3 <= Flaky.calls <= 5:
                raise RuntimeError("simulated ModelFittingError")
            return super().fit(X_train, y_train, y_noise)

    base = Flaky(kernel=gpytorch.kernels.MaternKernel(nu=2.5), show_progression=False)
    model = CensoredEMGPModel(base, censoring_limit=limit, max_iter=10, refit_attempts=2)
    with pytest.warns(RuntimeWarning):
        model.fit(X, y_obs)
    mu, sigma = model.predict(X[:3])
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0)


def test_kfold_e_step_works_without_gpytorch_internals(censored_dataset):
    """Model-agnostic fallback: any surrogate with fit/predict is supported."""

    class RidgeLike:
        """Deliberately not a GP - it hides no ``_model``/``_likelihood``."""

        def __init__(self):
            self._w = None
            self._s = 1.0

        def fit(self, X, y):
            A = np.c_[np.ones(len(X)), np.asarray(X)]
            self._w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y)
            self._s = float(np.std(y - A @ self._w)) or 0.1
            return self

        def predict(self, X):
            A = np.c_[np.ones(len(X)), np.asarray(X)]
            return A @ self._w, np.full(len(X), self._s)

    X, y_obs, y_true, _, limit, mask = censored_dataset(fraction_censored=0.25, n=60)
    model = CensoredEMGPModel(RidgeLike(), censoring_limit=limit, max_iter=15, n_splits=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    assert model.em_result_.e_step == "kfold"
    assert np.all(model.em_result_.imputed_y[mask] >= limit - 1e-9)


def test_history_records_per_iteration_diagnostics(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(fraction_censored=0.25, n=40)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit, max_iter=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    history = model.em_result_.history
    assert history and set(history[0]) == {
        "iteration", "max_change", "mean_imputed", "censored_loglik", "mean_predictive_sd",
    }


def test_bad_arguments_are_rejected_early():
    base = _gp()
    for kwargs in (
        dict(method="median"),
        dict(e_step="magic"),
        dict(direction="up"),
        dict(damping=0.0),
        dict(damping=1.5),
        dict(max_iter=0),
        dict(n_splits=1),
    ):
        with pytest.raises(ValueError):
            CensoredEMGPModel(base, **kwargs)


def test_mismatched_lengths_are_rejected(censored_dataset):
    X, y_obs, _, _, limit, _ = censored_dataset(n=30)
    model = CensoredEMGPModel(_gp(), censoring_limit=limit)
    with pytest.raises(ValueError):
        model.fit(X, y_obs[:10])


def test_sample_method_draws_inside_the_truncation_region(censored_dataset):
    X, y_obs, _, _, limit, mask = censored_dataset(fraction_censored=0.25, n=40)
    model = CensoredEMGPModel(
        _gp(), censoring_limit=limit, method="sample", max_iter=6, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y_obs)
    assert np.all(model.em_result_.imputed_y[mask] >= limit - 1e-9)
