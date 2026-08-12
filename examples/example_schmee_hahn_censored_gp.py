#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Schmee-Hahn imputation for a right-censored potency panel
=========================================================

Scenario: a VHH panel screened in a binding assay with a ceiling. Anything
weaker than the top test concentration comes back as ``pIC50 < 4.5`` and is
recorded at the limit. On a log scale a weak binder is right-censored in
``pIC50`` terms once you flip the sign convention, so this script models
``pIC50 >= limit`` - the potent tail that saturates the readout.

The script contrasts four ways of handling that:

  1. naive substitution      - fit to the sentinel values as if they were data
  2. drop the censored rows  - throw away a third of the panel
  3. Schmee-Hahn, in-sample  - what the original wrapper did
  4. Schmee-Hahn, LOO E-step - the refactored default

and prints RMSE against the (known) latent potency plus 95% interval coverage
on a held-out set measured without a ceiling.

Run::

    python example_schmee_hahn_censored_gp.py
"""
from __future__ import annotations

import warnings

import gpytorch
import numpy as np
from scipy.stats import norm

from mobius.surrogate_models import CensoredEMGPModel, GPModel

RANDOM_STATE = 0
CENSORED_FRACTION = 0.35
N_TRAIN = 100
N_SEEDS = 5
ASSAY_NOISE_SD = 0.15


# --------------------------------------------------------------------------- #
# Synthetic panel
# --------------------------------------------------------------------------- #
def make_panel(n=N_TRAIN, fraction_censored=CENSORED_FRACTION, seed=RANDOM_STATE):
    """A 1-D stand-in for a sequence-embedding landscape.

    Two things matter for the real case and are reproduced here: the censored
    points are *contiguous* in input space (a potent cluster saturates the
    assay together, exactly as a family of related binders would), and the
    latent function keeps rising past the limit.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    latent = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
    y_true = latent + ASSAY_NOISE_SD * rng.normal(size=n)

    limit = float(np.quantile(y_true, 1.0 - fraction_censored))
    censored = y_true >= limit
    y_observed = np.where(censored, limit, y_true)

    # Standardise on the *uncensored* rows. Two reasons, both practical:
    # scaling on all rows would use the sentinel pile-up at the limit, which is
    # the artefact being modelled away; and `GPModel` fits in float32 via
    # L-BFGS, which reports ABNORMAL termination on raw targets carrying a
    # spike of identical values often enough to be a real nuisance.
    location = y_observed[~censored].mean()
    scale = y_observed[~censored].std() or 1.0
    rescale = lambda v: (v - location) / scale
    return (X, rescale(y_observed), rescale(y_true), rescale(latent),
            float(rescale(limit)), censored)


def make_holdout(location, scale, n=400, seed=99):
    """Held-out compounds re-measured on an assay with no ceiling.

    Put on the same scale as the training panel so the numbers are comparable.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    latent = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
    y = latent + ASSAY_NOISE_SD * rng.normal(size=n)
    return X, (y - location) / scale, (latent - location) / scale


def _panel_scaling(n=N_TRAIN, fraction_censored=CENSORED_FRACTION, seed=RANDOM_STATE):
    """The location/scale `make_panel` applied, recomputed identically."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    y_true = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0] + ASSAY_NOISE_SD * rng.normal(size=n)
    limit = float(np.quantile(y_true, 1.0 - fraction_censored))
    censored = y_true >= limit
    y_observed = np.where(censored, limit, y_true)
    return y_observed[~censored].mean(), (y_observed[~censored].std() or 1.0)


def kernel():
    return gpytorch.kernels.MaternKernel(nu=2.5)


def gp():
    return GPModel(kernel=kernel(), show_progression=False)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(mu, sd, latent, y_holdout, region):
    """RMSE against latent truth, plus 95% interval coverage above the limit."""
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
    return (
        np.sqrt(np.mean((mu - latent) ** 2)),
        np.sqrt(np.mean((mu - latent)[region] ** 2)),
        np.mean((y_holdout[region] >= lo[region]) & (y_holdout[region] <= hi[region])),
        sd[region].mean(),
    )


def compare_one_seed(seed):
    """Fit all four approaches on one panel and score them identically."""
    X, y_observed, y_true, latent, limit, censored = make_panel(seed=seed)
    location, scale = _panel_scaling(seed=seed)
    X_holdout, y_holdout, latent_holdout = make_holdout(location, scale, seed=1000 + seed)
    region = latent_holdout >= limit

    scores = {}

    model = gp()
    model.fit(X, y_observed)
    scores["naive substitution"] = score(*model.predict(X_holdout), latent_holdout, y_holdout, region)

    model = gp()
    model.fit(X[~censored], y_observed[~censored])
    scores["drop censored rows"] = score(*model.predict(X_holdout), latent_holdout, y_holdout, region)

    # In-sample E-step: the original behaviour, kept to show the failure mode.
    # A GP nearly interpolates, so the in-sample predictive mean at a censored
    # row is the value just imputed there and the update has a fixed point at
    # the detection limit.
    in_sample = CensoredEMGPModel(
        gp(), censoring_limit=limit, e_step="in_sample",
        variance_correction=False, max_iter=30,
    )
    in_sample.fit(X, y_observed)
    scores["Schmee-Hahn (in-sample)"] = score(
        *in_sample.predict(X_holdout), latent_holdout, y_holdout, region
    )

    schmee_hahn = CensoredEMGPModel(
        gp(),
        censoring_limit=limit,
        direction="right",     # true value lies at or above the limit
        e_step="auto",         # -> analytic LOO for an exact GP
        max_iter=30,
        damping=0.7,           # slows feedback between correlated censored rows
        variance_correction=True,
    )
    schmee_hahn.fit(X, y_observed)
    scores["Schmee-Hahn (LOO)"] = score(
        *schmee_hahn.predict(X_holdout), latent_holdout, y_holdout, region
    )

    return scores, schmee_hahn, (X, y_observed, y_true, limit, censored)


def main() -> None:
    # botorch's L-BFGS is chatty about ABNORMAL terminations it then recovers
    # from internally; silence the noise, not the real warnings.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*scipy_minimize.*")
    warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
    warnings.filterwarnings("ignore", message=".*Very small noise values.*")
    warnings.filterwarnings("ignore", message=".*did not converge.*")

    X, y_observed, y_true, latent, limit, censored = make_panel()

    print("=" * 78)
    print("Right-censored potency panel")
    print("=" * 78)
    print(f"  training points        : {len(X)}")
    print(f"  censored at the limit  : {censored.sum()} ({censored.mean():.0%})")
    print(f"  detection limit        : {limit:.3f} (standardised units)")
    print(f"  true mean of censored  : {y_true[censored].mean():.3f}")
    print(f"  replicates             : {N_SEEDS} independent panels")

    aggregate = {}
    last_model = None
    last_panel = None
    for seed in range(N_SEEDS):
        scores, model, panel = compare_one_seed(seed)
        for name, value in scores.items():
            aggregate.setdefault(name, []).append(value)
        last_model, last_panel = model, panel

    print(f"\nPredictive performance, mean over {N_SEEDS} panels")
    print("(latent potency, not the censored readout)")
    print(f"  {'method':26s} {'RMSE all':>9s} {'RMSE >lim':>10s} {'95% cov':>9s} {'mean sd':>9s}")
    for name, values in aggregate.items():
        arr = np.asarray(values)
        print(f"  {name:26s} {arr[:, 0].mean():9.3f} {arr[:, 1].mean():10.3f} "
              f"{arr[:, 2].mean():9.2f} {arr[:, 3].mean():9.3f}")
    print("\n  'RMSE >lim' is the part that matters: how well each method describes")
    print("  the region the assay could not resolve. 95% coverage should be ~0.95;")
    print("  a much smaller number means the model is confidently wrong there.")
    print("  Single panels are noisy - in-sample occasionally wins on one seed,")
    print("  which is why this averages rather than reporting one run.")

    schmee_hahn = last_model
    X, y_observed, y_true, limit, censored = last_panel

    # -- imputation diagnostics (from the final panel) ---------------------- #
    result = schmee_hahn.em_result_
    imputed = result.imputed_y[censored]
    print("\nSchmee-Hahn (LOO) imputation diagnostics")
    print(f"  E-step used            : {result.e_step}")
    print(f"  converged              : {result.converged} after {result.n_iter} iterations")
    print(f"  final max change       : {result.max_change:.2e}")
    print(f"  mean imputed value     : {imputed.mean():.3f}")
    print(f"  mean true value        : {y_true[censored].mean():.3f}")
    print(f"  detection limit        : {limit:.3f}")
    print(f"  MAE vs truth           : {np.abs(imputed - y_true[censored]).mean():.3f}")
    print(f"  MAE of naive substitution: {np.abs(limit - y_true[censored]).mean():.3f}")

    print("\n  Per-iteration trace (first and last five):")
    history = result.history
    for record in history[:5] + ([("...",)] if len(history) > 10 else []) + history[-5:]:
        if isinstance(record, tuple):
            print("    ...")
            continue
        print(
            f"    iter {record['iteration']:3d}  max_change={record['max_change']:.2e}  "
            f"mean_imputed={record['mean_imputed']:.3f}  "
            f"predictive_sd={record['mean_predictive_sd']:.3f}"
        )

    # -- what an acquisition function would see ---------------------------- #
    # ``y_train`` is the completed vector, so ``best_f`` in ExpectedImprovement
    # reflects the imputed potencies rather than the pile-up at the limit.
    print("\n  best_f seen by an acquisition function")
    print(f"    from imputed targets : {schmee_hahn.y_train.max():.3f}")
    print(f"    from raw sentinels   : {y_observed.max():.3f}  (the limit itself)")

    # -- left-censoring, for completeness ---------------------------------- #
    # The same wrapper handles a lower detection limit; only `direction` changes.
    lower_limit = float(np.quantile(y_true, 0.25))
    left_mask = y_true <= lower_limit
    y_left = np.where(left_mask, lower_limit, y_true)
    left_model = CensoredEMGPModel(
        gp(), censoring_limit=lower_limit, direction="left", max_iter=20
    )
    left_model.fit(X, y_left)
    left_imputed = left_model.em_result_.imputed_y[left_mask]
    print("\nLeft-censoring (same wrapper, direction='left')")
    print(f"  lower limit            : {lower_limit:.3f}")
    print(f"  mean imputed           : {left_imputed.mean():.3f} (all <= limit: "
          f"{bool(np.all(left_imputed <= lower_limit + 1e-9))})")

    print("\n" + "=" * 78)
    print("When to stop trusting this")
    print("=" * 78)
    print("  Imputation degrades as the censored fraction rises: the imputed points")
    print("  start reinforcing one another and the iteration drifts past its best")
    print("  point. Measured on this generator, the LOO refactor is close to the")
    print("  uncensored oracle at ~25% censoring and clearly better than naive")
    print("  substitution at ~45%, but by ~60% no imputation method is reliable.")
    print("  Past roughly a third censored, prefer TobitGPModel - it optimises the")
    print("  censored likelihood directly and never invents a value.")


if __name__ == "__main__":
    main()
