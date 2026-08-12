#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tobit Gaussian Process for a heavily censored potency panel
===========================================================

Same scenario as ``example_schmee_hahn_censored_gp.py`` - a panel with an assay
ceiling - but at a censoring rate where imputation stops being trustworthy.

``TobitGPModel`` never invents a value. It writes the inequality into the
likelihood: an observed row contributes a Gaussian density, a censored row
contributes ``log P(Y >= limit)``. Hyperparameters are then chosen to maximise
the *observed-data* likelihood, so a run of sequences pinned at the ceiling no
longer looks like a suspiciously flat, noiseless region of the landscape.

The script covers:

  1. a sanity check - with nothing censored it reduces to an ordinary GP
  2. a head-to-head against naive substitution and Schmee-Hahn at ~45% censoring
  3. calibration, which is where naive substitution fails hardest
  4. ``predict_censored_probability`` for triaging a proposed batch
  5. ``censored_log_likelihood``, the only fair yardstick across methods
  6. a note on the optimiser, which needs more care than an exact GP

Run::

    python example_tobit_gp.py
"""
from __future__ import annotations

import warnings

import gpytorch
import numpy as np

from mobius.surrogate_models import (
    CensoredEMGPModel,
    CensoringSpec,
    GPModel,
    TobitGPModel,
    censored_normal_logpdf,
)

RANDOM_STATE = 0
N_TRAIN = 100
N_SEEDS = 3
ASSAY_NOISE_SD = 0.15
HEAVY_CENSORING = 0.45


def make_panel(fraction_censored=HEAVY_CENSORING, n=N_TRAIN, seed=RANDOM_STATE):
    """Panel with a ceiling, standardised on the uncensored rows.

    Standardising on all rows would use the sentinel pile-up at the limit,
    which is the artefact being modelled away. ``TobitGPModel`` does this
    internally too, but the baselines it is compared against do not.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    latent = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
    y_true = latent + ASSAY_NOISE_SD * rng.normal(size=n)

    limit = float(np.quantile(y_true, 1.0 - fraction_censored))
    censored = y_true >= limit
    y_observed = np.where(censored, limit, y_true)

    location = y_observed[~censored].mean()
    scale = y_observed[~censored].std() or 1.0
    rescale = lambda v: (v - location) / scale
    return (X, rescale(y_observed), rescale(y_true), rescale(latent),
            float(rescale(limit)), censored, (location, scale))


def make_holdout(location, scale, n=400, seed=99):
    """Held-out compounds re-measured on an assay with no ceiling."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    latent = 1.5 * np.sin(X[:, 0]) + 0.4 * X[:, 0]
    y = latent + ASSAY_NOISE_SD * rng.normal(size=n)
    return X, (y - location) / scale, (latent - location) / scale


def kernel():
    return gpytorch.kernels.MaternKernel(nu=2.5)


def gp():
    return GPModel(kernel=kernel(), show_progression=False)


def tobit(limit, **kwargs):
    kwargs.setdefault("n_iter", 1500)
    kwargs.setdefault("patience", 100)
    return TobitGPModel(
        kernel=kernel(),
        censoring_limit=limit,
        direction="right",
        show_progression=False,
        random_state=RANDOM_STATE,
        **kwargs,
    )


def score(mu, sd, latent, y_holdout, region):
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
    return (
        np.sqrt(np.mean((mu - latent) ** 2)),
        np.sqrt(np.mean((mu - latent)[region] ** 2)),
        np.mean((y_holdout[region] >= lo[region]) & (y_holdout[region] <= hi[region])),
        sd[region].mean(),
    )


# --------------------------------------------------------------------------- #
def part_one_sanity_check():
    """With no censored rows the Tobit ELBO is a plain variational GP objective."""
    X, _, y_true, latent, _, _, (location, scale) = make_panel()
    X_test = np.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    latent_test = (1.5 * np.sin(X_test[:, 0]) + 0.4 * X_test[:, 0] - location) / scale

    exact = gp()
    exact.fit(X, y_true)
    mu_exact, _ = exact.predict(X_test)

    model = TobitGPModel(kernel=kernel(), show_progression=False,
                         random_state=RANDOM_STATE, n_iter=800)
    model.fit(X, y_true)  # no censoring_limit -> nothing is censored
    mu_tobit, _ = model.predict(X_test)

    print("1. Sanity check: no censoring")
    print(f"   RMSE(latent)  exact GP {np.sqrt(np.mean((mu_exact - latent_test) ** 2)):.4f}"
          f"   Tobit {np.sqrt(np.mean((mu_tobit - latent_test) ** 2)):.4f}")
    print(f"   correlation between the two predictive means: "
          f"{np.corrcoef(mu_exact, mu_tobit)[0, 1]:.4f}")
    print("   -> the censored likelihood degrades gracefully to the ordinary one.\n")


def part_two_head_to_head():
    """Naive vs Schmee-Hahn vs Tobit at a censoring rate that breaks imputation."""
    aggregate = {}
    for seed in range(N_SEEDS):
        X, y_observed, _, _, limit, censored, (location, scale) = make_panel(seed=seed)
        X_holdout, y_holdout, latent_holdout = make_holdout(location, scale, seed=1000 + seed)
        region = latent_holdout >= limit

        model = gp()
        model.fit(X, y_observed)
        aggregate.setdefault("naive substitution", []).append(
            score(*model.predict(X_holdout), latent_holdout, y_holdout, region)
        )

        schmee_hahn = CensoredEMGPModel(gp(), censoring_limit=limit, max_iter=30)
        schmee_hahn.fit(X, y_observed)
        aggregate.setdefault("Schmee-Hahn (LOO)", []).append(
            score(*schmee_hahn.predict(X_holdout), latent_holdout, y_holdout, region)
        )

        model = tobit(limit)
        model.fit(X, y_observed)
        aggregate.setdefault("Tobit GP", []).append(
            score(*model.predict(X_holdout), latent_holdout, y_holdout, region)
        )

    print(f"2. Head-to-head at ~{HEAVY_CENSORING:.0%} censoring, "
          f"mean over {N_SEEDS} panels")
    print(f"   {'method':22s} {'RMSE all':>9s} {'RMSE >lim':>10s} {'95% cov':>9s} {'mean sd':>9s}")
    for name, values in aggregate.items():
        arr = np.asarray(values)
        print(f"   {name:22s} {arr[:, 0].mean():9.3f} {arr[:, 1].mean():10.3f} "
              f"{arr[:, 2].mean():9.2f} {arr[:, 3].mean():9.3f}")
    print("   -> read the coverage column, not just RMSE. Naive substitution is")
    print("      not merely wrong above the limit, it is confidently wrong; the")
    print("      Tobit posterior stays wide because a censored point genuinely")
    print("      does not pin down a value.\n")


def part_three_batch_triage():
    """Which proposed designs would the current assay fail to resolve?"""
    X, y_observed, _, _, limit, censored, _ = make_panel()
    model = tobit(limit)
    model.fit(X, y_observed)

    probability = model.predict_censored_probability(X)
    print("3. Batch triage with predict_censored_probability")
    print(f"   mean P(censored) on rows that came back censored : {probability[censored].mean():.2f}")
    print(f"   mean P(censored) on rows that did not            : {probability[~censored].mean():.2f}")

    X_new = np.linspace(-3.0, 3.0, 9).reshape(-1, 1)
    mu, sd = model.predict(X_new)
    p_new = model.predict_censored_probability(X_new)
    print("\n   A hypothetical proposed batch:")
    print(f"   {'x':>6s} {'pred':>8s} {'sd':>7s} {'P(censored)':>12s}  note")
    for x, m, s, p in zip(X_new[:, 0], mu, sd, p_new):
        note = "assay cannot resolve" if p > 0.5 else ""
        print(f"   {x:6.2f} {m:8.3f} {s:7.3f} {p:12.2f}  {note}")
    print("   -> a design predicted potent AND likely to saturate is a candidate")
    print("      for a different readout, not another run of the same assay.\n")


def part_four_fair_comparison():
    """R^2 against imputed values scores invented numbers; use the censored likelihood."""
    X, y_observed, _, _, limit, _, _ = make_panel()
    spec = CensoringSpec.from_arrays(y_observed, censoring_limit=limit)

    naive = gp()
    naive.fit(X, y_observed)
    mu, sd = naive.predict(X)
    naive_ll = float(np.sum(censored_normal_logpdf(
        y_observed, mu, sd, spec.lower, spec.upper, observed=~spec.censored
    )))

    schmee_hahn = CensoredEMGPModel(gp(), censoring_limit=limit, max_iter=30)
    schmee_hahn.fit(X, y_observed)
    mu, sd = schmee_hahn.predict(X)
    sh_ll = float(np.sum(censored_normal_logpdf(
        y_observed, mu, sd, spec.lower, spec.upper, observed=~spec.censored
    )))

    model = tobit(limit)
    model.fit(X, y_observed)

    print("4. Observed-data log-likelihood (higher is better)")
    print(f"   naive substitution : {naive_ll:9.2f}")
    print(f"   Schmee-Hahn (LOO)  : {sh_ll:9.2f}")
    print(f"   Tobit GP           : {model.censored_log_likelihood():9.2f}")
    print(f"   fitted noise variance: {model.noise:.4f} "
          f"(true {ASSAY_NOISE_SD ** 2:.4f}, on the standardised scale it differs)")
    print("   -> observed rows contribute a density, censored rows the probability")
    print("      of their interval. This is comparable across methods; R^2 against")
    print("      imputed targets is not.\n")


def part_five_optimiser_notes():
    """The one real cost of the Tobit model: it has an optimiser to tune."""
    X, y_observed, _, _, limit, _, (location, scale) = make_panel()
    X_holdout, y_holdout, latent_holdout = make_holdout(location, scale)
    region = latent_holdout >= limit

    print("5. Optimiser sensitivity (Adam on the ELBO, not closed-form L-BFGS)")
    print(f"   {'lr':>6s} {'n_iter':>7s} {'patience':>9s} {'steps':>6s} "
          f"{'RMSE >lim':>10s} {'95% cov':>9s}")
    for lr, n_iter, patience in [(0.05, 400, 30), (0.05, 1500, 100), (0.02, 3000, 300)]:
        model = TobitGPModel(
            kernel=kernel(), censoring_limit=limit, show_progression=False,
            random_state=RANDOM_STATE, learning_rate=lr, n_iter=n_iter, patience=patience,
        )
        model.fit(X, y_observed)
        _, rmse_region, coverage, _ = score(
            *model.predict(X_holdout), latent_holdout, y_holdout, region
        )
        print(f"   {lr:6.2f} {n_iter:7d} {patience:9d} {len(model.loss_history_):6d} "
              f"{rmse_region:10.3f} {coverage:9.2f}")
    print("   -> running the ELBO to convergence does not monotonically improve")
    print("      RMSE: the variational fit keeps buying likelihood in the censored")
    print("      region by widening the posterior there. Early stopping via")
    print("      `patience` is doing real work, and the defaults are a compromise.")
    print("      If you tune anything, tune this on a held-out uncensored subset.\n")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*scipy_minimize.*")
    warnings.filterwarnings("ignore", message=".*input matches the stored training data.*")
    warnings.filterwarnings("ignore", message=".*Very small noise values.*")
    warnings.filterwarnings("ignore", message=".*did not converge.*")

    X, _, _, _, limit, censored, _ = make_panel()
    print("=" * 78)
    print("Tobit GP on a heavily censored panel")
    print("=" * 78)
    print(f"  training points       : {len(X)}")
    print(f"  censored at the limit : {censored.sum()} ({censored.mean():.0%})")
    print(f"  detection limit       : {limit:.3f} (standardised units)\n")

    part_one_sanity_check()
    part_two_head_to_head()
    part_three_batch_triage()
    part_four_fair_comparison()
    part_five_optimiser_notes()

    print("=" * 78)
    print("Choosing between the two models")
    print("=" * 78)
    print("  Up to roughly a third censored, CensoredEMGPModel is cheaper, has no")
    print("  optimiser to tune, and is close enough. Past that, the imputed points")
    print("  begin reinforcing one another and TobitGPModel is the safer choice -")
    print("  particularly for uncertainty, which is what an acquisition function")
    print("  actually consumes. Both are worth running; if they disagree sharply")
    print("  about the censored region, treat that disagreement as the signal that")
    print("  the assay window, not the model, is the thing to change.")


if __name__ == "__main__":
    main()
