---
name: eiq-model-implementation
description: |
  Write and validate the modeling code behind a new Everesteer hackathon-event (futures dataset) model — either a templated built-in via MCP train(model=<preset>), or your own training script run on Everesteer compute via MCP train(model="custom", custom_model_fn=...). Use when an idea needs real modeling code (a new model type, a custom fit/predict routine, an ensemble) rather than just a different hyperparameter sweep. Covers the fit/predict contract, leakage-safe validation, and the futures-specific patterns that move AIMC.
---

# Implementing a Model for an Everesteer Hackathon Event

An Everesteer hackathon event asks you to rank global futures **chains** (grouped into **clusters**) at each **exped** against the `target_everest_20` target. This skill is about the *code* that produces those rankings: how to express a model so it runs cleanly on Everesteer compute, and how to convince yourself the model is real before you put value behind it.

You never touch any platform-internal repository. Everything here is built on the public `everestapi` SDK, the Everesteer MCP tools, and the helper code in `example-scripts/` plus a `models/` directory you own.

## Two ways to produce a model

Both go through the same unified `train` tool — the difference is just the `model` argument:

1. **`train(model=<lightgbm|xgboost|ridge|mlp|random_forest>)` (templated).** Pick a built-in model family, pass a config, and the platform fits it for you. No code to write — go straight to `eiq-experiment-design`. Use this for baselines and for anything a standard learner handles well.

2. **`train(model="custom", custom_model_fn=...)` (your code).** When the modeling idea is not expressible as a templated config — a bespoke ensemble, a residualized target, a custom fit loop, an exotic learner — you write a `build_model(params) -> estimator` factory as Python **source**, pass it as `custom_model_fn`, and pull the fitted artifact back with `get_model_download_url`. The platform runs it on GPU/CPU compute in an isolated, network-denied sandbox (the source is static-safety-scanned first).

This skill focuses on case 2. Reach for it only when a templated `train(model=<preset>)` genuinely can't express what you want — extra moving parts mean extra ways to leak or break.

## The model contract

A custom model is any object with two methods. Keep the surface this small; the harness does the rest.

```python
import pandas as pd
import numpy as np


class MyEverestModel:
    """An Everesteer event model. Implements fit() and predict() only."""

    def __init__(self, **hyperparams):
        # Stash config; build nothing heavy here.
        self.hp = hyperparams
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "MyEverestModel":
        """Train.

        X : feature frame. Columns are encoded, quintile-binned names of the
            form feature_<theme>_<n>, each value an integer in {0,1,2,3,4}.
            Rows are (exped, chain) observations.
        y : target_everest_20, rank-normalized forward return, aligned to X.index.
        sample_weight : optional per-row weights (see cluster weighting below).
        """
        Xn = X.to_numpy(dtype=np.float32)
        yn = y.to_numpy(dtype=np.float32)
        # ... fit your learner on Xn/yn (+ weights) ...
        self._fitted = ...  # the trained object
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Score. One value per row, index identical to X.index."""
        if self._fitted is None:
            raise RuntimeError("predict() called before fit().")
        raw = self._fitted.predict(X.to_numpy(dtype=np.float32))
        return pd.Series(raw, index=X.index)
```

Rules that matter:

- **DataFrame in, Series out.** Convert to numpy *inside* the method, never at the boundary — the index is your contract with the scorer and you must preserve it.
- **One score per row, index-aligned.** The platform ranks your raw scores cross-sectionally within each exped, so the absolute scale is irrelevant — only the within-exped ordering of chains is scored. Don't pre-normalize to `[-1, 1]`; don't drop or reorder rows.
- **No NaNs out.** A single NaN poisons that exped's rank. Fill or guard before returning.
- **Lazy-import optional deps** with an actionable message, so a missing package fails loudly rather than at fit time:

```python
def fit(self, X, y, sample_weight=None):
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError(
            "catboost not available in this environment — pick a preset or a "
            "dependency from the sandbox's preinstalled ML stack."
        ) from e
    ...
```

Keep custom models in a participant-owned `models/` directory (e.g. `models/everest_catboost.py`) so model code stays separate from experiment glue.

## Running it on Everesteer compute

The custom path is a **source string**, not a script upload: you pass `custom_model_fn` — Python source defining `build_model(params) -> estimator` — and the platform's harness loads the data, calls your factory, fits the estimator, and stores the artifact. Your code never downloads anything (the sandbox is network-denied; the harness hands it the data).

```python
from everestapi import EverestAPI
client = EverestAPI(api_key="...", tournament="futures")

# 1. Get the data locally to develop/smoke-test your model class against. With a
#    hackathon key `validation` is a target-blanked, server-scored practice board,
#    not a labeled split — build your holdout from an embargoed tail of `train`
#    instead (Step 2 below).
client.download_dataset(universe="futures", split="train", output_path="train.parquet")

# 2. Ship the factory source to compute (GPU for heavy learners).
#    gpu in {CPU, T4, A10G, A100} — the DEFAULT is T4; CPU is cheapest (no GPU line
#    item) and right for tree models. max_hours caps at 4.0 and the platform enforces
#    a hard runtime ceiling (~45 min), so keep jobs tight.
CUSTOM_FN = '''
def build_model(params):
    # Everything must be defined or imported INSIDE this source string —
    # the sandbox cannot see your local files. Paste your model class here.
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**params, verbose=0)
'''
# Preview cost first over MCP: the `train` TOOL takes dry_run=true, which resolves
# defaults and returns estimated_hold_cents without reserving credits or launching
# anything. The Python client's train() has no dry_run parameter, so from here you
# launch directly:
job = client.train(model="custom", custom_model_fn=CUSTOM_FN,
                   params={"iterations": 600, "depth": 6},
                   gpu="A100", max_hours=2.0)

# 3. Wait, then retrieve the fitted artifact.
result = client.wait_for_job(job["job_id"])
client.download_model(job["job_id"], output_path="model.pkl")
```

Via MCP the equivalents are `train` (same `custom_model_fn` source-string argument), `get_job_status`, `get_job_log`, and `get_model_download_url`; check budget first with `get_compute_credits`, fetch column names with `get_dataset_schema`, and use `download_dataset` for the parquet. `train` runs on metered GPU/CPU — the `CPU` tier is cheapest for anything templated; use `train(..., dry_run=true)` to preview the cost before you commit.

Two contract details that bite:

- **Seeding goes through `params`** (e.g. `params={"seed": 7}` for lightgbm, `{"random_state": 7}` for sklearn-style estimators). A top-level `seed=` argument is rejected with a 400 — the request schema is strict, so typos fail loudly rather than silently no-op.
- **Keep `build_model` self-contained.** Inline everything the factory needs (imports inside the function, class definitions in the same source string). The sandbox has the common ML stack available but no network — it cannot `pip install`, fetch data, or read your local files.

> **Pickle is RCE-equivalent.** Only `pickle.load` artifacts from jobs *you* launched. Never load a model file handed to you by someone else without inspecting it in isolation.

## Validation gate — do this before you stake

The hardest part of the event is not training; it's knowing whether the number you see is real. You will eventually decide whether to **stake real value** on a model, and that decision is only as good as your out-of-sample estimate. Treat validation as a gate, not a formality.

**Step 1 — smoke run on a small exped subset.** Fit on a slice, predict, and assert the contract holds:

```python
sub = train[train["exped"].isin(train["exped"].unique()[:200])]
m = MyEverestModel().fit(sub[feats], sub["target_everest_20"])
p = m.predict(sub[feats])
assert isinstance(p, pd.Series) and len(p) == len(sub)
assert p.index.equals(sub.index) and not p.isna().any()
```

**Step 2 — sanity-bound the CORR.** Compute per-exped rank correlation against the target on a *held-out* split, never the rows you trained on. With a hackathon key that is not the downloadable `validation` split — its target columns are blanked (it's a server-scored practice board) — so carve an embargoed tail off the labeled `train` split instead, the same way `notebooks/02_train_and_submit.ipynb` does: hold out the last N expeds, and discard 20 more before the boundary so `target_everest_20`'s 20-day forward window can't leak across it:

```python
EMBARGO = 20  # target_everest_20 is a 20-day forward return; embargo the boundary
tail = train["exped"].unique()[-100:]
holdout = train[train["exped"].isin(tail)]

metrics = EverestAPI.evaluate(preds, holdout, target="target_everest_20")
```

A healthy futures model lands at a **small positive** CORR — on the order of a few hundredths. Both tails are red flags:

- **CORR near zero or negative:** the model isn't learning, or features/target are misaligned. Check the index join and the feature filter.
- **CORR suspiciously high** (e.g. an order of magnitude above what `ai_model` and the leaderboard achieve): assume **leakage** until proven otherwise. The usual culprits are evaluating on training rows, leaking the target through a derived column, or an index that lets future expeds bleed in.

Run `run_validation_diagnostics` (MCP) for the platform's own read on feature exposure and per-cluster behavior before trusting a number.

**Step 3 — guard the fit/predict loop against subtle leakage.** Early stopping is the classic trap: if a validation fold steers the stopping point and that same fold feeds your reported metric, you've contaminated the estimate. Tune stopping inside a nested split, then report on data that played no role in fitting. Any per-feature standardization, target encoding, or neutralization must be fit on train only and *applied* to validation — never re-fit there.

## Patterns that move AIMC (and why)

Payout is a weighted blend of CORR, AIMC, and NCORR — call `explain_scoring` for the live weights and cap; don't hardcode which term dominates. **AIMC** (AI Model Contribution — the contribution beyond the live stake-weighted ai-model consensus) and **NCORR** (Neutralized Correlation) both mean a merely-accurate model that echoes consensus pays little on those components. That score is then scaled by a per-round **payout factor**, frozen at the round's stake lock: 1 below a fixed total-stake threshold, shrinking above it, so it can differ round to round. AIMC is only measurable once a round resolves, so offline you can't observe it directly — judge candidate changes qualitatively, by whether they measurably lower correlation with the benchmark / `ai_model` without giving up real CORR, not against a fabricated offline number. Uniqueness is the lever; the patterns below all chase AIMC.

- **Residualize the target against `ai_model`.** Train on the residual of `target_everest_20` after projecting out the benchmark, so the model can only learn what the benchmark misses. Raises AIMC by lowering correlation with the static benchmark and the live ai-model.
- **Neutralize predictions against the benchmark / heavy features.** OLS-project your raw scores onto the benchmark (or a few dominant feature exposures) and subtract the projection. Lowers correlation to consensus, raising AIMC, usually at a modest CORR cost — tune the neutralization proportion.
- **Blend multiple targets.** The auxiliary peak targets (k2, lhotse, manaslu, … at 20d/60d) carry related-but-distinct signal; a weighted blend can be steadier than chasing `everest_20` alone. (Check the target correlation matrix first — any pair near correlation −1.0 are inverses; never include both raw.)
- **Bag / ensemble.** Average several seeds or row-subsamples to cut variance. Steadier rankings translate to steadier AIMC across rounds, which matters more than a single hot exped.

Each of these earns its keep only if it raises differentiated signal — accuracy that everyone already has is nearly free on the payout formula.

## Futures-specific concerns

- **Cluster-aware sample weighting.** Clusters differ wildly in size and in how dispersed their returns are. Equal per-row weighting lets the largest cluster dominate the fit. Pass `sample_weight` to balance influence — e.g. inverse-frequency by cluster — so the model generalizes across the universe rather than overfitting one corner.
- **Missing chains / contracts.** The live universe shifts as chains onboard, expire, or fall out of coverage; a chain present in training may be absent live (and vice versa). Never assume a fixed instrument set. Reindex defensively and impute missing features within {0..4} rather than dropping rows.
- **Robustness across clusters.** A model with a great blended CORR but negative CORR in two clusters is fragile. Check the per-cluster breakdown (from your own out-of-sample predictions, plus `run_validation_diagnostics`) and prefer broadly-positive models over ones that win on one cluster.

## Where this hands off

Once the model code passes the validation gate:

1. Hand off to **`eiq-experiment-design`** to run multiple rounds, compare configs, and scale the winners until they plateau.
2. Then **`eiq-event-submission`** to deploy the chosen model and submit predictions into the open round.

Don't skip ahead to submission from a single promising backtest — the experiment loop is what separates a real edge from a lucky split.
