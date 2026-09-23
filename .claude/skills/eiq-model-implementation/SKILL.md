---
name: eiq-model-implementation
description: |
  Write and validate the modeling code behind a new Everesteer hackathon-event model, either a templated built-in via MCP train(model=<preset>), or your own training script run on Everesteer compute via MCP train(model="custom", custom_model_fn=...). Use when an idea needs real modeling code (a new model type, a custom fit/predict routine, an ensemble) rather than just a different hyperparameter sweep. Covers the fit/predict contract, leakage-safe validation, and the patterns that move AIMC.
---

# Implementing a Model for an Everesteer Hackathon Event

An Everesteer hackathon event asks you to rank the instruments present at each **exped** against the event dataset's graded target, the column `get_dataset_schema` reports as `primary_target`, read at runtime rather than hardcoded. This skill is about the *code* that produces those rankings: how to express a model so it runs cleanly on Everesteer compute, and how to convince yourself the model is real before you put value behind it.

**The event panel is obfuscated, and that shapes the code you write.** Rows carry no instrument identity, no chain and no cluster: `meta_cols` is `exped` and `data_type`, and `data_type` is a constant split marker. Feature names are opaque labels with no decodable structure. So there is nothing to group by except time, and any pattern that needs instrument or sector metadata - cluster sample-weighting, roll-window checks, liquidity tiers - belongs to the live tournament, not here. `get_features` answers a hackathon key with `403 scope_mismatch` and `get_universe` returns an empty instrument list; `get_dataset_schema` (and `verbose=True` for feature-set membership) is your column source.

You never touch any platform-internal repository. Everything here is built on the public `everestapi` SDK, the Everesteer MCP tools, and the helper code in `example-scripts/` plus a `models/` directory you own.

## Two ways to produce a model

Both go through the same unified `train` tool. The difference is just the `model` argument:

1. **`train(model=<lightgbm|xgboost|ridge|mlp|random_forest>)` (templated).** Pick a built-in model family, pass a config, and the platform fits it for you. No code to write. Go straight to `eiq-experiment-design`. Use this for baselines and for anything a standard learner handles well.

2. **`train(model="custom", custom_model_fn=...)` (your code).** When the modeling idea is not expressible as a templated config, a bespoke ensemble, a residualized target, a custom fit loop, an exotic learner, you write a `build_model(params) -> estimator` factory as Python **source**, pass it as `custom_model_fn`, and pull the fitted artifact back with `get_model_download_url`. The platform runs it on GPU/CPU compute in an isolated, network-denied sandbox (the source is static-safety-scanned first).

This skill focuses on case 2. Reach for it only when a templated `train(model=<preset>)` genuinely can't express what you want. Extra moving parts mean extra ways to leak or break.

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

        X : feature frame. Columns are opaque bin-coded feature names - read the
            real ones from the schema or the parquet, they follow no pattern you
            can match on. Each value is an integer bin; the bin count, value range
            and missing sentinel come from the schema's feature_encoding.
            Rows are (exped, anonymous instrument) observations.
        y : the graded target (schema primary_target), aligned to X.index.
        sample_weight : optional per-row weights.
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

- **DataFrame in, Series out.** Convert to numpy *inside* the method, never at the boundary. The index is your contract with the scorer and you must preserve it.
- **One score per row, index-aligned.** The platform ranks your raw scores cross-sectionally within each exped, so the absolute scale is irrelevant. Only the within-exped ordering of rows is scored. Don't pre-normalize to `[-1, 1]`; don't drop or reorder rows.
- **No NaNs out.** A single NaN poisons that exped's rank. Fill or guard before returning.
- **Lazy-import optional deps** with an actionable message, so a missing package fails loudly rather than at fit time:

```python
def fit(self, X, y, sample_weight=None):
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError(
            "catboost not available in this environment. Pick a preset or a "
            "dependency from the sandbox's preinstalled ML stack."
        ) from e
    ...
```

Keep custom models in a participant-owned `models/` directory (e.g. `models/everest_catboost.py`) so model code stays separate from experiment glue.

**That directory does not travel with your pickle unless you say so.** cloudpickle stores a class from an importable module *by reference*: the artifact records `models.everest_catboost.MyEverestModel` by name, and unpickling it anywhere without your repo on the path fails with `ModuleNotFoundError: No module named 'models'`. Register your own modules by value before you dump the `predict` callable:

```python
import cloudpickle
import models.everest_catboost

cloudpickle.register_pickle_by_value(models.everest_catboost)   # your code, embedded
cloudpickle.dump(build_predict(model, feature_cols, missing), fh)
```

Register only your own modules. Third-party libraries (`catboost`, `lightgbm`, …) stay by reference and come from the sandbox's library set, which is why you pickle against `library_pins`. A class defined inside the same script or notebook that pickles it is `__main__` and already goes by value. The round never catches this: the pickle is store-only there, so an unloadable one costs you only the daily-predictions lane, silently. Check it: load the `.pkl` from a directory outside your repo and call it on a few served rows before you upload.

## Running it on Everesteer compute

The custom path is a **source string**, not a script upload: you pass `custom_model_fn`, Python source defining `build_model(params) -> estimator`, and the platform's harness loads the data, calls your factory, fits the estimator, and stores the artifact. Your code never downloads anything (the sandbox is network-denied; the harness hands it the data).

```python
from everestapi import EverestAPI
client = EverestAPI(api_key="...", tournament="futures")

# 1. Get the data locally to develop/smoke-test your model class against. With a
#    hackathon key `validation` is a target-blanked, server-scored practice board,
#    not a labeled split: build your holdout from an embargoed tail of `train`
#    instead (Step 2 below).
client.download_dataset(universe="futures", split="train", output_path="train.parquet")

# 2. Ship the factory source to compute (GPU for heavy learners).
#    gpu in {CPU, T4, A10G, A100}: the DEFAULT is T4; CPU is cheapest (no GPU line
#    item) and right for tree models. max_hours caps at 4.0 and the platform enforces
#    a hard runtime ceiling (~45 min), so keep jobs tight.
CUSTOM_FN = '''
import numpy as np


class MaskMissing:
    """Turn the dataset's missing value into NaN before the learner sees it.

    The harness fills NaN with 0.0 and passes the missing value (-1) through
    as an ordinary number, so without this the learner treats "missing" as a
    bin below the lowest real one. Masking inside the model means fit and
    predict always see the same thing, wherever the model later runs.
    """

    def __init__(self, learner, missing):
        self.learner, self.missing = learner, missing

    def _mask(self, X):
        X = np.array(X, dtype="float64")  # a copy: never edit the harness's array
        X[X == self.missing] = np.nan
        return X

    def fit(self, X, y, sample_weight=None):
        self.learner.fit(self._mask(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.learner.predict(self._mask(X))


def build_model(params):
    # Everything must be defined or imported INSIDE this source string,
    # the sandbox cannot see your local files.
    from catboost import CatBoostRegressor   # handles NaN natively
    params = dict(params)
    missing = params.pop("missing", -1.0)    # passed in from the schema below
    return MaskMissing(CatBoostRegressor(**params, verbose=0), missing)
'''
schema = client.get_dataset_schema()
missing = (schema.get("feature_encoding") or {}).get("missing", -1.0)

# The fit must stop before your embargo and holdout: without train_filter a job
# fits the whole train split, and every holdout score is in-sample.
# FIRST_EMBARGO_EXPED is the first exped of the embargo in front of your holdout.
job = client.train(model="custom", custom_model_fn=CUSTOM_FN,
                   features="all",                  # the default "small" is a prefix here
                   train_filter={"exped": {"cutoff_lt": FIRST_EMBARGO_EXPED}},
                   params={"iterations": 600, "depth": 6, "missing": missing},
                   gpu="A100", max_hours=2.0)

# 3. Wait, then retrieve the fitted artifact and the record of what it was fit on.
result = client.wait_for_job(job["job_id"])
client.download_model(job["job_id"], output_path="model.pkl")
manifest = result["feature_manifest"]   # ordered features + the harness's preprocessing

# 4. Score it on YOUR holdout, and wrap it before uploading. Do not select on the
#    job's own CV metrics, and never upload model.pkl as downloaded: it has come
#    back as a bare estimator, which the upload gate refuses. starter_hosted.py's
#    make_scorer() and build_predict() do both, and accept either artifact shape.
```

Via MCP the equivalents are `train` (same `custom_model_fn` source-string argument), `get_job_status`, `get_job_log`, and `get_model_download_url`; check budget first with `get_compute_credits`, fetch column names with `get_dataset_schema`, and use `download_dataset` for the parquet. `train` runs on metered GPU/CPU, and the `CPU` tier is cheapest for tree models. Over MCP, `train(..., dry_run=true)` validates the call and resolves its defaults before you launch. Its `estimated_hold_cents` is a flat worst-case reservation for the tier, not a per-config cost.

Two contract details that bite:

- **Seeding goes through `params`** (e.g. `params={"seed": 7}` for lightgbm, `{"random_state": 7}` for sklearn-style estimators). A top-level `seed=` argument is rejected with a 400. The request schema is strict, so typos fail loudly rather than silently no-op.
- **Keep `build_model` self-contained.** Inline everything the factory needs (imports inside the function, class definitions in the same source string). The sandbox has the common ML stack available but no network. It cannot `pip install`, fetch data, or read your local files.

> **Pickle is RCE-equivalent.** Only `pickle.load` artifacts from jobs *you* launched. Never load a model file handed to you by someone else without inspecting it in isolation.

## Validation gate: do this before you stake

The hardest part of the event is not training; it's knowing whether the number you see is real. You will eventually decide whether to **stake real value** on a model, and that decision is only as good as your out-of-sample estimate. Treat validation as a gate, not a formality.

**Step 1. Smoke run on a small exped subset.** Fit on a slice, predict, and assert the contract holds:

```python
sub = train[train["exped"].isin(train["exped"].unique()[:200])]
m = MyEverestModel().fit(sub[feats], sub[TARGET])  # TARGET = schema['primary_target']
p = m.predict(sub[feats])
assert isinstance(p, pd.Series) and len(p) == len(sub)
assert p.index.equals(sub.index) and not p.isna().any()
```

**Step 2. Sanity-bound the CORR.** Compute per-exped rank correlation against the target on a *held-out* split, never the rows you trained on. With a hackathon key that is not the downloadable `validation` split. Its target columns are blanked (it's a server-scored practice board), so carve an embargoed tail off the labeled `train` split instead, the same way `notebooks/02_train_and_submit.ipynb` does: hold out the last N expeds, and discard enough more before the boundary that the
target's forward window cannot leak across it. **The horizon is a dataset fact, and this
dataset does not publish it**: not in the target's name, not in any schema field. So do not
reverse-engineer one from a column name. Embargo generously instead - erring wide costs
almost nothing, erring short is the leak:

```python
EMBARGO = 20  # a deliberately wide round number, NOT a horizon read off the data
# train's exped tokens are zero-padded, so their string order is their time order.
ordered = sorted(train["exped"].unique())
holdout_expeds = ordered[-100:]
fit_rows = train[train["exped"] < ordered[-100 - EMBARGO]]   # stops EMBARGO expeds short
holdout = train[train["exped"].isin(holdout_expeds)].dropna(subset=[TARGET])

m = MyEverestModel().fit(fit_rows[feats], fit_rows[TARGET])
holdout = holdout.assign(prediction=m.predict(holdout[feats]))

# CORR is computed within each exped, then averaged. Don't use EverestAPI.evaluate for
# this: it pools every row into one correlation, which is not the per-exped CORR the
# board scores.
corr = holdout.groupby("exped")[["prediction", TARGET]].apply(
    lambda g: g["prediction"].rank().corr(g[TARGET].rank())
).dropna()
print(f"CORR {corr.mean():+.4f} | std {corr.std():.4f} | {(corr > 0).mean():.0%} of expeds positive")
```

A healthy futures model usually lands at a **small positive** CORR, on the order of a few hundredths. But one holdout is a noisy read, and the two tails mean very different things:

- **CORR near zero or negative:** common, and not a verdict on its own. Per-exped CORR swings widely, and some periods are simply harder to predict than others. It happens often on the `validation` practice board too (see the negative-score tip in `AGENTS.md`). Rule out the mechanical causes first, a broken index join or a feature filter that dropped the columns you meant, and then look at whether the edge holds across both halves of the holdout and how CORR compares with NCORR. Never flip the sign of your predictions to fix it: that fits the one period you can see.
- **CORR suspiciously high** (e.g. an order of magnitude above what the published benchmark and the leaderboard achieve): assume **leakage** until proven otherwise. The usual culprits are evaluating on training rows, leaking the target through a derived column, or an index that lets future expeds bleed in.

Run `run_validation_diagnostics` (MCP) for the platform's own read on feature exposure and per-exped behaviour before trusting a number.

**Step 3. Guard the fit/predict loop against subtle leakage.** Early stopping is the classic trap: if a validation fold steers the stopping point and that same fold feeds your reported metric, you've contaminated the estimate. Tune stopping inside a nested split, then report on data that played no role in fitting. Any per-feature standardization, target encoding, or neutralization must be fit on train only and *applied* to validation. Never re-fit there.

## Patterns that move AIMC (and why)

The round score is a weighted blend of CORR, AIMC and NCORR, bounded per round. Call `explain_scoring` for the live weights; don't hardcode which term dominates. **AIMC** and **NCORR** both mean a merely-accurate model that re-expresses what the reference already says pays little on those components. On a money event the round score is then mapped to a payout through a **bounded** function, `A * tanh(payout_factor * score / A)`; `get_event_staking` reports the `payout_factor` and `stake_return_amplitude` each round froze, and `everestapi.scoring.payout` takes both.

**What AIMC is measured against is a per-product setting, and on a hackathon event it works in your favour.** `explain_scoring`'s `metrics.aimc` reports it as your contribution over **the event's own reference benchmark predictions**, not the live crowd consensus the tournament uses. The benchmark is downloadable over `train`, so unlike the tournament case you *can* build a close offline proxy: residualize your predictions against the benchmark per exped, then correlate the residual with the target. Treat it as a proxy still - the real number comes back after you submit - but not as an unobservable.

```python
from scipy.stats import spearmanr   # plus pandas as pd, numpy as np
# EXPED and PRIMARY_TARGET come from get_dataset_schema() - read them, don't hardcode.

# The reference to measure against: the mean of the benchmark models over `train`.
# The validation/live benchmark splits are withheld while an event runs and 404 by
# design (not an outage, and retrying will not clear it); `train` always works.
bench = pd.read_parquet(client.download_benchmark("futures", "train")).reset_index()
bench_cols = [c for c in bench.columns if c not in ("id", EXPED)]
bench["consensus"] = bench[bench_cols].mean(axis=1)
# The id is the parquet INDEX on both frames, so reset it on the holdout too before merging.
# Merge on id AND exped: a benchmark built from an older train file can share ids with this
# one on different expeds, and an id-only join scores it against the wrong rows.
holdout = (holdout.reset_index()
           .merge(bench[["id", EXPED, "consensus"]], on=["id", EXPED], how="left")
           .set_index("id"))

def neutralize(preds, neutralizers, proportion=1.0):
    """Remove `proportion` of whatever `neutralizers` explains linearly; return the residual."""
    y = preds.to_numpy(dtype="float64").reshape(-1, 1)
    X = np.hstack([neutralizers.to_numpy(dtype="float64"), np.ones((len(neutralizers), 1))])
    projection = X @ (np.linalg.pinv(X, rcond=1e-6) @ y)
    return pd.Series((y - proportion * projection).ravel(), index=preds.index)

def contribution(df, pred_col, target_col=PRIMARY_TARGET, bench_col="consensus"):
    """AIMC proxy, per exped: residualize against the benchmark, correlate with the target."""
    out = {}
    for e, g in df.groupby(EXPED):
        g = g[g[bench_col].notna()]   # an all-NaN exped hands pinv a NaN matrix and raises
        if len(g) < 3:
            continue
        rho, _ = spearmanr(neutralize(g[pred_col], g[[bench_col]]), g[target_col])
        if np.isfinite(rho):
            out[e] = rho
    return pd.Series(out)
```

Check the merge before you trust the number: `holdout["consensus"].notna().mean()` is the share of holdout rows the benchmark covers on the same id and exped, and it should be close to 1. Both come from `train`, so a low share means one of two things: the holdout carve is wrong, or the benchmark was built from a different train file than the one served now (ids that exist in both but sit on different expeds). Either way, don't read `contribution()` until the share is back near 1, because the rows it does score are a biased few.

Neutralization is **cross-sectional**, so `neutralize` is applied per exped in both uses: against the benchmark for the proxy above, and against a feature block for the exposure fix below.

- **Residualize the target against the benchmark.** Train on the residual of the graded target after projecting out the benchmark series, so the model can only learn what the benchmark misses. Raises AIMC directly, since the benchmark is what AIMC is measured against here.
- **Neutralize predictions against the benchmark / heavy features.** OLS-project your raw scores onto the benchmark (or a few dominant feature exposures) and subtract the projection. Lowers correlation to the reference, raising AIMC, usually at a modest CORR cost; tune the neutralization proportion. Note NCORR's own neutralization is a spectrally-anchored ridge against a frozen core feature set whose membership is not published, so your own OLS residualization will not reproduce that number.
- **Blend multiple targets.** The auxiliary targets (every entry in the schema's `targets` other than the graded one) carry related-but-distinct signal; a weighted blend can be steadier than chasing the graded target alone. Which auxiliaries are diverse and which are near-duplicates is a property of the dataset you are on, so **compute the target correlation matrix yourself**: do not carry numbers over from another event. Any pair near correlation −1.0 are inverses of one signal; never include both raw.
- **Bag / ensemble.** Average several seeds or row-subsamples to cut variance. Steadier rankings translate to steadier AIMC across rounds, which matters more than a single hot exped.

Each of these earns its keep only if it raises differentiated signal. Accuracy that everyone already has is nearly free on the payout formula.

## Event-specific concerns

- **Never impute over the missing sentinel.** A feature value equal to the schema's
  `feature_encoding.missing` (commonly `-1.0`) means the source was not onboarded for that
  row. Map it to `NaN` and let a NaN-aware learner handle it, or treat it as its own
  category. Do **not** fill it with a value inside the real bin range: that tells the model
  "missing" is an ordinal position on the same scale, which is false. And do not fill with
  `0`, which is a real bin. (A NaN *target* is different: that row was uncomputable and is
  never imputed, so drop it.)
- **Never assume a fixed column set.** The served split can carry a different set of feature
  columns than the one you trained on. Select features **by name** inside `predict` and
  reindex defensively, so an absent column becomes an honest `NaN` rather than silently
  shifting a positional array underneath the model.
- **Robustness over time, not over clusters.** There is no cluster axis on this panel, so
  the honest fragility check is temporal: split your holdout in half and confirm the edge
  survives in both. An edge that lives in one stretch of expeds is a regime artifact.
  Per-exped CORR spread and the worst run of negative expeds are the numbers to look at,
  alongside `run_validation_diagnostics`.
- **Sample weighting has no grouping column to key on.** Inverse-frequency weighting by
  cluster or sector is not available here; if you weight, weight on something the panel
  actually publishes, such as exped recency.

## Where this hands off

Once the model code passes the validation gate:

1. Hand off to **`eiq-experiment-design`** to run multiple rounds, compare configs, and scale the winners until they plateau.
2. Then **`eiq-event-submission`** to deploy the chosen model and submit predictions into the open round.

Don't skip ahead to submission from a single promising backtest. The experiment loop is what separates a real edge from a lucky split.
