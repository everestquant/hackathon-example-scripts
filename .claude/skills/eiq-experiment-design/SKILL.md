---
name: eiq-experiment-design
description: >-
  Plan, run, and iterate Everesteer hackathon-event experiments using the scout→scale
  research loop. Clarify the idea, align a baseline against the event's published
  benchmark, write configs, train via the Everesteer MCP server (the unified train tool,
  templated or custom), select experiments on an offline estimate of the round score
  (the live weights applied to CORR and the AIMC proxy), iterate in rounds, stop at a
  plateau, and scale the winner. Use when asked to design an event experiment, decide what to try next, or
  turn a model idea into a structured, multi-round research plan.
---

# Everesteer Experiment Design (hackathon event)

A repeatable research loop for an Everesteer hackathon event's sealed rounds. You hold the
`everestapi` SDK, the Everesteer MCP server, your downloaded datasets, and the
example-scripts helpers. That is all you need. Everything below is framed around those
tools and a `configs/` + `experiments/` layout that **you own** in your own repo.

The job is never "get one good run." It is: turn an idea into a sequence of cheap,
interpretable rounds, read the evidence, and decide the next round, until the signal
stops improving.

## Ground truth you must not get wrong

**Your key is hackathon-scoped, and the event dataset is obfuscated on purpose.** Three
tournament reads that research write-ups reach for do not work here, and two of them fail
*quietly*, which is worse:

| Call | On a hackathon key |
|---|---|
| `get_features` | `403 scope_mismatch`. Use `get_dataset_schema(verbose=True)` |
| `get_universe` | returns `instruments: []`, `count: 0` with a note. There is no live tournament round to describe |
| `get_benchmarks` | returns `models: []` with a note. Use `download_benchmark` |

- **Rows** are anonymous instruments at an **exped** (plural *expeds*; one exped ≈ one
  trading day). There is **no instrument identity, no
  chain, and no cluster column** - `meta_cols` is `exped` and `data_type`, and `data_type`
  is a constant split marker, not a dimension you can group by. Any advice framed around
  clusters, contract rolls or liquidity tiers is tournament advice and does not apply.
  **Time is the only axis you can slice on.**
- **CV is exped-purged and embargoed.** The label is a forward return, so a fold boundary
  leaks unless you drop the training rows whose label window overlaps the test fold, plus a
  buffer after it. Never plain k-fold. The hosted `train` tool does this inside its own CV,
  but it fits the **whole** train split, so a holdout you carve from its artifacts is
  in-sample. Build your own holdout, from your own split of `train`, with your own embargo.
- **Primary target**: the column `get_dataset_schema` reports as `primary_target`. Never
  hardcode it, and do not assume a horizon from the name: this dataset does not encode one
  there, and no schema field publishes it. Embargo generously instead.
- **Features** are encoded into cross-sectional bins. The bin count, value range and
  missing sentinel come from the schema's `feature_encoding`. Read it rather than
  assuming. They are already binned: do not re-standardize them, and never treat the
  missing sentinel as an ordinal below the lowest real bin. **Feature names are opaque
  labels** with no decodable structure, so read real ones out of the schema or the parquet
  rather than matching a pattern.
- **Benchmark** = whatever `download_benchmark("futures", "train")` serves. Read the
  column names off the frame; do not hardcode one (the event panel publishes a single
  model). The `validation` and `live` benchmark splits are **withheld while an event is
  running** and 404 by design, because serving the event benchmark's scored predictions
  would let its score be cloned. `train` is the split that always works, which is the
  other reason your holdout is carved out of `train`.
- **Metrics** (call `explain_scoring` for the live weights and definitions; it is the
  authority, this file is not):
  - **CORR**: per-exped rank correlation of your predictions vs the graded target. A
    scored term, and the one you can measure most precisely offline. One input to your
    selection score, not the whole of it (see the checklist below).
  - **AIMC**: your contribution measured against a **reference series**. On a hackathon
    event `explain_scoring` reports that series as **the event's own benchmark
    predictions**, not the crowd consensus the live tournament uses. That is a real
    advantage: the benchmark is downloadable over `train`, so you can build a close
    offline proxy - residualize your predictions against the benchmark per exped, then
    correlate the residual with the target. **`eiq-model-implementation`** carries that
    as a `contribution()` helper you can lift. It is still a proxy, confirmed server-side
    after you submit, but it is not the unobservable quantity a tournament write-up would
    tell you it is.
  - **NCORR**: correlation after neutralizing against a **frozen core feature set**. The
    schema's `core_feature_overlap` tells you how many of those core features fall inside
    each published feature set; the membership is deliberately not published. A high
    overlap is not an escape route: it means the core features already sit inside the ones
    you trained on. Two things before you try to reproduce the number offline. It runs on
    your **rank-gaussianized** predictions, not your raw ones, and the platform neutralizes
    with a spectrally-anchored ridge rather than exact OLS (today's core set is
    rank-deficient, which keeps the ridge branch active), so an exact residualization will
    not match it. NCORR is **null** when none of the core features are present on the
    scored frame, and a null term means no round score at all: those entries rank below
    every scored one.
  - Always sanity-check **correlation-with-benchmark**: a config with high CORR but
    correlation-with-benchmark near 1.0 is re-expressing the benchmark and will earn
    little AIMC.
- **The round score is a weighted blend of CORR, AIMC and NCORR, bounded per round. Call
  `explain_scoring` for the live weights.** Don't hardcode which term dominates; it has
  changed before. On a money event that score is then mapped to a payout through a
  **bounded** function, `A * tanh(payout_factor * score / A)`; `get_event_staking` reports
  the `payout_factor` and `stake_return_amplitude` each round actually froze. Pass them to
  `everestapi.scoring.payout` rather than estimating proportionally.

## The loop in one breath

```
clarify idea → align baseline to the published benchmark → scout round (cheap, sampled)
→ read the offline round score → decide next round → repeat → plateau? → scale winner → confirm
```

---

## Step 0: Clarify the idea (disambiguate before spending compute)

If the request is vague ("try a different signal angle", "make it more robust"), do **not** guess.
Enumerate **2-4 genuinely different interpretations**, run one cheap scout `train(model=<preset>, ...)`
per interpretation on a sampled subset, and let the offline round score pick the winner.

> *"Use the other targets"* could mean: (a) train on one diverse auxiliary target instead
> of the graded one, (b) train per target and rank-blend the predictions, or (c) train on
> the graded target and use the auxiliaries only to build a residualized label. These are
> different experiments. Scout all three on a quarter of the expeds, compare their offline
> round scores, commit to the best, and write down why in `experiment.md`.

Document the chosen interpretation and the rejected ones. That reasoning is part of the result.

## Step 1: Planning checklist (answer before any training)

- **Idea & novelty.** One sentence: what is being tested and why it might add AIMC.
- **Research type.** Name which one kind of change you are testing: a new target or feature
  engineering, a new architecture, an ensemble or blend, a training procedure, or a data
  change. That decides what you may vary and what you must hold fixed; the table is under
  [Sweep selection by research type](#sweep-selection-by-research-type).
- **Baseline.** The published benchmark. Download it over `train` and score it on **your
  own embargoed holdout**, carved from the labeled `train` split, so every round has a
  baseline row. A hackathon key cannot score `validation` locally: its target columns are
  blanked and the practice board scores it server-side.
- **Selection metric** = the **offline round score**: read the live weights from
  `explain_scoring` and apply them to the terms you can measure on your holdout, CORR and
  the `contribution()` AIMC proxy. Do not select on CORR alone. The board ranks on the
  blend, and a model that wins on one term can lose on the score. NCORR cannot be
  reproduced offline (see above), so guard it indirectly with the feature-concentration
  check below. **Diagnostics** = correlation-with-benchmark and per-exped stability.
- **Budget.** Max rounds (≈4-5 expected), compute credits, wall-clock. Check
  `get_compute_credits` before you start so you don't strand a round half-finished. The MCP
  `train` tool's `dry_run=true` validates a call before you launch it, but its
  `estimated_hold_cents` is a flat worst-case reservation for the GPU tier, identical for
  every job on that tier. It tells you what a job can reserve, not what one config costs
  compared to another.
- **Stopping rule.** Pre-commit to the plateau criterion below *now*, before you see results.
  This prevents fishing for a lucky round.

## Step 2: Folder layout (you own this)

One experiment = one line of inquiry = one folder. Keep configs, results, and your
narrative together so the whole study is reproducible from the repo alone.

```
experiments/<experiment_name>/
  experiment.md            # hypothesis, baseline, per-round tables, decisions, final story
  configs/
    r1_lgbm_baseline.yaml  # one file per config; name = the single thing it varies
    r1_xgb_baseline.yaml
    r2_lgbm_depth8.yaml
  results/
    r1.csv                 # holdout metrics for every config in round 1
    r2.csv
  predictions/             # saved holdout/live prediction parquets per promoted config
  best.pkl                 # winning model artifact (downloaded from its compute job)
```

`experiment.md` is the lab notebook. It opens with the hypothesis and the declared baseline,
gains a metrics table after each round, and ends with a short narrative of what worked, what
didn't, and which config won.

## Step 3: Rounds, not runs (persistence is required)

Work in **rounds of ~4-5 configs**. Within a round, change **exactly one variable per config**
so the comparison is causal. A round is only finished when **every** job in it has completed,
poll `get_job_status`, then synthesize. Do not report off a single early-returning run.

Note that "round" here means a round of *your* experiment, not one of the event's sealed
scoring rounds. Keep the two straight in `experiment.md`.

After each round:
1. Score every config on your own embargoed holdout: CORR, the `contribution()` AIMC
   proxy, the offline round score built from them, and correlation-with-benchmark. The
   holdout only counts if the fit never saw it; for a hosted job, that means the
   `train_filter` cutoff in Step 4.
2. Build the round table in `experiment.md`, adding a per-exped stability number. There is
   no cluster breakdown to add on this panel; the time axis is what you have.
3. Pick the round winner by the **offline round score**, with stability and
   correlation-with-benchmark as diagnostics.
4. Decide the next round: which dimension to push, what to drop. Write the decision down.

## Step 4: Scout → Scale

**Keep the holdout out of every hosted fit.** Hosted `train` fits the whole train split
unless you tell it not to, so pass a `train_filter` that ends the fit before your embargo
and holdout begin:

```python
train_filter={"exped": {"cutoff_lt": FIRST_EMBARGO_EXPED}}   # e.g. "exped_6405"
```

`FIRST_EMBARGO_EXPED` is the first exped of the embargo in front of your holdout, read from
the data you downloaded. Without this filter every holdout number is in-sample.

**Scout (early rounds).** Iterate fast and cheap so most ideas die before they cost much.
- Sample the expeds with `train_filter`'s `sample` key alongside the cutoff, e.g.
  `{"exped": {"cutoff_lt": ...}, "sample": {"fraction": 0.25, "unit": "exped"}}`. There is
  no `sample_pct` argument: unknown top-level fields are rejected with a 400.
- **Always pass `features`**, as `"all"` or an explicit list. Left out, it defaults to
  `"small"`, and on a dataset that publishes only `all` (this one) `"small"` silently
  becomes an alphabetical prefix of the feature list, a cutoff rather than a curated set.
  To scout on fewer features, pass the list you chose.
- Run via `train(model=<preset>, gpu="CPU", ...)` (templated configs on the cheapest
  tier), with modest model sizes.
- Evaluate on your **full** embargoed holdout even though you trained on a sample, so the
  metric isn't itself sampling-noisy.

**Scale (later rounds).** Promote only the top 1-2 scout configs by offline round score.
- Drop the `sample` key (keep the cutoff) and widen to more features if you scouted on a
  subset.
- Use `train(model="custom", custom_model_fn=..., gpu=<T4|A10G|A100>, ...)` when a winner
  needs a bigger model or a custom objective the templates don't cover.
- Expect the metric to move when you scale. That's the point. A scout result that
  *collapses* at full scale was overfit to the sample; keep the version that survives.

**One confirmatory scale step.** After you plateau on sampled data, run the surviving config
once on all expeds before the cutoff, with the full feature set it will ship with, to
confirm the edge is real before reporting or submitting. Then refit without the cutoff for
the model you actually submit: the confirmation used the holdout to check the edge, and the
submitted model can use that data too.

## Step 5: Plateau / stopping criteria

Stop when **two consecutive rounds fail to beat the running-best offline round score by a
meaningful margin** *and* the untried knobs are redundant with what you already swept. Then
do the single confirmatory scale step and write the report.

What "meaningful" means is yours to set per study: fix the threshold up front and judge it
against round-to-round noise in the score, not against zero. A tiny wobble inside the noise
band is a plateau, not progress. Record the explicit decision in `experiment.md`, e.g.:

```
Round 1 → 2:  score +0.0042   continue
Round 2 → 3:  score +0.0012   continue
Round 3 → 4:  score +0.0002   within noise
Round 4 → 5:  score +0.0000   within noise  → STOP (2 flat rounds)
Winner: r3_lgbm_depth8  (best offline score, correlation-with-benchmark 0.71, stable across the holdout)
```

(Numbers above are illustrative of the log format only. Do not treat them as a target
to hit; fix your own per-study threshold as described above.)

---

## Sweep selection by research type

Match the sweep to the question. One variable at a time, per config, within a round.

| Research type | What to vary | What to leave fixed |
|---|---|---|
| **New target / feature engineering** | which target you fit (graded vs an auxiliary), feature subset, binning/preprocessing | model + hyperparameters (use a fixed reference model) |
| **New architecture** | depth/width, learning rate, regularization, estimators/epochs | features, target |
| **Ensemble / blend** | member weights, blend rule, bag count, stacker | the members themselves |
| **Training procedure** | residualization strength vs the benchmark, neutralization proportion, loss weighting | model + features |
| **Data change** | exped sampling (which expeds, how many), feature subset within the published set | model + target |

**Never sweep the evaluation itself.** The embargo and the holdout are fixed once, before
round one, and stay fixed for the whole run. Shrink the embargo and CORR goes up because the
leak comes back, so a sweep that selects on CORR will reliably pick the leakiest setting.
Varying the holdout window is the same trap: you end up choosing the period that flatters you.

If one parameter clearly dominates the results, spend a whole round mapping its range
(including the extremes) with everything else pinned.

---

## Event-specific evaluation (don't skip this)

A single average metric hides the things that sink a model here.

- **Per-exped stability & drawdown.** Look at the spread of per-exped CORR and the worst
  run of negative expeds, not just the mean. A high-mean, high-variance config that spends
  long stretches underwater is worse than a steadier one. (Sharpe, std-dev and max
  drawdown are display-only on the board, but they are exactly the right *selection*
  diagnostics offline.)
- **Split the holdout in time.** With no cluster axis, the honest robustness check is
  whether the edge holds in the first half of the holdout as well as the second. An edge
  that lives in one stretch of expeds is a regime artifact, not skill.
- **Feature concentration.** A model resting almost entirely on one or two features is
  fragile and scores poorly on NCORR, which is a scored term. Measure it as the largest
  absolute correlation between your predictions and any single feature, and fix it by
  neutralizing per exped against the heavy block at a swept proportion
  (**`eiq-model-implementation`** has both).
- **Rounds cover different periods.** One round's standing is a relative signal only:
  neither the level nor the ordering of your candidates transfers reliably to the next
  round. Keep several genuinely different models alive rather than betting on last
  round's winner.
- **Select on the round score, not on one term of it.** Boards rank on the weighted blend,
  so a config that gives up some CORR for a larger AIMC can be the better model. Whether it
  is depends on the live weights, so compute it from `explain_scoring` each time rather than
  assuming which term leads. Correlation-with-benchmark is a diagnostic here, never the
  objective: it tells you why the AIMC proxy moved, not whether the model got better.

---

## Baseline alignment to the published benchmark

- Declare the benchmark as the baseline in `experiment.md` and include a baseline row in
  every results table. Name it by the column you actually found in the downloaded frame.
- Pull it once with `download_benchmark("futures", "train")` and score it the same way you
  score your configs, on the same holdout rows, so the comparison is apples-to-apples.
- Keep the feature set consistent between a config and the baseline comparison you cite: a
  richer-feature config beating a small-feature baseline tells you nothing.

---

## Tooling map (MCP + SDK)

| Need | Tool |
|---|---|
| Columns, targets, feature sets, encodings | `get_dataset_schema`, `get_dataset_schema(verbose=True)` for membership |
| Download data | `download_dataset` |
| Download the benchmark | `download_benchmark("futures", "train")` |
| Live scoring weights and metric definitions | `explain_scoring` |
| Cheap templated training (scout) | `train(model=<preset>, features="all", gpu="CPU", train_filter=...)` |
| Custom / GPU training (scale) | `train(model="custom", custom_model_fn=...)` |
| Validate a call before launching | `train(..., dry_run=true)`, **MCP tool only** |
| Poll a training job | `get_job_status` |
| Check budget before a round | `get_compute_credits` |
| Round board / standings | `get_diagnostics_leaderboard`, `get_diagnostics_standings` |
| Submit the winner into a round | `submit_event_predictions` (see `eiq-event-submission`) |

Confirmed signatures: `train(model=<lightgbm|xgboost|ridge|mlp|random_forest>,
features=<a feature-set name from the schema, or an explicit feature list>,
universe="futures", gpu="CPU", params={...})` (universe defaults as shown, and
**omit `target=` unless you deliberately want an auxiliary one**; left out, the
platform trains on the dataset's own graded column;
the **gpu default is `T4`**. Pass `gpu="CPU"` explicitly for cheap scouts; seed via
`params`, e.g. `params={"seed": 7}`. A top-level `seed=` is rejected; returns a job,
poll `get_job_status`). `train(model="custom", custom_model_fn=<source defining
build_model(params) -> estimator>, gpu=<CPU|T4|A10G|A100>, max_hours<=4.0)` (runs in an
isolated, network-denied sandbox; keep the source self-contained). Over MCP, pass
`dry_run=true` to either form to validate the call and resolve its defaults without
launching the job; the Python client's `train()` does not accept it. Check the resolved
`features` in the response: `"small"` there means you forgot to pass it.

Two things about the hosted CV numbers before you select on them: `train` reports the CV
metrics it can compute for your job, which may not include every term the board scores you
on, and it fits the **whole** train split unless a `train_filter` cutoff stops it, so a
"holdout" carved from an unfiltered job's artifacts is in-sample. See Step 4.

A typical scout round, conceptually:

```text
get_compute_credits                      # enough budget for ~4 configs?
get_dataset_schema                       # graded target, encodings, feature-set sizes
download_dataset(universe="futures", split="train")
download_benchmark(universe="futures", split="train")   # the baseline; validation/live are
                                                        # withheld while an event is running
                                                        # (404 by design, not an outage)
explain_scoring                          # the live weights for the offline round score
for cfg in round_1_configs:              # ~4 configs, each varies ONE thing
    train(model=cfg.model, features="all", gpu="CPU",
          train_filter={"exped": {"cutoff_lt": FIRST_EMBARGO_EXPED},
                        "sample": {"fraction": 0.25, "unit": "exped"}})  -> job_id
poll get_job_status(job_id) until all done
# score each config on YOUR embargoed holdout: CORR, contribution(), the offline round
# score from the live weights, corr-with-benchmark as a diagnostic
# write results/r1.csv + experiment.md table; pick the best score; decide round 2
```

## Reporting

When you stop, write the closing section of `experiment.md` as a short scientific narrative,
not a metrics dump: the hypothesis, the path the rounds took, what won and *why*, the final
table (with the benchmark baseline row and the per-exped stability numbers), and the explicit
stopping decision. Then submit the confirmed winner with `submit_event_predictions` (via
`eiq-event-submission`) if entering the open round is the goal, so a single session can carry
an idea from clarification all the way to a submitted model.
