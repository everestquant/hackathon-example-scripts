---
name: eiq-experiment-design
description: >-
  Plan, run, and iterate Everesteer hackathon-event experiments using the scout→scale
  research loop. Clarify the idea, align a baseline against the event's published
  benchmark, write configs, train via the Everesteer MCP server (the unified train tool,
  templated or custom), select experiments on CORR plus divergence-from-benchmark as the
  offline read on AIMC (a scored term), iterate in rounds, stop at a plateau, and scale
  the winner. Use when asked to design an event experiment, decide what to try next, or
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
- CV is **exped-purged + embargoed**, which the hosted `train` tool does for you.
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
    scored term, and the one you can measure precisely offline every round. Primary
    selection lever.
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
    each published feature set; the membership is deliberately not published. Note the
    platform computes it with a spectrally-anchored ridge, not exact OLS, so reimplementing
    exact residualization yourself will not reproduce the number.
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
→ read CORR + correlation-with-benchmark → decide next round → repeat → plateau? → scale winner → confirm
```

---

## Step 0: Clarify the idea (disambiguate before spending compute)

If the request is vague ("try a different signal angle", "make it more robust"), do **not** guess.
Enumerate **2-4 genuinely different interpretations**, run one cheap scout `train(model=<preset>, ...)`
per interpretation on a sampled subset, and let CORR plus correlation-with-benchmark pick the winner.

> *"Use the other targets"* could mean: (a) train on one diverse auxiliary target instead
> of the graded one, (b) train per target and rank-blend the predictions, or (c) train on
> the graded target and use the auxiliaries only to build a residualized label. These are
> different experiments. Scout all three at `sample_pct≈0.25`, compare CORR and
> correlation-with-benchmark, commit to the best, and write down why in `experiment.md`.

Document the chosen interpretation and the rejected ones. That reasoning is part of the result.

## Step 1: Planning checklist (answer before any training)

- **Idea & novelty.** One sentence: what is being tested and why it might add AIMC.
- **Research type.** New target/feature-eng · new architecture · ensemble/blend ·
  training-procedure · data change. This decides what you sweep (see below).
- **Baseline.** The published benchmark. Download it over `train` and score it on **your
  own embargoed holdout**, carved from the labeled `train` split, so every round has a
  baseline row. A hackathon key cannot score `validation` locally: its target columns are
  blanked and the practice board scores it server-side.
- **Primary metric** = **CORR** (selection, a scored term, above noise; see
  `explain_scoring` for live weights). **Differentiation guard** = correlation-with-benchmark
  (lower is better; it is the offline read on AIMC). **Diagnostics** = per-exped stability.
- **Budget.** Max rounds (≈4-5 expected), compute credits, wall-clock. Check
  `get_compute_credits` before you start so you don't strand a round half-finished, and use
  the MCP `train` tool's `dry_run=true` to preview a config's `estimated_hold_cents`
  before committing credits to it.
- **Stopping rule.** Pre-commit to the plateau criterion below *now*, before you see results.
  This prevents fishing for a lucky round.

## Step 2: Folder layout (you own this)

One experiment = one line of inquiry = one folder. Keep configs, results, and your
narrative together so the whole study is reproducible from the repo alone.

```
experiments/<experiment_name>/
  experiment.md            # hypothesis, baseline, per-round tables, decisions, final story
  configs/
    r1_lgbm_small.yaml     # one file per config; name = the single thing it varies
    r1_xgb_small.yaml
    r2_lgbm_all.yaml
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
1. Score every config on your own embargoed holdout: CORR, correlation-with-benchmark, and
   the `contribution()` AIMC proxy.
2. Build the round table in `experiment.md`, adding a per-exped stability number. There is
   no cluster breakdown to add on this panel; the time axis is what you have.
3. Pick the round winner by **CORR** (lower correlation-with-benchmark as tie-breaker;
   stability as a diagnostic).
4. Decide the next round: which dimension to push, what to drop. Write the decision down.

## Step 4: Scout → Scale

**Scout (early rounds).** Iterate fast and cheap so most ideas die before they cost much.
- Sample the expeds: a **scout subset** = every Nth exped (e.g. `sample_pct≈0.25`, ~25%).
- Use a small feature subset and modest model sizes.
- Run via `train(model=<preset>, gpu="CPU", ...)` (templated configs on the cheapest
  tier). Over MCP, `dry_run=true` gives a cost preview first.
- Evaluate on your **full** embargoed holdout even though you trained on a sample, so the
  metric isn't itself sampling-noisy.

**Scale (later rounds).** Promote only the top 1-2 scout configs (by CORR, with lower
correlation-with-benchmark as the tie-breaker).
- Move to full expeds (`sample_pct≈1.0`) and richer features.
- Use `train(model="custom", custom_model_fn=..., gpu=<T4|A10G|A100>, ...)` when a winner
  needs a bigger model or a custom objective the templates don't cover.
- Expect the metric to move when you scale. That's the point. A scout result that
  *collapses* at full scale was overfit to the sample; keep the version that survives.

**One confirmatory scale step.** After you plateau on sampled data, run the surviving config
once at full expeds + full features to confirm the edge is real before reporting/submitting.

## Step 5: Plateau / stopping criteria

Stop when **two consecutive rounds fail to beat the running-best CORR by a meaningful
margin** *and* the untried knobs are either redundant with what you already swept or
likely to just raise correlation-with-benchmark (which would erode the AIMC term). Then do
the single confirmatory scale step and write the report.

What "meaningful" means is yours to set per study: fix the threshold up front and judge it
against round-to-round CORR noise, not against zero. A tiny wobble inside the noise band is a
plateau, not progress. Record the explicit decision in `experiment.md`, e.g.:

```
Round 1 → 2:  CORR +0.0021   continue
Round 2 → 3:  CORR +0.0006   continue
Round 3 → 4:  CORR +0.0001   within noise
Round 4 → 5:  CORR +0.0000   within noise  → STOP (2 flat rounds)
Winner: r3_lgbm_all  (best CORR, correlation-with-benchmark 0.71, stable across the holdout)
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
| **Training procedure** | residualization strength vs the benchmark, neutralization proportion, loss weighting, embargo | model + features |
| **Data change** | exped sampling, feature-set size, holdout length | model + target |

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
- **CORR first; correlation-with-benchmark as the differentiation guard, never the
  objective.** Boards rank on the round score, a weighted blend. Call `explain_scoring`
  for the live weights, and never trade real CORR for uniqueness. *Then* prefer designs
  that diverge from the benchmark: when two configs tie on CORR, take the one with lower
  correlation-with-benchmark, because that is what the AIMC term pays for.

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
| Cheap templated training (scout) | `train(model=<preset>, gpu="CPU")` |
| Custom / GPU training (scale) | `train(model="custom", custom_model_fn=...)` |
| Preview cost before launching | `train(..., dry_run=true)`, **MCP tool only** |
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
`dry_run=true` to either form to get a cost preview (`estimated_hold_cents`) without
launching the job; the Python client's `train()` does not accept it.

Two things about the hosted CV numbers before you select on them: `train` reports the CV
metrics it can compute for your job, which may not include every term the board scores you
on, and it fits the **whole** train split, so a "holdout" carved from its artifacts
afterwards is in-sample. Build your honest holdout from your own split of `train`.

A typical scout round, conceptually:

```text
get_compute_credits                      # enough budget for ~4 configs?
get_dataset_schema                       # graded target, encodings, feature-set sizes
download_dataset(universe="futures", split="train")
download_benchmark(universe="futures", split="train")   # the baseline; validation/live are
                                                        # withheld while an event is running
                                                        # (404 by design, not an outage)
for cfg in round_1_configs:              # ~4 configs, each varies ONE thing, sampled expeds
    train(model=cfg.model, gpu="CPU", ...)  -> job_id
poll get_job_status(job_id) until all done
# score each config on YOUR embargoed holdout: CORR, corr-with-benchmark, contribution()
# write results/r1.csv + experiment.md table; pick best CORR; decide round 2
```

## Reporting

When you stop, write the closing section of `experiment.md` as a short scientific narrative,
not a metrics dump: the hypothesis, the path the rounds took, what won and *why*, the final
table (with the benchmark baseline row and the per-exped stability numbers), and the explicit
stopping decision. Then submit the confirmed winner with `submit_event_predictions` (via
`eiq-event-submission`) if entering the open round is the goal, so a single session can carry
an idea from clarification all the way to a submitted model.
