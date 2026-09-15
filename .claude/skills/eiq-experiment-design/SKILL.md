---
name: eiq-experiment-design
description: >-
  Plan, run, and iterate Everesteer hackathon-event (futures dataset) experiments using
  the scout→scale research loop. Clarify the idea, align a baseline against ai_model,
  write configs, train via the Everesteer MCP server (the unified train tool, templated
  or custom), select experiments on CORR plus divergence-from-benchmark as the offline
  read on likely AIMC (a live payout metric), iterate in rounds, stop at a plateau, and
  scale the winner. Use
  when asked to design an event experiment, decide what to try next, or turn a model
  idea into a structured, multi-round research plan.
---

# Everesteer Experiment Design (Event / Futures dataset)

A repeatable research loop for an Everesteer hackathon event's sealed rounds, run
against the futures dataset. You hold the `everestapi` SDK, the Everesteer MCP server,
your downloaded datasets, and the example-scripts helpers — that is all you need.
Everything below is framed around those tools and a `configs/` + `experiments/` layout
that **you own** in your own repo.

The job is never "get one good run." It is: turn an idea into a sequence of cheap,
interpretable rounds, read the evidence, and decide the next round — until the signal
stops improving.

## Ground truth you must not get wrong

- **Universe** = futures **chains** grouped into **clusters** (energy, rates, ags, FX,
  metals, equity, power, vol, …). Discover the live shape with `get_universe` /
  `get_features` — never hardcode counts.
- **Time unit** = **exped** (plural *expeds*). CV is **exped-purged + embargoed**.
- **Primary target**: the column `get_dataset_schema` reports as `primary_target`. Never hardcode it, and do not assume a horizon from
  the name: most datasets do not encode one there.
- **Features** are encoded into cross-sectional bins. The bin count, value range and
  missing sentinel come from the schema's `feature_encoding` — read it rather than
  assuming. They are already binned — do not re-standardize them.
- **Benchmark** = `ai_model` (pull it with `download_benchmark`). Your baseline and every
  comparison aligns to it.
- **Metrics**:
  - **CORR** — per-exped rank correlation of your predictions vs the graded
    target; one of
    the payout components (call `explain_scoring` for the live weights). This is the one
    metric you can measure precisely offline, every round — treat it as your primary
    selection lever.
  - **AIMC** — AI Model Contribution: your unique signal beyond the *live* stake-weighted
    ai-model consensus. It is a paid component and the ultimate target, but it is only
    measurable once a round resolves — you cannot compute it during scout/scale rounds.
    Offline, you cannot read AIMC directly; the best you can do is track **CORR** alongside
    **correlation-with-benchmark** (below) as a qualitative read on whether a config is
    likely to differentiate once it resolves. Do not invent an offline AIMC number.
  - **NCORR** — Neutralized Correlation: your predictions' correlation with the target after
    neutralizing against dominant feature exposures. A paid futures term alongside CORR
    and AIMC.
  - Always sanity-check **correlation-with-benchmark** (corr of your predictions vs
    `ai_model`): a config with high CORR but correlation-with-benchmark near 1.0 is just
    re-expressing the static benchmark / `ai_model` and is unlikely to earn AIMC once the
    round resolves.
- **Payout is a weighted blend of CORR, AIMC, and NCORR — call `explain_scoring` for the
  live weights and cap.** Don't hardcode which term dominates; it has changed before.
  Design for divergence from the crowd — low correlation-with-benchmark, not just high
  CORR — since that is what AIMC pays for once it resolves. That score is then scaled by
  a per-round **payout factor**, frozen at the round's stake lock: 1 below a fixed
  total-stake threshold, shrinking above it, so it can differ round to round.

## The loop in one breath

```
clarify idea → align baseline to ai_model → scout round (cheap, sampled)
→ read CORR + correlation-with-benchmark → decide next round → repeat → plateau? → scale winner → confirm
```

---

## Step 0 — Clarify the idea (disambiguate before spending compute)

If the request is vague ("try a directional-signal angle", "make it more robust"), do **not** guess.
Enumerate **2–4 genuinely different interpretations**, run one cheap scout `train(model=<preset>, ...)`
per interpretation on a sampled subset, and let CORR plus correlation-with-benchmark pick the winner.

> *"Add cross-asset features"* could mean: (a) include FX + rates features in an
> energy-focused model, (b) train one model across all clusters jointly, or (c) build
> per-cluster models and blend. These are different experiments. Scout all three at
> `sample_pct≈0.25`, compare CORR and correlation-with-benchmark, commit to the best, and
> write down why in `experiment.md`.

Document the chosen interpretation and the rejected ones — that reasoning is part of the result.

## Step 1 — Planning checklist (answer before any training)

- **Idea & novelty.** One sentence: what is being tested and why it might add AIMC.
- **Research type.** New target/feature-eng · new architecture · ensemble/blend ·
  training-procedure · data/universe change. This decides what you sweep (see below).
- **Baseline.** Always `ai_model`. Download it and score it on **your own embargoed
  holdout**, carved from the labeled `train` split, so every round has a baseline row. A
  hackathon key cannot score `validation` locally: its target columns are blanked and the
  practice board scores it server-side.
- **Primary metric** = **CORR** (selection — a payout component, above noise; see
  `explain_scoring` for live weights). **Differentiation guard** = correlation-with-benchmark
  (lower is better — it is the offline read on likely AIMC once a round resolves).
  **Diagnostics** = per-exped stability.
- **Budget.** Max rounds (≈4–5 expected), compute credits, wall-clock. Check
  `get_compute_credits` before you start so you don't strand a round half-finished, and use
  the MCP `train` tool's `dry_run=true` to preview a config's `estimated_hold_cents`
  before committing credits to it.
- **Stopping rule.** Pre-commit to the plateau criterion below *now*, before you see results
  — this prevents fishing for a lucky round.

## Step 2 — Folder layout (you own this)

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
    r1.csv                 # validation metrics for every config in round 1
    r2.csv
  predictions/             # saved val/live prediction parquets per promoted config
  best.pkl                 # winning model artifact (downloaded from its compute job)
```

`experiment.md` is the lab notebook. It opens with the hypothesis and the declared baseline,
gains a metrics table after each round, and ends with a short narrative of what worked, what
didn't, and which config won.

## Step 3 — Rounds, not runs (persistence is required)

Work in **rounds of ~4–5 configs**. Within a round, change **exactly one variable per config**
so the comparison is causal. A round is only finished when **every** job in it has completed —
poll `get_job_status`, then synthesize. Do not report off a single early-returning run.

After each round:
1. Pull validation diagnostics for each config (`run_validation_diagnostics`).
2. Build the round table in `experiment.md`: CORR, correlation-with-benchmark, plus a
   per-cluster CORR breakdown and a stability number (AIMC once rounds resolve).
3. Pick the round winner by **CORR** (lower correlation-with-benchmark as tie-breaker;
   stability as a diagnostic).
4. Decide the next round: which dimension to push, what to drop. Write the decision down.

## Step 4 — Scout → Scale

**Scout (early rounds).** Iterate fast and cheap so most ideas die before they cost much.
- Sample the expeds: a **scout subset** = every Nth exped (e.g. `sample_pct≈0.25`, ~25%).
- Use a small feature subset and modest model sizes.
- Run via `train(model=<preset>, gpu="CPU", ...)` (templated configs on the cheapest
  tier) — this is the cheap tier. Over MCP, `dry_run=true` gives a cost preview first.
- Evaluate on your **full** embargoed holdout even though you trained on a sample, so the
  metric isn't itself sampling-noisy.

**Scale (later rounds).** Promote only the top 1–2 scout configs (by CORR, with lower
correlation-with-benchmark as the tie-breaker).
- Move to full expeds (`sample_pct≈1.0`) and richer features.
- Use `train(model="custom", custom_model_fn=..., gpu=<T4|A10G|A100>, ...)` when a winner
  needs a bigger model or a custom objective the templates don't cover.
- Expect the metric to move when you scale — that's the point. A scout result that
  *collapses* at full scale was overfit to the sample; keep the version that survives.

**One confirmatory scale step.** After you plateau on sampled data, run the surviving config
once at full expeds + full features to confirm the edge is real before reporting/submitting.

## Step 5 — Plateau / stopping criteria

Stop when **two consecutive rounds fail to beat the running-best CORR by a meaningful
margin** *and* the untried knobs are either redundant with what you already swept or
likely to just raise correlation-with-benchmark (which would erode differentiation, and
so the AIMC you'd expect once a round resolves). Then do the single confirmatory scale
step and write the report.

What "meaningful" means is yours to set per study — fix the threshold up front and judge it
against round-to-round CORR noise, not against zero. A tiny wobble inside the noise band is a
plateau, not progress. Record the explicit decision in `experiment.md`, e.g.:

```
Round 1 → 2:  CORR +0.0021   continue
Round 2 → 3:  CORR +0.0006   continue
Round 3 → 4:  CORR +0.0001   within noise
Round 4 → 5:  CORR +0.0000   within noise  → STOP (2 flat rounds)
Winner: r3_lgbm_all  (best CORR, correlation-with-benchmark 0.71, holds across all clusters)
```

(Numbers above are illustrative of the log format only — do not treat them as a target
to hit; fix your own per-study threshold as described above.)

---

## Sweep selection by research type

Match the sweep to the question. One variable at a time, per config, within a round.

| Research type | What to vary | What to leave fixed |
|---|---|---|
| **New target / feature engineering** | target variant or transform, feature subset, binning/preprocessing | model + hyperparameters (use a fixed reference model) |
| **New architecture** | depth/width, learning rate, regularization, estimators/epochs | features, target |
| **Ensemble / blend** | member weights, blend rule, bag count, stacker | the members themselves |
| **Training procedure** | residualization strength vs `ai_model`, loss weighting, cluster sample-weights, embargo | model + features |
| **Data / universe change** | cluster inclusion, exped sampling, feature-set size | model + target |

If one parameter clearly dominates the results, spend a whole round mapping its range
(including the extremes) with everything else pinned.

---

## Futures-specific evaluation (don't skip this)

The event dataset is futures, and a single average metric hides the things that sink futures models.

- **Cluster-aware CORR.** Break CORR down per cluster (and AIMC per cluster once rounds
  resolve). An edge that lives entirely in one cluster (e.g. energy) is fragile and
  may be a roll/liquidity artifact, not skill. Favor configs whose CORR is positive across
  *several* clusters.
- **Per-exped stability & drawdown.** Look at the spread of per-exped CORR/AIMC and the
  worst run of negative expeds, not just the mean. A high-mean, high-variance config that
  spends quarters underwater is worse than a steadier one at the cap.
- **Contract-roll / liquidity awareness.** Continuous-futures series carry roll seams, and
  thin contracts add noise. Distrust an edge concentrated around roll windows or in the
  least-liquid chains — verify it survives when those expeds/instruments are down-weighted.
- **Cluster sample-weighting.** Clusters differ in size and signal density. Consider
  weighting so a few large clusters don't quietly dominate training; treat the weighting
  itself as a sweepable training-procedure dimension.
- **CORR first; correlation-with-benchmark as the differentiation guard, never the
  objective.** Every board ranks on the round score, a weighted blend — call
  `explain_scoring` for the live
  weights — so never trade CORR for uniqueness. *Then* prefer designs that diverge from
  the static benchmark, `ai_model`, and the crowd: when two configs tie on CORR, take the
  one with lower correlation-with-benchmark — that differentiation is what the AIMC term
  pays for once a round resolves.

---

## Baseline alignment to `ai_model`

- Declare `ai_model` as the baseline in `experiment.md` and include a baseline row in every
  results table.
- Pull it once per universe/split with `download_benchmark` and score it the same way you
  score your configs, so the correlation-with-benchmark comparison (and AIMC once rounds
  resolve) is apples-to-apples.
- Keep the feature set consistent between a config and the baseline comparison you cite — a
  richer-feature config beating a small-feature baseline tells you nothing.

---

## Tooling map (MCP + SDK)

| Need | Tool |
|---|---|
| See the universe / clusters / features | `get_universe`, `get_features` |
| Download data + understand columns | `download_dataset`, `get_dataset_schema` |
| Download the `ai_model` benchmark | `download_benchmark` |
| Cheap templated training (scout) | `train(model=<preset>, gpu="CPU")` |
| Custom / GPU training (scale) | `train(model="custom", custom_model_fn=...)` |
| Preview cost before launching | `train(..., dry_run=true)` — **MCP tool only** |
| Poll a training job | `get_job_status` |
| Check budget before a round | `get_compute_credits` |
| Validation metrics for a config | `run_validation_diagnostics` |
| Round board / standings | `get_diagnostics_leaderboard`, `get_diagnostics_standings` |
| Submit the winner into a round | `submit_event_predictions` (see `eiq-event-submission`) |

Confirmed signatures: `train(model=<lightgbm|xgboost|ridge|mlp|random_forest>,
features=<a feature-set name from the schema, or an explicit feature list>,
universe="futures", gpu="CPU", params={...})` (universe defaults as shown, and
**omit `target=` unless you deliberately want an auxiliary one** — left out, the
platform trains on the dataset's own graded column;
the **gpu default is `T4`** — pass `gpu="CPU"` explicitly for cheap scouts; seed via
`params`, e.g. `params={"seed": 7}` — a top-level `seed=` is rejected; returns a job —
poll `get_job_status`). `train(model="custom", custom_model_fn=<source defining
build_model(params) -> estimator>, gpu=<CPU|T4|A10G|A100>, max_hours<=4.0)` (runs in an
isolated, network-denied sandbox; keep the source self-contained). Over MCP, pass
`dry_run=true` to either form to get a cost preview (`estimated_hold_cents`) without
launching the job; the Python client's `train()` does not accept it.

A typical scout round, conceptually:

```text
get_compute_credits                      # enough budget for ~4 configs?
get_universe / get_features              # confirm clusters + feature names this round
download_dataset(universe="futures", split="train")
download_benchmark(universe="futures", split="train")   # ai_model baseline; validation/live are
                                                          # withheld from a hackathon-scoped key
                                                          # while an event is running (404 by design)
for cfg in round_1_configs:              # ~4 configs, each varies ONE thing, sampled expeds
    train(model=cfg.model, gpu="CPU", ...)  -> job_id
poll get_job_status(job_id) until all done
run_validation_diagnostics(...) per config
# write results/r1.csv + experiment.md table; pick best CORR; decide round 2
```

## Reporting

When you stop, write the closing section of `experiment.md` as a short scientific narrative,
not a metrics dump: the hypothesis, the path the rounds took, what won and *why*, the final
table (with the `ai_model` baseline row and per-cluster CORR, AIMC where resolved), and the
explicit stopping decision. Then submit the confirmed winner with `submit_event_predictions`
(via `eiq-event-submission`) if entering the open round is the goal — so a single session
can carry an idea from clarification all the way to a submitted model.
