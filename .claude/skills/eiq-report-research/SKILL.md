---
name: eiq-report-research
description: Turn a finished Everesteer hackathon-event experiment run into a durable, scientific write-up in experiment.md (abstract, motivation, method, results table, decisions, stopping rationale, findings, next steps) and generate/link the standard cumulative-CORR plot. Use after running Everesteer event experiments, or when asked to "write up the results", "produce a full report", "update experiment.md", or "generate the standard plot".
---

# Everesteer Report Research

Convert one or more experiment runs in your `experiments/` folder into a finished report
a reader can follow end to end: what you tested, what won, why you stopped, and whether
you'd stake on it. The goal is a short scientific paper, not a metrics dump.

You own everything here: the `experiments/` folder, the `everestapi` SDK, the Everesteer MCP
tools, and the plotting/scoring helpers shipped in this example-scripts repo. There is no
internal platform repo to call into.

## Ground truth (Everesteer hackathon event)

- Time unit is the **exped**. The panel is obfuscated: rows carry no instrument identity
  and there is **no cluster or sector column**, so time is the only axis a breakdown can
  use. `get_features` 403s a hackathon key and `get_universe` comes back empty; read
  columns from `get_dataset_schema`.
- Primary target: the column `get_dataset_schema` reports as `primary_target`. Benchmark:
  whatever `download_benchmark("futures", "train")` serves, named by the column you find
  in that frame rather than assumed.
- Round score: a weighted blend of CORR, AIMC and NCORR, bounded per round. Call
  `explain_scoring` for the live weights; don't hardcode which term dominates, it has
  changed before. Uniqueness pays more than raw accuracy, say so in the write-up. On a
  money event the score is then mapped to a payout through a **bounded** function,
  `A * tanh(payout_factor * score / A)`; `get_event_staking` reports the `payout_factor`
  and `stake_return_amplitude` each round froze, and `everestapi.scoring.payout` takes both.
  Cumulative standings carry the **exped-weighted mean** of per-round scores, never a sum.
- Always report these:
  - **CORR**: mean per-exped rank correlation of your predictions vs the target; also a
    scored term (see `explain_scoring` for the live weights). This is the primary
    *experiment-selection* metric, since it's the one number you can compute precisely
    offline every round. Report it **two ways**: full-period CORR and a recent-window
    CORR (most recent ~20-40 expeds).
  - **AIMC**: your contribution over a reference series. On a hackathon event
    `explain_scoring` reports that reference as **the event's own benchmark predictions**,
    not a crowd consensus, and the benchmark is downloadable over `train`. So unlike the
    tournament case you can report a genuine offline proxy: residualize predictions
    against the benchmark per exped, correlate the residual with the target (the
    `contribution()` helper in **`eiq-model-implementation`**). Label
    it as the proxy it is, and report the server's number where rounds have resolved.
  - **NCORR**: correlation after neutralizing against a frozen core feature set whose
    membership is not published. Report it where rounds have resolved; note that the
    platform runs it on **rank-gaussianized** predictions and neutralizes with a
    spectrally-anchored ridge, so a local OLS residualization on raw predictions will not
    reproduce it. Report a resolved NCORR of `null` as null, never as zero: it means the
    core features were absent from the scored frame, which leaves that round without a
    round score.
  - **correlation-with-benchmark**: corr of your preds with the benchmark series. This is
    the tell for the "high CORR, high correlation-with-benchmark" trap: a model that just
    re-expresses the benchmark and will earn little AIMC.
  - **stability**: per-exped sharpe (mean/std of the per-exped score) and max drawdown
    of the cumulative score. Display-only on the board, but the right selection diagnostic
    offline.

## Step 1: Inventory what actually ran

Find the experiment folder. A typical layout:

```
experiments/<experiment_name>/
  experiment.yaml        # hypothesis + config
  configs/               # per-model configs you defined
  results/               # metric outputs (one file per run that executed)
  predictions/           # out-of-sample preds (one file per run that executed)
  experiment.md          # the report you write here
  plots/                 # standard plot output
```

Separate **what ran** from **what is only configured**:
- A config that has a matching `results/` + `predictions/` artifact ran.
- A config with no artifacts is *planned only*. It goes in the report as "configured,
  not run", never in the results table as if it had numbers.

If the run was staged in rounds, capture each round's **intent** (what changed vs the
prior round) and whether it beat the running best.

## Step 2: Pull the numbers

Compute metrics from the out-of-sample predictions you already hold locally, or pull them
with the SDK / MCP for anything already submitted:
- `run_validation_diagnostics`: validation-split metrics for a candidate (MCP name for
  `get_validation_diagnostics`).
- `get_diagnostics_leaderboard`: the board for a round you've submitted to.
- `get_diagnostics_standings`: cumulative standings across rounds, for context vs the field.

Build the per-exped stability series (sharpe, drawdown) yourself from the out-of-sample
predictions on disk. This skill's numbers should trace back to files in your own
`experiments/` folder wherever possible.

Pick the **best model by CORR** (recent-window CORR breaks ties). CORR is the
experiment-selection metric, since it's the one number you can compute precisely
offline. Use correlation-with-benchmark as the differentiation check and per-exped
stability (plus resolved-round AIMC where available) to confirm the edge isn't a single
lucky exped. A high-CORR model with high correlation-with-benchmark is *not* clearly the
winner, flag it as a likely benchmark-echo and note that its AIMC, once a round
resolves, may disappoint.

## Step 3: Write experiment.md

Use this template. Keep prose tight; every section earns its place.

```markdown
# <Experiment Name>: Experiment Report

**Date:** YYYY-MM-DD
**Event dataset:** futures
**Target:** <the schema's primary_target>
**Selection metric:** CORR, with correlation-with-benchmark as the differentiation guard  ·  **Round score:** weighted CORR+AIMC+NCORR blend, bounded per round (see `explain_scoring` for live weights)

## Abstract
Two to four sentences: what was tested, the headline result, and the decision
(stake / not yet). Lead with CORR and correlation-with-benchmark (AIMC alongside where resolved).

## Motivation
Why this idea should produce alpha *beyond the reference series*. I.e. why it should lower
correlation-with-benchmark (and so raise AIMC), not just raise CORR. State the hypothesis
you set out to test.

## Method
- Data: train / validation / live exped ranges actually used.
- Feature set and any transforms.
- Model type(s) and key hyperparameters.
- Cross-validation: scheme + embargo. The horizon is a dataset fact and this dataset does
  not publish it - not in the target name, not in any schema field - so state the embargo
  you chose and that it was chosen wide on purpose, rather than implying you matched a
  known horizon.
- How each round differed from the previous (if staged).

## Experiments run
One short subsection per config that *actually ran*. Name the artifacts
(results/preds files). List planned-but-not-run configs separately and clearly.

## Results

| Model | Round | CORR (full) | CORR (recent) | corr_w/_benchmark | AIMC (resolved) | per-exped sharpe | max DD | payout (est) | Status |
|-------|-------|-------------|----------------|-------------------|------------------|------------------|--------|--------------|--------|
| ...   | ...   | ...         | ...            | ...               | ...              | ...              | ...    | ...          | best / kept / dropped |

`payout (est)` is the weighted CORR+AIMC+NCORR blend, before the payout factor and (on
staked events) the per-round return bound. `explain_scoring` reads the weights and the
bound live, so don't hardcode an ordering. Call out any high-CORR / high-corr_w/_benchmark
rows explicitly. Accuracy that differentiates nothing scores well offline and still pays
badly once AIMC resolves.

### Round-by-round
For each round: what changed, the best result, and whether it beat the prior best. Keep
your experiment rounds and the event's sealed scoring rounds clearly distinct.

### Robustness over time
There is no cluster or sector axis on this panel, so the fragility check is temporal: split
the holdout in half and report whether the edge survives in both. An edge confined to one
stretch of expeds is a regime artifact, say so. Per-exped CORR spread and the worst run of
negative expeds belong here too.

## Standard plot
![cumulative CORR and correlation-with-benchmark](plots/cumulative_corr.png)
Cumulative CORR of the best model, and its rolling correlation with the published
benchmark, over expeds. Interpret it: is CORR accumulating steadily, and is
correlation-with-benchmark trending down (more differentiated) or up (converging on the
benchmark)?

## Decisions
The choices you made and why (feature set, model family, sweep picks, per-exped vs
global). Frame them against correlation-with-benchmark (the differentiation guard), not
just CORR.

## Stopping rationale
Why you stopped iterating, e.g. CORR plateau over N rounds, recent-window
correlation-with-benchmark no longer improving, diminishing payout per round, or a
confirmatory full-data run after a scout phase.

## Findings
What worked, what didn't, what the plot and the over-time robustness view actually show.
Honest about negative results.

## What we'd stake / why (or not yet)
A clear call in payout terms: would you stake this model, and why, or what specifically
must improve first (e.g. temporal breadth, benchmark de-correlation, resolved-round AIMC
once available).

## Next experiments
2-5 concrete, prioritized follow-ups tied to the findings above.
```

## Step 4: Generate the standard plot

The standard Everesteer plot is **cumulative CORR of the best model, plus its rolling
correlation with the published benchmark, over expeds**, built from the run's
out-of-sample predictions.

If the example-scripts repo ships a plotting helper, use it, e.g.:

```bash
python plot_experiment.py \
  --predictions experiments/<name>/predictions/<best_model>.parquet \
  --benchmark benchmark_futures_train.parquet \
  --out experiments/<name>/plots/cumulative_corr.png
```

Otherwise (the fallback always works), a minimal matplotlib equivalent (compute the
per-exped CORR series, cumsum it, and the rolling correlation-with-benchmark series,
plot both vs the benchmark line):

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# expeds: ordered list; corr_cum: cumulative per-exped CORR;
# bench_corr_roll: rolling correlation-with-benchmark; bench_cum: cumulative benchmark score
NAVY, TEAL, CORAL = "#09142F", "#007B63", "#EC9A5F"
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(expeds, corr_cum, color=TEAL, label="Cumulative CORR")
ax.plot(expeds, bench_corr_roll, color=CORAL, label="Rolling correlation-with-benchmark")
ax.plot(expeds, bench_cum, color=NAVY, linestyle="--", label="published benchmark")
ax.axhline(0, color=NAVY, linewidth=0.5)
ax.set_xlabel("exped"); ax.set_ylabel("cumulative score / correlation")
ax.set_title("Best model vs the published benchmark over expeds")
ax.legend()
fig.tight_layout()
fig.savefig("experiments/<name>/plots/cumulative_corr.png", dpi=150)
```

Embed it with a **relative** link so it resolves from inside the experiment folder. If
you have several strong candidates, either overlay them on one plot or emit one per
candidate and link each.

## Step 5: Final checks

- The plot file exists under `plots/` and the relative link in `experiment.md` resolves.
- Every number in the results table traces back to a real `results/` artifact.
- Runs that only have a config (no artifacts) are labeled planned, never tabulated as
  results.
- CORR is reported both full-period and recent-window (AIMC alongside where rounds have
  resolved); correlation-with-benchmark is shown so high-CORR/benchmark-echo cases are
  visible.
- The over-time robustness split is present and interpreted (there is no cluster axis on
  this panel to break down instead).
- The payout framing uses the weighted CORR+AIMC+NCORR blend, per `explain_scoring` (no
  hardcoded ordering or cap number).
- The "what we'd stake / why (or not yet)" conclusion is explicit.
- No synthetic data: all metrics come from real Everesteer predictions and scores.
