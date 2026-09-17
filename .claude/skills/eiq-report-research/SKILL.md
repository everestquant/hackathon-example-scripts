---
name: eiq-report-research
description: Turn a finished Everesteer hackathon-event (futures dataset) experiment run into a durable, scientific write-up in experiment.md (abstract, motivation, method, results table, decisions, stopping rationale, findings, next steps) and generate/link the standard cumulative-CORR plot. Use after running Everesteer event experiments, or when asked to "write up the results", "produce a full report", "update experiment.md", or "generate the standard plot".
---

# Everesteer Report Research

Convert one or more experiment runs in your `experiments/` folder into a finished report
a reader can follow end to end: what you tested, what won, why you stopped, and whether
you'd stake on it. The goal is a short scientific paper, not a metrics dump.

You own everything here: the `experiments/` folder, the `everestapi` SDK, the Everesteer MCP
tools, and the plotting/scoring helpers shipped in this example-scripts repo. There is no
internal platform repo to call into.

## Ground truth (Everesteer event: futures dataset)

- The event's rounds run on the futures dataset. Time unit is the **exped**.
- Primary target: the column `get_dataset_schema` reports as `primary_target`. Consensus benchmark: `ai_model`.
- Payout: a weighted blend of CORR, AIMC, and NCORR, capped per round. Call
  `explain_scoring` for the live weights and cap; don't hardcode which term dominates, it
  has changed before. Uniqueness pays more than raw accuracy, say so in the write-up. That
  score is then scaled by a per-round **payout factor**, frozen at the round's stake lock:
  1 below a fixed total-stake threshold, shrinking above it, so it can differ round to round.
- Always report these:
  - **CORR**: mean per-exped rank correlation of your predictions vs the target; also a
    payout component (see `explain_scoring` for the live weights). This is the primary
    *experiment-selection* metric, since it's the one number you can compute precisely
    offline every round. Report it **two ways**: full-period CORR and a recent-window
    CORR (most recent ~20-40 expeds).
  - **AIMC**, AI Model Contribution: contribution beyond the live stake-weighted
    ai-model consensus; a paid component, but only measurable once a round resolves.
    Report it where rounds have resolved; do not fabricate an offline substitute.
  - **NCORR**, Neutralized Correlation: the other paid futures term, alongside CORR and
    AIMC. Report it where rounds have resolved.
  - **correlation-with-benchmark**: corr of your preds with `ai_model`. This is the
    tell for the "high CORR, high correlation-with-benchmark" trap: a model that just
    re-derives the consensus and is unlikely to earn AIMC once resolved.
  - **stability**: per-exped sharpe (mean/std of the per-exped score) and max drawdown
    of the cumulative score.

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
**Selection metric:** CORR, with correlation-with-benchmark as the differentiation guard  ·  **Payout:** weighted CORR+AIMC+NCORR blend (see `explain_scoring` for live weights and cap)

## Abstract
Two to four sentences: what was tested, the headline result, and the decision
(stake / not yet). Lead with CORR and correlation-with-benchmark (AIMC alongside where resolved).

## Motivation
Why this idea should produce alpha *beyond the consensus*. I.e. why it should lower
correlation-with-benchmark (and so raise AIMC, the payout driver), not just raise CORR.
State the hypothesis you set out to test.

## Method
- Data: train / validation / live exped ranges actually used.
- Feature set and any transforms.
- Model type(s) and key hyperparameters.
- Cross-validation: scheme + embargo (a longer-horizon target needs a wider exped
  embargo; the horizon is a dataset fact, not something to read off the target name).
- How each round differed from the previous (if staged).

## Experiments run
One short subsection per config that *actually ran*. Name the artifacts
(results/preds files). List planned-but-not-run configs separately and clearly.

## Results

| Model | Round | CORR (full) | CORR (recent) | corr_w/_benchmark | AIMC (resolved) | per-exped sharpe | max DD | payout (est) | Status |
|-------|-------|-------------|----------------|-------------------|------------------|------------------|--------|--------------|--------|
| ...   | ...   | ...         | ...            | ...               | ...              | ...              | ...    | ...          | best / kept / dropped |

`payout (est)` is the weighted CORR+AIMC+NCORR blend, before the payout factor and (on
staked events) the per-round return bound. `explain_scoring` reads all of it live, so
don't hardcode an ordering or a cap. Call out any high-CORR / high-corr_w/_benchmark rows
explicitly. Accuracy that differentiates nothing scores well offline and still pays
badly against the crowd once AIMC resolves.

### Round-by-round
For each round: what changed, the best result, and whether it beat the prior best.

### Per-cluster breakdown
Does the edge generalize across the futures clusters, or is it concentrated in one or two?
A cluster-concentrated edge is fragile, say so.

## Standard plot
![cumulative CORR and correlation-with-benchmark vs ai_model](plots/cumulative_corr.png)
Cumulative CORR of the best model, and its rolling correlation with the `ai_model`
benchmark, over expeds. Interpret it: is CORR accumulating steadily, and is
correlation-with-benchmark trending down (more differentiated) or up (converging on
consensus)?

## Decisions
The choices you made and why (feature set, model family, sweep picks, per-exped vs
global). Frame them against correlation-with-benchmark (the differentiation guard), not
just CORR.

## Stopping rationale
Why you stopped iterating, e.g. CORR plateau over N rounds, recent-window
correlation-with-benchmark no longer improving, diminishing payout per round, or a
confirmatory full-data run after a scout phase.

## Findings
What worked, what didn't, what the plot and per-cluster view actually show. Honest
about negative results.

## What we'd stake / why (or not yet)
A clear call in payout terms: would you stake this model, and why, or what specifically
must improve first (e.g. cluster breadth, benchmark de-correlation, resolved-round AIMC
once available).

## Next experiments
2-5 concrete, prioritized follow-ups tied to the findings above.
```

## Step 4: Generate the standard plot

The standard Everesteer plot is **cumulative CORR of the best model, plus its rolling
correlation with the `ai_model` benchmark, over expeds**, built from the run's
out-of-sample predictions.

If the example-scripts repo ships a plotting helper, use it, e.g.:

```bash
python plot_experiment.py \
  --predictions experiments/<name>/predictions/<best_model>.parquet \
  --benchmark ai_model \
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
ax.plot(expeds, bench_cum, color=NAVY, linestyle="--", label="ai_model benchmark")
ax.axhline(0, color=NAVY, linewidth=0.5)
ax.set_xlabel("exped"); ax.set_ylabel("cumulative score / correlation")
ax.set_title("Best model vs ai_model over expeds")
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
- The per-cluster breakdown is present and interpreted.
- The payout framing uses the weighted CORR+AIMC+NCORR blend, per `explain_scoring` (no
  hardcoded ordering or cap number).
- The "what we'd stake / why (or not yet)" conclusion is explicit.
- No synthetic data: all metrics come from real Everesteer predictions and scores.
