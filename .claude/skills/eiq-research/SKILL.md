---
name: eiq-research
description: >
  Top-level orchestrator for Everesteer hackathon-event (futures dataset) research.
  Drives a new idea from hypothesis to a submitted model by sequencing four sibling
  skills: eiq-experiment-design, eiq-model-implementation, eiq-event-submission, and
  eiq-report-research. Reach for this whenever the request is "try/test a new idea",
  "run an experiment", "sweep configs", "compare models", "improve my Everesteer scores",
  or any open-ended "do Everesteer event research" task. It enforces scout-then-scale
  discipline, benchmarks against the Everesteer ai_model, selects experiments on CORR
  plus correlation-with-benchmark (the offline read on likely AIMC once a round
  resolves), and treats per-exped stability as a diagnostic.
---

# Everesteer Research Orchestrator

You are running research for an Everesteer hackathon event, scored against Everesteer's
live agent-native *futures* dataset. This skill does not do the work itself; it routes
you through the four sibling skills in the right order and holds the connective tissue —
defaults, gates, and handoffs — so a loose "try this idea" request lands as a submitted,
documented model.

You only have the **participant surface**: the `everestapi` SDK (`pip install "everestapi>=0.3.32"`),
the Everesteer MCP server (`python -m everestapi.mcp`, tools named `eiq_*` — the
`mcp__<server>__` prefix depends on how your client registered the server), datasets
you download, and the helpers shipped in this repo. There is no internal platform source
to read.

Your key is hackathon-scoped: there are no live *tournament* rounds. Your event runs its
own sealed rounds, entered through `submit_event_predictions` (over MCP, the
`submit_event_predictions_batch` tool submits several at once — there is no batch method
on the Python client), read on `get_diagnostics_leaderboard`.
`submit_validation_diagnostics` is the display-only practice board and matches none of an
open round's ids. See the repo-root `AGENTS.md` for the full event mechanics.

## The shape of every research run

```
  orient  →  design  →  (implement?)  →  scout  →  scale  →  report  →  submit
            \_______ eiq-experiment-design ______/         \_ report _/ \ submit /
                       \_ eiq-model-implementation (only if new code) _/
```

Skip nothing silently. If you skip a stage, say why.

---

## Stage 0 — Orient (do this yourself, fast)

Pull the current state before committing compute. A few MCP calls:

- `get_started` / `get_status` — where you are in the event's cadence
  (`cadence.open_window`, `phase_ends_at`, `intake_fenced`), and your remaining
  `uploads_remaining`.
- `get_universe` + `get_features` — which chains/clusters and which `feature_<theme>_<n>`
  columns exist this round. Features are encoded into integer bins whose count and
  missing sentinel the schema declares (`feature_encoding`);
  do not try to attach economic meaning.
- `get_dataset_schema` — confirm the graded target column (`primary_target`) and
  the split layout.
- `get_models` + `get_diagnostics_leaderboard` + `get_diagnostics_standings` — what you
  already have running and how it ranks.
- `get_compute_credits` — your budget ceiling for the whole run.

Write a two-line read of the situation: what's the bottleneck — raw accuracy (low CORR)
or differentiation (CORR fine, correlation-with-benchmark high / AIMC flat)? That framing
drives the design stage.

---

## Stage 1 — Design  →  hand to `eiq-experiment-design`

Translate the idea into a concrete plan. Invoke **`eiq-experiment-design`**, which is
responsible for:

- pinning the hypothesis and the one metric that decides win/lose (default: **CORR** as
  the experiment-selection metric, with correlation-with-benchmark as the differentiation
  guard and per-exped stability alongside),
- choosing the feature scope and CV (exped-purged + embargoed — never plain k-fold),
- laying out **rounds of ~4-5 configs**, scout-sized first,
- defining the plateau rule and the scale trigger.

If the idea is vague, let the design skill run a couple of cheap scout probes to
disambiguate rather than guessing. Do not start firing `train` jobs from here —
that is the design skill's job to specify and the run stages' job to execute.

---

## Stage 2 — Implement *only if the idea needs new code*  →  `eiq-model-implementation`

Most ideas ride existing model families (`train(model=<preset>)` covers templated
LightGBM and friends; `train(model="custom", custom_model_fn=...)` runs an arbitrary
GPU script — same tool, different `model` value). Reach for
**`eiq-model-implementation`** only when the hypothesis genuinely requires a new
estimator, a custom target transform, or bespoke `fit`/`predict` behaviour. That skill
writes the training script, wires it for `train(model="custom", ...)`, and proves it with
a smoke run (a non-degenerate CORR on a tiny sample) before you spend real credits.

If you are reusing what exists, state "no new code needed" and move on.

---

## Stage 3 — Scout (cheap, wide)

Execute round one exactly as the design specified.

- Downsample: run on a **subset of expeds**, not the full history. The point is to rank
  configs cheaply, not to measure final performance.
- Prefer `train(model=<preset>, gpu="CPU")` for templated configs; reserve
  `train(model="custom", ...)` for the variants that need it. Over MCP, the `train`
  tool's `dry_run=true` previews cost before you launch (the Python client's `train()`
  has no `dry_run` parameter). Poll with `get_job_status`; pull artifacts with
  `get_model_download_url`.
- Score every config the same way: download the benchmark (`download_benchmark` /
  `get_benchmarks`) and evaluate predictions against the Everesteer **`ai_model`**. Compute
  CORR, correlation-with-benchmark, and (where rounds have resolved) AIMC, and use
  `run_validation_diagnostics` for a sanity pass.
- Rank candidates on the **round score**, not on any single term. Which terms carry
  weight, and how much, is a live platform setting — call `explain_scoring` and rank on
  what it reports. A ranking built on one term leaves the rest of the score untouched.
  Use **correlation-with-benchmark** — the offline read on likely AIMC — as the
  differentiation guard beneath it, with per-exped stability as the robustness
  check. Promote the top ~2 configs.

**Gate before you scale:** if no config clears the design's correlation-with-benchmark
bar, do not scale. A config with healthy CORR but correlation-with-benchmark near 1.0 is
shadowing the static benchmark (and, live, the ai-model) — change the idea (new feature
angle, a benchmark-residualized target to lower correlation with the benchmark and
ai-model, which is what AIMC pays for) rather than throwing more data at it.

---

## Stage 4 — Scale (only the survivors)

Re-run the promoted configs at full exped coverage and a wider feature scope, plus a
couple of tight variations. Confirm CORR holds up out of sample and that the lift was not
a small-sample artifact. Re-check on the held-out split before declaring a winner.

**Stop** when any of these is true:

- two straight rounds add no meaningful CORR (you have plateaued),
- correlation-with-benchmark is clearly below the design's bar *and* CORR is real but not
  suspiciously high (very high CORR on training expeds usually means overfit, not skill);
  per-exped stability corroborates,
- remaining compute credits no longer justify another round.

Track everything as you go — one row per config per round (CORR / correlation-with-benchmark
/ AIMC / notes). A per-exped breakdown computed from your own out-of-sample predictions is
useful for spotting a config that wins only in one regime.

---

## Stage 5 — Report  →  hand to `eiq-report-research`

Once you have a winner (or a clean negative result), invoke **`eiq-report-research`** to
produce the writeup: the hypothesis, what was tried each round, the metric table with
CORR leading (correlation-with-benchmark alongside, AIMC where rounds have resolved), the
deciding plots, why the winner won, and what to try next. A negative
result is still worth reporting — it stops the next run from repeating it.

---

## Stage 6 — Submit  →  hand to `eiq-event-submission`

Submit **only with the user's explicit go-ahead**, and only if the winner clears the bar
the design stage set (meaningful correlation-with-benchmark differentiation and a real,
non-overfit CORR). Then invoke
**`eiq-event-submission`**, which owns:

- `create_model` (if this is a new model name),
- prediction formatting and the **`submit_event_predictions`** call into the open round
  (or `submit_validation_diagnostics` for the practice board when none is open),
- post-submit confirmation via the round's board (`get_diagnostics_leaderboard`),
- and, if the user asks, the event-staking follow-through (`get_event_staking`,
  `set_stake_allocation`).

---

## Defaults and principles

- **CORR is the primary experiment-selection lever.** Accuracy is what both live surfaces
  reward: call `explain_scoring` for the live weights and rank on those. No document,
  this one included, can tell you which term leads — the weights are settings and they
  have changed. Rank
  configs by CORR first; per-exped stability is the robustness check;
  correlation-with-benchmark, next, is the differentiation guard, not the objective.
- **AIMC is the differentiation term.** **AIMC** (AI Model Contribution) is your unique
  signal beyond the *live* stake-weighted ai-model consensus, measured once a round
  resolves. You cannot compute it offline — the best available offline signal is
  **correlation-with-benchmark** (corr of your predictions vs the static `ai_model`
  benchmark you downloaded): lower means more differentiated, and more likely to earn
  AIMC once the round resolves. Never pay CORR to buy differentiation, and never invent
  a precise offline AIMC number.
- **Payout is a weighted blend of CORR, AIMC, and NCORR — call `explain_scoring` for the
  live weights and cap.** Don't hardcode an ordering; it has changed before. Uniqueness
  pays more than raw accuracy — keep the search pointed at differentiated alpha, not at
  chasing CORR. That score is then scaled by a per-round **payout factor**, frozen at the
  round's stake lock: 1 below a fixed total-stake threshold, shrinking above it, so it can
  differ round to round.
- **Scout before you scale.** Always a downsampled-exped round first; full data only for
  survivors.
- **Iterate in rounds and stop at a plateau.** ~4-5 configs per round; two flat rounds
  means the idea is spent.
- **Benchmark against `ai_model` every time.** It is the comparison baseline;
  residualizing/neutralizing against it lowers correlation-with-benchmark, which is what
  drives AIMC once a round resolves. No correlation-with-benchmark or AIMC number is
  meaningful without the downloaded benchmark to compare against.
- **Respect the CV.** Exped-purged with embargo. The 20-day target needs a wide enough
  embargo to avoid look-ahead — let the design skill set it; never substitute plain k-fold.
- **Compute is metered.** Check `get_compute_credits` up front and let the budget bound
  the number of rounds.
- **Real data only.** Everything comes from the downloaded Everesteer datasets and the SDK/MCP —
  never fabricate features, targets, or scores.

## Worked example

> User: "Try training on benchmark-residualized targets — I think we're just echoing the ai_model."

1. **Orient** — `get_diagnostics_standings` + the round board show solid CORR,
   correlation-with-benchmark near 1.0. Bottleneck is differentiation, exactly the
   user's hunch.
2. **Design** (`eiq-experiment-design`) — three-round plan; dimension under test is the
   target transform (the raw graded target vs benchmark-residualized); a
   correlation-with-benchmark bar is set.
3. **Implement** — none; the residualized-target path is templated. State so and skip.
4. **Scout** — ~5 configs on a 20%-exped sample. Raw-target configs land decent CORR but
   correlation-with-benchmark near 1.0 (static-benchmark echo). One residualized config
   clears the bar → promote it.
5. **Scale** — full expeds + wider features; correlation-with-benchmark stays low and
   CORR stays sane (per-exped stability corroborates) → winner.
6. **Report** (`eiq-report-research`) — table with CORR leading, residualization plot,
   rationale.
7. **Submit** (`eiq-event-submission`) — on the user's OK, `create_model` if needed,
   then `submit_event_predictions` into the open round; confirm via the round's board.

<!-- NOTE: the exact correlation-with-benchmark bar and embargo width are owned by
     eiq-experiment-design; this orchestrator intentionally does not hardcode them. -->
