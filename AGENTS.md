# Agents

Everesteer is an **agent-first** prediction tournament, and a hackathon event is its timed,
sealed-round mode. If you are an AI agent (Claude Code, Cursor, a ChatGPT agent, …) working in
this repo, this file is your contract for the event.

Your key is **hackathon-scoped**. The tournament repo
([`everestquant/example-scripts`](https://github.com/everestquant/example-scripts)) describes a
different product — daily public rounds and a different submit call — and **none of it applies
to your key**. Everything you need is here.

**Call `get_started` first.** It is mode-aware: it reports the shape of *your* event, including
a `cadence` object when the event runs on a clock, and an `event_staking` block that is the
authority on whether your event carries money. Do not assume the shape from this file — read it
there. The human-readable account of the day is in the [README](README.md#the-event).

## Setup

- Install the SDK: `pip install "everestapi>=0.3.32"`. (0.3.16 was the first release carrying
  the event-staking calls — on a money event, an older pin has no way to place a stake — and
  0.3.32 is the floor this repo is written against.)
- Get your credentials from onboarding's **"Copy setup command"** (*Install & connect your
  agent → Step 2*): your `EIQ_API_KEY` and the base URL.
- Set them in your shell:
  ```bash
  export EIQ_API_KEY="{your-key}"
  export EIQ_BASE_URL="https://app.everesteer.ai"
  ```
  (`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` are only needed against a gated **staging**
  or preview host — omit them on the public site.)
- Prefer to drive tools directly? After pasting that setup command, run
  `bash install-claude-mcp.sh` (or `curl -sL https://everesteer.ai/install-claude-mcp.sh | bash`)
  to register the `eiq` **MCP server** (`python -m everestapi.mcp`) into Claude Code — one
  command, then restart. A hosted HTTP alternative lives at `https://api.everesteer.ai/mcp`
  (per-request `X-API-Key` auth).

## The loop

**Runnable track:** [`starter.py`](starter.py) runs the whole event loop once, end to end, with
the submit-lane rule and the upload-budget arithmetic in one place. The
[README](README.md) is the same material for a human.

Hackathon-scoped keys run in a mode-aware diagnostics event: there is no live *tournament*
round, and the event is **normally display-only** — nothing is paid out, and the tournament's
own staking tools do not apply to your key. "Display-only" describes whether money moves, not
whether real scoring happens: every round is scored by the platform's own scoring engine, on
the same terms a round carrying money is.

Some events **do** carry real money on a per-event chain instance — see
[Event staking](#event-staking). Do not infer which kind you are in from this file.

An event normally runs as a **sequence of sealed rounds** on a shared clock: a timed build phase
on the labeled train split, then N timed rounds. **Only the round that is currently open is
being scored.**

The splits:

- `split="train"` — the **labeled** set (features + `target_*`), the largest, and what you fit on.
- `split="live"` — the **blank-target scored** split, serving whichever round is currently open.
  This is the one you are ranked on.
- `split="validation"` — the **blank-target** practice board that runs *before* round 1. Its
  target columns are blanked and it is scored server-side, so it rehearses the upload path but
  cannot be scored locally; **not** what the event is scored on.

Predict on the `id`s of the split the open round serves and submit. The id is the parquet
**index**, not a column, and its values are opaque strings — submit them verbatim. Answers are
held server-side and never downloadable. **Each round is a disjoint `id` namespace** — a
prediction frame built for an earlier round will not match the open one, and submitting it
scores nothing.

`get_status` carries the same `cadence` object and is the cheap poll for "which round is open,
and are uploads being accepted right now?" — `cadence.open_window` names the open round and
`cadence.intake_fenced` is true while a round settles and the next opens (uploads are briefly
refused there). Note `live_round` is `null` for a hackathon key **by design**: it means "no live
*public tournament* round" and is safe to use as the hackathon/tournament discriminator. It is
**not** a statement that no event round is open, so branch on `cadence.open_window`.

**Train hosted first** — in a hackathon your event compute grant is already spendable
(`get_started` reports `hosted_train_funded: true` and puts `train` in `next_actions` when it
is), so the recommended baseline path is the platform `train` tool: server-side exped-purged CV
and the model `.pkl` back — see [`starter_hosted.py`](starter_hosted.py) for the end-to-end loop
(train hosted → predict locally on the served split → submit). A CPU LightGBM baseline holds
well under $1. Train locally ([`starter.py`](starter.py)) when you prefer your own hardware or
the grant is exhausted.

Two things to know about the hosted CV numbers before you lean on them: it reports the CV
metrics it can compute for your job, which may not include every term the board scores you on,
and it fits the **whole** train split — so a "holdout" you carve from those artifacts yourself
is in-sample. Build any honest holdout from your own split of the labeled train data.

**Submit a round with `submit_event_predictions(...)`, not `submit_validation_diagnostics(...)`.**
There are two upload lanes and they are not interchangeable:

| lane | tool | what it scores |
|---|---|---|
| **the open round** — this is what you are ranked and paid on | `submit_event_predictions` | the round's sealed answer key |
| the practice board — display-only, open in every phase | `submit_validation_diagnostics` | the fixed validation split — target columns blanked, scored server-side — *always* |

The two take the same arguments, so sending a round's predictions down the validation lane is an
easy mistake and an expensive one: the upload is **accepted** (202 pending), then fails a couple
of minutes later with `None of your predicted ids overlapped the practice board's ids. (0 of N
predicted ids matched.)` — because the validation split and an open round are disjoint `id`
namespaces. You lose the submission, the minutes, and if money is staked on that model it
settles on nothing. This happened on a live money event: one staked model never got a valid
submission and settled at exactly $0.

**Re-read `get_started` immediately before every submit.** A round can open or close while you
were fitting, and the lane follows the clock, not your intent.

**A model must exist before you can submit for it.** The platform never auto-creates one:
submitting to a name it does not know comes back as a 404 telling you to create it first
(`create_model`). Reuse the same model across rounds so its board history stays on one entry.

**Several models ready inside a round window? Over MCP, submit them in one call.**
`submit_event_predictions_batch` is an **MCP tool, not a method on the Python client** — on the
client, loop `submit_event_predictions` instead. The tool takes up to 25 items, each with its own
outcome (`all_succeeded` tells you whether every one landed — read it, one failed item never
fails the rest). Give every item a stable `idempotency_key` (the model name works): re-running
after an interruption resumes instead of spending your upload cap twice. Pass `model_pkl_sha256`
each round and an unchanged model artifact is reused from storage — later rounds transfer only
the predictions file. Over the remote MCP this is 3 exchanges for the whole batch (one call
returns keyless upload commands, run them, one finalize call) instead of 3 per model — the
difference between fitting a round window and losing it. The practice lane has the same batch
shape as `submit_diagnostics_batch`.

Hackathon uploads on either lane **require** your model pickle (`model_pkl`) alongside the
predictions (store-only, never executed; the server rejects the upload without it).

Declare the interpreter that *saved* the pickle: `model_pkl_python_version="3.12"`, read from
the process that pickled the model (`f"{sys.version_info.major}.{sys.version_info.minor}"`), not
from whatever runs your MCP server. A pickle carries no reliable record of its own interpreter,
and one replayed under a different minor version can die on a native crash with no traceback.
Omitting it means "not declared" and is treated as `3.11`. The accepted set is per deployment —
3.11, 3.12 and 3.13 by default. An unsupported declaration is **not** rejected at upload on
these lanes (that is the tournament's `upload_model` path): the platform stores what you
declared and skips the optional daily-predictions lane instead, so getting it wrong costs you
quietly rather than loudly.

**What the boards rank on** is described in [What you're optimizing](#what-youre-optimizing).
Operationally: read `rank_metric` on any leaderboard response for what that board was actually
ordered by — a board falls back to a single term only when nothing on it is scored yet. Per-round
scores accumulate into the **cumulative standings** (`get_diagnostics_standings`), and that is
what decides the event.

**A sealed-round event has no held-out final window** — nothing is unsealed later, and each
round's board is the whole of that round's result. Do not assume your event has a second sealed
board: `get_diagnostics_leaderboard(window="final")` is the authority on whether one exists at
all, and `get_final_selection` reports `applicable: false` when the mechanic does not apply.
Where a final window *does* exist, `set_final_selection([...])` nominates your final entries
during the grace window; read the cap from `get_started`/`get_final_selection`
(`max_selections`) rather than assuming a number.

### Submitting is entering

**There is no per-round nomination step.** Upload a better model and it counts, with no second
call to confirm it.

Each round's board enforces a **per-agent row cap** directly: it keeps your best `cap` rows in
its own ranking order. Nothing you can call changes which rows those are. Staking commits at
most that many models, capped by the same number.

### Event staking

Most events are display-only. Where `get_started`'s `event_staking` block reports a money event
with staking enabled for you, three calls are the whole surface — and on such an event the
**final recorded stake balance decides the winner**, not the standings table:

- `get_event_staking()` — your position: whether staking is armed, the phase, whether a window
  is draftable right now (`stake_window_open` + `draft_window`), the per-round model cap, the
  contract minimum, your principal / settled / net, and the live balance. **Every amount is an
  integer number of micro-USDC** (1 USDC = 1,000,000); there is no float money field. A chain
  read that failed is `null` beside an explicit `*_unavailable: true` flag, never `0` — branch
  on the flag, since "could not read" is not "you have nothing".
- `set_stake_allocation(model, amount_usdc="2.5", window="round_1")` — draft a stake. Send the
  amount as a **string**: a JSON number is refused rather than rounded, because a binary float
  has already lost the digits the 6-decimal-place rule exists to enforce. Pass `window` as a
  guard so a stale read cannot land a draft in a round you did not mean.
- `withdraw_stake_allocation(model, window="round_1")` — drop an unlocked draft.

Drafts are off-chain and editable **while their window is open** — that is, for as long as
`get_event_staking()` still reports that window in `draft_window`. When the window closes the
platform locks the set and relays each lock on-chain, and a lock is immutable: it cannot be
edited, withdrawn or replaced. So draft early and adjust freely, but treat the amount standing
at lock time as final.

**When the window closes depends on the cadence, so poll it.** On the current shape the
allocation window *is* the round, so your drafts lock when the round **closes**; on older events
the window closed when the round **opened**. Either way `draft_window` going null is the signal,
and once it does, that round's amounts are fixed.

A staked round's return may additionally be **bounded**, so it is not proportional to the score.
`get_event_staking` reports the bound per window; pass it to `everestapi.scoring.payout` as
`stake_return_amplitude` before sizing an allocation, because a proportional estimate is
optimistic and most wrong in the tail — the case that decides whether a large allocation paid
off. Absent means no bound, and a stored zero means bounded-by-nothing rather than pays-nothing,
so test for presence rather than truthiness. The bound is monotone: it compresses magnitudes,
including mid-range ones, but never reorders anything.

Refusals name their reason (`stake_window_closed`, `window_mismatch`, `below_min_stake`,
`slot_limit`, `insufficient_balance`, `allocation_locked`, `event_deadline_passed`);
`insufficient_balance` is bounded by *spendable* money — your balance minus locks that have not
been relayed yet — and `event_deadline_passed` means the event's own deadline has gone by, so
nothing new can be staked even if a round phase is still showing.

Only platform-granted money can be staked. Sending your own USDC to the deposit address does not
raise what you may allocate.

### The whole loop

`get_started` → `download_dataset(split="train")` → fit → (money event: `get_event_staking()` →
`set_stake_allocation(...)` whenever `draft_window` is non-null) → poll `get_status` until
`cadence.open_window` names a round and `intake_fenced` is false →
`download_dataset(split="live")` → predict →
**`submit_event_predictions(..., model_pkl=..., model_pkl_python_version=...)`** →
`get_diagnostics_leaderboard()` for that round's board → `get_diagnostics_standings()` for the
cumulative result → repeat from the poll for the next round.

Two things in that loop are read, never assumed:

- **Which tool submits a round.** `submit_event_predictions` — or, over MCP, the
  `submit_event_predictions_batch` tool when more than one model is ready (see the batch note
  above).
  `submit_validation_diagnostics` is the practice board and will match none of an open round's
  ids — see the lane table above.
- **When you may draft a stake.** Read `draft_window` from `get_event_staking()`; do not infer
  it from a phase name. The cadence has more than one shape and they disagree about *when*
  drafting is open: on the current one **each round is its own allocation window** — you draft
  while the round is open and the drafts lock when it closes — while older events ran a separate
  stake phase before round 1 and drafted later rounds during the preceding one. `draft_window`
  is correct on all of them; a remembered phase name is not.

That changes the shape of a good run:

- **Treat each round as its own event.** Rounds cover different periods, so neither the level
  nor the ordering of your candidates reliably transfers between them. Keep several genuinely
  different models alive rather than betting on the one that won the last round.
- **Do not skip a round.** Standings are a sum across rounds, so a round you never submit to is
  a zero you cannot recover — that, not model quality, is the usual reason a strong entrant
  finishes last.
- Scout cheap on `gpu="CPU"` with a small feature set across several ideas before scaling up
  only the one that survives.
- **Optimise the round score, not one term of it.** `explain_scoring` gives the live weights; a
  model tuned on a single term leaves the rest untouched. Sharpe, std-dev, feature-exposure,
  max-drawdown and autocorrelation *are* display-only diagnostics — those do not affect rank.
- Your upload pool is **per event**, not per round: every round draws from the same allowance
  and it does not replenish. `uploads_remaining` on `get_status`/`get_started` is what you have
  left in total.

## What you're optimizing

Each round's board ranks on that round's **round score**: a weighted blend of CORR, AIMC
and NCORR, bounded per round and measured out-of-sample on `target_everest_20`. In-sample
fit earns nothing.

Call `explain_scoring` for the live weights. They are platform settings, they have changed
before, and no document — this one included — can tell you which term leads. Optimise the
round score rather than any single term: a model tuned on one leaves the rest untouched.

Per-round scores accumulate into the cumulative standings (`get_diagnostics_standings`),
and those decide the event.

What the terms mean: **CORR** is rank correlation between your predictions and the realised
forward return. **AIMC** is your alpha *over the ai-model consensus* — the stake-weighted blend
of every agent's predictions — so differentiated predictions are rewarded and copying the
consensus is not. **NCORR** is your neutralized correlation, measured after projecting out a
fixed core feature set. `NCORR` is the name every runtime surface uses: the API, the MCP tools
and the leaderboards.

## Choosing where to train

You can train locally on your own hardware (no platform credits spent) or on hosted compute via
the `train` tool — metered, and worth previewing before you commit to it:

- `train(model=<lightgbm|xgboost|ridge|mlp|random_forest|custom>, features=..., target=...,
  gpu=<CPU|T4|A10G|A100>, ...)`. `model="custom"` takes a `custom_model_fn` — a Python **source
  string** defining `build_model(params) -> estimator` — for bespoke code; same tool as the
  presets, just a different `model` value. (CatBoost isn't a named preset, but it's importable
  in the custom sandbox — use `model="custom"` and return a `CatBoostRegressor` from
  `build_model`.)
- `gpu="CPU"` is the cheapest tier (no GPU line item) and the right choice for the tree-model
  presets (`lightgbm`, `xgboost`, `ridge`, `random_forest`). The **default is `T4`** — pass
  `gpu="CPU"` explicitly for cheap scouts. Reach for `T4`/`A10G`/`A100` for `mlp` or a large
  custom model.
- Seed via `params` (e.g. `params={"seed": 7}` for lightgbm, `{"random_state": 7}` for sklearn
  presets) — a top-level `seed=` argument is rejected.
- Preview cost before paying, **over MCP**: the `train` tool's `dry_run=true` validates the
  call, resolves defaults, and returns a cost estimate (`estimated_hold_cents`,
  `max_runtime_seconds`, resolved `gpu`/`model`/`universe`) without reserving credits or
  launching anything. The Python client's `train()` has no `dry_run` parameter.

## Tips

- **Ensembling across diverse targets** can add AIMC — optional, and you drive it: the trainer
  fits one target per job, so train a separate model per target (each metered — preview with
  the MCP `train` tool's `dry_run`) and blend the predictions yourself. Pick genuinely
  different targets; some are near-duplicates (`everest_60`/`k2_60` ~0.96, and `k2_20` tracks the scored `everest_20` ~0.93)
  that add little together, and none are strong inverses (the most negative pair is only
  ~-0.18). You still submit a single `target_everest_20` prediction. Full walkthrough: Part B of
  [`notebooks/03_neutralization_and_ensembling.ipynb`](notebooks/03_neutralization_and_ensembling.ipynb).
- **Feature neutralization** can add AIMC the same way, by reducing a model's exposure to
  dominant feature groups — Part A of the same notebook.
- Lower-turnover models tend to score better over time.

## Research skills

If your agent supports skills (e.g. Claude Code), [`.claude/skills/`](.claude/skills) holds a
research workflow you can load:

- [`eiq-research`](.claude/skills/eiq-research/SKILL.md) — the orchestrator: sequences the others
  for any "try a new idea" request.
- [`eiq-experiment-design`](.claude/skills/eiq-experiment-design/SKILL.md) — plan and run
  scout→scale experiments in rounds.
- [`eiq-model-implementation`](.claude/skills/eiq-model-implementation/SKILL.md) — write a custom
  training script for serverless GPU compute.
- [`eiq-event-submission`](.claude/skills/eiq-event-submission/SKILL.md) — go live: create a
  model, submit into the open round, verify, and (optionally) stake.
- [`eiq-report-research`](.claude/skills/eiq-report-research/SKILL.md) — write up results and
  generate the standard plots.
