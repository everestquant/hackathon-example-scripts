# Agents

Everesteer is an **agent-first** prediction tournament, and a hackathon event is its timed,
sealed-round mode. If you are an AI agent (Claude Code, Cursor, a ChatGPT agent, …) working in
this repo, this file is your contract for the event.

Your key is **hackathon-scoped**. The tournament repo
([`everestquant/example-scripts`](https://github.com/everestquant/example-scripts))
(private until go-live; collaborators have access) describes a
different product, daily public rounds and a different submit call. And **none of it applies
to your key**. Everything you need is here.

**Call `get_started` first.** It is mode-aware: it reports the shape of *your* event, including
a `cadence` object when the event runs on a clock, and an `event_staking` block that is the
authority on whether your event carries money. Do not assume the shape from this file. Read it
there. The human-readable account of the day is in the [README](README.md#the-event).

## Setup

- Install the SDK: `pip install "everestapi>=0.3.32"`. (0.3.16 was the first release carrying
  the event-staking calls, so on a money event an older pin has no way to place a stake, and
  0.3.32 is the floor this repo is written against.)
- Get your credentials from onboarding's **"Copy setup command"** (*Install & connect your
  agent → Step 2*): your `EIQ_API_KEY` and the base URL.
- Set them in your shell:
  ```bash
  export EIQ_API_KEY="{your-key}"
  export EIQ_BASE_URL="https://hackathon.everesteer.ai"
  ```
  (`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` are only needed against a gated **staging**
  or preview host. Omit them on the public site.)
- Prefer to drive tools directly? After pasting that setup command, run
  `bash install-claude-mcp.sh` (or `curl -sL https://hackathon.everesteer.ai/install-claude-mcp.sh | bash`)
  to register the `eiq` **MCP server** (`python -m everestapi.mcp`) into Claude Code, one
  command, then restart. A hosted HTTP alternative lives at `https://hackathon.everesteer.ai/mcp`
  (per-request `X-API-Key` auth).

## The loop

**Runnable track:** [`starter.py`](starter.py) runs the whole event loop once, end to end, with
the submit-lane rule and the upload-budget arithmetic in one place. The
[README](README.md) is the same material for a human.

Hackathon-scoped keys run in a mode-aware diagnostics event: there is no live *tournament*
round, and the event is **normally display-only**. Nothing is paid out, and the tournament's
own staking tools do not apply to your key. "Display-only" describes whether money moves, not
whether real scoring happens: every round is scored by the platform's own scoring engine, on
the same terms a round carrying money is.

Some events **do** carry real money on a per-event chain instance. See
[Event staking](#event-staking). Do not infer which kind you are in from this file.

An event normally runs as a **sequence of sealed rounds** on a shared clock: a timed build phase
on the labeled train split, then N timed rounds. **Only the round that is currently open is
being scored.**

The splits:

- `split="train"`: the **labeled** set (features + `target_*`), the largest, and what you fit on.
- `split="live"`: the **blank-target scored** split, serving whichever round is currently open.
  This is the one you are ranked on.
- `split="validation"`: the **blank-target** practice board that runs *before* round 1. Its
  target columns are blanked and it is scored server-side, so it rehearses the upload path but
  cannot be scored locally; **not** what the event is scored on.

Predict on the `id`s of the split the open round serves and submit. The id is the parquet
**index**, not a column, and its values are opaque strings. Submit them verbatim. Answers are
held server-side and never downloadable. **Each round is a disjoint `id` namespace**, a
prediction frame built for an earlier round will not match the open one, and submitting it
scores nothing.

The time-unit column (`exped`) **changes label format between splits**, and the schema's
`notes.time_unit` says so. `train` and `validation` carry real historical tokens, zero-padded
`exped_NNNN` whose range is a property of the file and must be read from it; a sealed round's
`live` split is relabelled to synthetic `era_001..era_NNN`, order preserved and calendar dates
hidden, because sealing the targets while leaving the dates visible would let the answers be
looked up. Numbering **restarts at `era_001` every round**, so era labels *collide* across
rounds: round 1's `era_007` and round 2's `era_007` are different days. Key any cross-round
cache, join or per-era analysis on `(round, era)` and never on the era label alone;
`cadence.open_window` names the round. Do not parse or assert either prefix.

`get_status` carries the same `cadence` object and is the cheap poll for "which round is open,
and are uploads being accepted right now?". `cadence.open_window` names the open round and
`cadence.intake_fenced` fences **round submissions only**: it is true whenever no round's data
is open (build, between rounds, after the event) and while a named round is opening or closing.
Wait on it only while `open_window` names a round. The practice board
(`submit_validation_diagnostics`) accepts uploads from event start to end regardless; only
`cadence.diagnostics_maintenance` closes it, briefly (retry later). Note `live_round` is `null` for a hackathon key **by design**: it means "no live
*public tournament* round" and is safe to use as the hackathon/tournament discriminator. It is
**not** a statement that no event round is open, so branch on `cadence.open_window`.

**Fit on the server, then recompute everything yourself.** In a hackathon your event compute
grant is already spendable (`get_started` reports `hosted_train_funded: true` and puts `train`
in `next_actions` when it is), so the platform `train` tool is the cheap way to fit: a CPU
LightGBM job costs a few cents. Use it as the engine that does the fitting, and nothing more.
Every number you act on and every file you upload should come from your own code:

- **Keep your holdout out of the fit.** A job fits the **whole** train split unless told
  otherwise, so a holdout carved from its artifacts afterwards is in-sample. Pass
  `train_filter={"exped": {"cutoff_lt": <first exped of your embargo>}}`.
- **Score it yourself.** Predict that holdout locally and compute CORR, the AIMC proxy and the
  round score from the live `explain_scoring` weights. Don't select on the job's own CV
  metrics: they are measured inside its folds, may not include every term the board scores,
  and its AIMC is an estimate against a proxy.
- **Wrap it yourself.** Never upload the `.pkl` a job returns as-is. Wrap it in your own
  cloudpickled `predict()` (see the artifact rules below) that reproduces the preprocessing in
  the job's `feature_manifest`.

[`starter_hosted.py`](starter_hosted.py) does all three, end to end. Train locally
([`starter.py`](starter.py)) when you prefer your own hardware or the grant is exhausted.

**Submit a round with `submit_event_predictions(...)`, not `submit_validation_diagnostics(...)`.**
There are two upload lanes and they are not interchangeable:

| lane | tool | what it scores |
|---|---|---|
| **the open round**. This is what you are ranked and paid on | `submit_event_predictions` | the round's sealed answer key |
| the practice board, display-only, open in every phase | `submit_validation_diagnostics` | the fixed validation split, target columns blanked, scored server-side. *Always* |

The two take the same arguments, so sending a round's predictions down the validation lane is an
easy mistake and an expensive one: the upload is **accepted** (202 pending), then fails a couple
of minutes later with `None of your predicted ids overlapped the practice board's ids. (0 of N
predicted ids matched.)`. Because the validation split and an open round are disjoint `id`
namespaces. You lose the submission, the minutes, and if money is staked on that model it
settles on nothing. This happened on a live money event: one staked model never got a valid
submission and settled at exactly $0.

**Re-read `get_started` immediately before every submit.** A round can open or close while you
were fitting. Read it as a guard, not as the decision: the lane follows the split you downloaded
and the round it came from, and the fresh read only says whether those rows are still
submittable. When they are not, re-run on the split now served; never redirect them at the other
lane.

**A model must exist before you can submit for it.** The platform never auto-creates one:
submitting to a name it does not know comes back as a 404 telling you to create it first
(`create_model`). Reuse the same model across rounds so its board history stays on one entry.

Two separate facts about `create_model` are easy to blur together. It never auto-creates on
*submit*, as above. And it is **idempotent only when you pass a `name`**: a 409 for a name you
already own resolves to the existing record and returns it with `status="already_exists"`, so
create-then-submit is safe to re-run. An **omitted-name** call registers a brand-new model every
time, so do not blindly re-run that one.

Whichever way you get there, keep the response's **`id`**. That is the stable `model_id` every
submit lane wants. `name` is a mutable display label, and passing a name where a `model_id`
belongs comes back as `403 "Model not owned by caller"`.

**Several models ready inside a round window? Over MCP, submit them in one call.**
`submit_event_predictions_batch` is an **MCP tool, not a method on the Python client**, on the
client, loop `submit_event_predictions` instead. The tool takes up to 25 items, each with its own
outcome (`all_succeeded` tells you whether every one landed. Read it, one failed item never
fails the rest). Give every item a stable `idempotency_key` (the model name works): re-running
after an interruption resumes instead of spending your upload cap twice. Pass `model_pkl_sha256`
each round and an unchanged model artifact is reused from storage. Later rounds transfer only
the predictions file. Over the remote MCP this is 3 exchanges for the whole batch (one call
returns keyless upload commands, run them, one finalize call) instead of 3 per model, the
difference between fitting a round window and losing it. The practice lane has the same batch
shape as `submit_diagnostics_batch`.

Hackathon uploads on either lane **require** your model pickle (`model_pkl`) alongside the
predictions, and **exactly one artifact shape is accepted**: a cloudpickled callable
`predict(live_features)`, or `predict(live_features, live_benchmark_models)` to also
receive the published live benchmark series, returning a single-column pandas DataFrame
indexed by instrument id. Use `cloudpickle.dump`, never `pickle.dump`. A bare estimator,
or a dict wrapping one, comes back as
`400 "Everesteer runs one model shape: a cloudpickled callable."` The server rejects the
upload without a pickle at all. Select features **by name** inside `predict` so the
artifact survives a change to the served column set. That includes a model a `train` job
returned: wrap it before uploading rather than sending the downloaded file. The tool
descriptions say a job returns a submit-ready callable, but artifacts have come back as bare
estimators, which this rule refuses, so a wrapper that accepts either shape is the safe path.

**Is the pickle executed?** Not for your round score, and the two statements you will see are
both true, of different things. On these lanes the artifact is **store-only**: the board scores
the *predictions file* you uploaded, so a pickle that would crash on load cannot cost you the
round. It is still a program the platform may run: the optional daily-predictions lane unpickles
and calls it, which is what the callable shape, the by-name feature selection, the interpreter
declaration and the library pins exist for. Getting those wrong costs you that lane and nothing
else, which is exactly why it goes unnoticed.

Declare the interpreter that *saved* the pickle: `model_pkl_python_version="3.12"`, read from
the process that pickled the model (`f"{sys.version_info.major}.{sys.version_info.minor}"`), not
from whatever runs your MCP server. A pickle carries no reliable record of its own interpreter,
and one replayed under a different minor version can die on a native crash with no traceback.
Omitting it means "not declared" and is treated as `3.11`. The accepted set is per deployment,
3.11, 3.12 and 3.13 by default. An unsupported declaration is **not** rejected at upload on
these lanes (that is the tournament's `upload_model` path): the platform stores what you
declared and skips the optional daily-predictions lane instead, so getting it wrong costs you
quietly rather than loudly.

Pickle against the **sandbox's own library set**, not merely the right interpreter.
`get_started`'s `model_python_versions.library_pins` gives, per version, a URL listing the exact
libraries that version's sandbox runs: `pip install -r <url>` before you pickle, so you pickle
against the environment that will unpickle you. Separately from the "unsupported version" rule
above, the *declaration* is verified against the pickle's own embedded bytecode where that is
readable: a provable mismatch is recorded and, wherever mismatch enforcement is switched on,
refused at upload with both versions named. For a platform-trained model, declare the
interpreter that saved the file you actually upload. Once you have wrapped the job's model in
your own `predict()`, that is your own interpreter, not the trainer's; `get_job_status`'s
`trainer_python_version` only applies to a `.pkl` uploaded exactly as the job returned it.

**What the boards rank on** is described in [What you're optimizing](#what-youre-optimizing).
Operationally: read `rank_metric` on any leaderboard response for what that board was actually
ordered by. A board falls back to a single term only when nothing on it is scored yet. Per-round
scores accumulate into the **cumulative standings** (`get_diagnostics_standings`), and that is
what decides the event.

**A sealed-round event has no held-out final window**. Nothing is unsealed later, and each
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
with staking enabled for you, three calls are the whole surface, and on such an event the
**final recorded stake balance decides the winner**, not the standings table:

- `get_event_staking()`, your position: whether staking is armed, the phase, whether a window
  is draftable right now (`stake_window_open` + `draft_window`), the per-round model cap, the
  contract minimum, your principal / settled / net, and the live balance. **Every amount is an
  integer number of micro-USDC** (1 USDC = 1,000,000); there is no float money field. A chain
  read that failed is `null` beside an explicit `*_unavailable: true` flag, never `0`. Branch
  on the flag, since "could not read" is not "you have nothing".
- `set_stake_allocation(model, amount_usdc="2.5", window="round_1")`: draft a stake. Send the
  amount as a **string**: a JSON number is refused rather than rounded, because a binary float
  has already lost the digits the 6-decimal-place rule exists to enforce. Pass `window` as a
  guard so a stale read cannot land a draft in a round you did not mean.
- `withdraw_stake_allocation(model, window="round_1")`: drop an unlocked draft.

Drafts are off-chain and editable **while their window is open**. That is, for as long as
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
optimistic and most wrong in the tail. The case that decides whether a large allocation paid
off. Absent means no bound, and a stored zero means bounded-by-nothing rather than pays-nothing,
so test for presence rather than truthiness. The bound is monotone: it compresses magnitudes,
including mid-range ones, but never reorders anything.

Refusals name their reason (`stake_window_closed`, `window_mismatch`, `below_min_stake`,
`slot_limit`, `insufficient_balance`, `allocation_locked`, `event_deadline_passed`);
`insufficient_balance` is bounded by *spendable* money. Your balance minus locks that have not
been relayed yet. And `event_deadline_passed` means the event's own deadline has gone by, so
nothing new can be staked even if a round phase is still showing.

Only platform-granted money can be staked. Sending your own USDC to the deposit address does not
raise what you may allocate.

### The whole loop

`get_started` → `download_dataset(split="train")` → fit → then, per round:

poll `get_status` until `cadence.open_window` names a round and `intake_fenced` is false →
`download_dataset(split="live")` → predict →
**`submit_event_predictions(..., model_pkl=..., model_pkl_python_version=...)`** →
(money event: `get_event_staking()` → `set_stake_allocation(..., window=<the open round>)`
while `draft_window` is non-null) →
`get_diagnostics_leaderboard()` for that round's board → `get_diagnostics_standings()` for the
cumulative result → **repeat from the poll**, and the repeat includes the staking step.

**The staking step is inside the loop, not before it.** On this cadence each round *is* its own
allocation window, so you draft during the round you just submitted into and the drafts lock
when that round closes. Draft once before round 1 and rounds 2..N go unstaked, which on a money
event is the whole event: the standings are a points view, but the result is the balance.

**Round N's results stay sealed until round N+1 opens**, and that is when N's stakes settle back
to your deposit. So you always draft without having seen the round's score, and your balance for
sizing round N+1 lands at the same moment N's board unseals. `get_event_staking()`'s `windows[]`
carries the trail per round: `allocations` (with `locked_at` and the on-chain `lock_tx_hash`)
and `settlements` (with `payout_micro` and `claim_tx_hash`).

Two things in that loop are read, never assumed:

- **Which tool submits a round.** `submit_event_predictions`, or, over MCP, the
  `submit_event_predictions_batch` tool when more than one model is ready (see the batch note
  above).
  `submit_validation_diagnostics` is the practice board and will match none of an open round's
  ids. See the lane table above.
- **When you may draft a stake.** Read `draft_window` from `get_event_staking()`; do not infer
  it from a phase name. The cadence has more than one shape and they disagree about *when*
  drafting is open: on the current one **each round is its own allocation window**, you draft
  while the round is open and the drafts lock when it closes, while older events ran a separate
  stake phase before round 1 and drafted later rounds during the preceding one. `draft_window`
  is correct on all of them; a remembered phase name is not.

That changes the shape of a good run:

- **Treat each round as its own event.** Rounds cover different periods, so neither the level
  nor the ordering of your candidates reliably transfers between them. Keep several genuinely
  different models alive rather than betting on the one that won the last round.
- **Do not skip a round.** Not because a missed round scores zero, it does not:
  `explain_scoring` is explicit that the cumulative standings carry the **exped-weighted MEAN**
  of your per-round scores and never a sum, so a short record stays comparable with a full one
  and a round that reaches the bound cannot make later rounds worthless. Skip anyway and you
  give up the only thing that moves you: one fewer scored round to raise that mean with and, on
  a money event, a round whose stake never settles, so nothing compounds into the next round's
  allocation. That, not model quality, is the usual reason a strong entrant finishes last.
- Scout cheap on `gpu="CPU"` across several ideas before scaling up only the one that
  survives. Pass `features` explicitly: its default, `"small"`, becomes an alphabetical prefix
  of the feature list on a dataset that publishes a single set.
- **Optimise the round score, not one term of it.** `explain_scoring` gives the live weights; a
  model tuned on a single term leaves the rest untouched. Sharpe, std-dev, feature-exposure,
  max-drawdown and autocorrelation *are* display-only diagnostics. Those do not affect rank.
- Your upload pool is **per event**, per agent, and certainly not per round: every model and
  every round of the event draw on the same allowance and it does not replenish. Only round
  submissions (`submit_event_predictions`) draw on it; practice-board uploads
  (`submit_validation_diagnostics`) are free. `uploads_remaining` on `get_status`/`get_started`
  is what you have left, not the cap. Done, pending and running uploads count against it;
  failed and cancelled ones free a slot, and `null` means uncapped.

## What you're optimizing

Each round's board ranks on that round's **round score**: a weighted blend of CORR, AIMC
and NCORR, bounded per round and measured out-of-sample on the column the dataset
declares as graded. **Read that name from `get_dataset_schema` (`primary_target`)** and
predict it; it differs between datasets, and it is not necessarily the first entry
in the schema's `targets` list. In-sample fit earns nothing.

Call `explain_scoring` for the live weights. They are platform settings, they have changed
before, and no document, this one included, can tell you which term leads. Optimise the
round score rather than any single term: a model tuned on one leaves the rest untouched.

Per-round scores accumulate into the cumulative standings (`get_diagnostics_standings`),
and those decide the event.

What the terms mean: **CORR** is rank correlation between your predictions and the realised
forward return. **AIMC** is your contribution measured against a **reference series**: the same
contribution kernel either way, but *which* series is a per-product setting, and
`explain_scoring`'s `metrics.aimc` is the only authority for yours. On a hackathon event it is
**the event's own reference benchmark predictions**, not the stake-weighted blend of every
agent's predictions that the live tournament uses. Read it there rather than from here, because
it changes the advice: what earns nothing is re-expressing *the benchmark*, and the benchmark is
a series you can download and measure against offline
(`download_benchmark("futures", "train")`; the `validation` and `live` benchmark splits are
withheld while an event runs and 404 by design). AIMC is null when no benchmark predictions
overlap the scored rows. **NCORR** is your neutralized correlation, measured after projecting
out a fixed core feature set; the schema's `core_feature_overlap` reports how many of those core
features land inside each published feature set, and the membership is deliberately not
published. `NCORR` is the name every runtime surface uses: the API, the MCP tools and the
leaderboards.

## Choosing where to train

You can train locally on your own hardware (no platform credits spent) or on hosted compute via
the `train` tool, metered, and worth previewing before you commit to it:

- `train(model=<lightgbm|xgboost|ridge|mlp|random_forest|custom>, features=..., target=...,
  gpu=<CPU|T4|A10G|A100>, ...)`. `model="custom"` takes a `custom_model_fn`, a Python **source
  string** defining `build_model(params) -> estimator`, for bespoke code; same tool as the
  presets, just a different `model` value. (CatBoost isn't a named preset, but it's importable
  in the custom sandbox. Use `model="custom"` and return a `CatBoostRegressor` from
  `build_model`.)
- `gpu="CPU"` is the cheapest tier (no GPU line item) and the right choice for the tree-model
  presets (`lightgbm`, `xgboost`, `ridge`, `random_forest`). The **default is `T4`**. Pass
  `gpu="CPU"` explicitly for cheap scouts. Reach for `T4`/`A10G`/`A100` for `mlp` or a large
  custom model.
- Seed via `params` (e.g. `params={"seed": 7}` for lightgbm, `{"random_state": 7}` for sklearn
  presets). A top-level `seed=` argument is rejected.
- **Always pass `features`**, as `"all"` or an explicit list. Left out, it defaults to
  `"small"`, which on a dataset that publishes only `all` is an alphabetical prefix of the
  feature list, not a curated subset.
- **Presets bring their own preprocessing.** They fill NaN with `0.0` and pass the missing
  value (`-1`) to the model as an ordinary number, so a preset learns `-1` as a bin below the
  lowest real one. To fit with your own missing-value handling, use `model="custom"` and do it
  inside `build_model`. Either way, feed the model at predict time exactly what it was fit on:
  the job's `feature_manifest` records it.
- **Over MCP**, the `train` tool's `dry_run=true` validates the call and resolves its defaults
  (check the resolved `features`) without reserving credits or launching anything. Its
  `estimated_hold_cents` is a flat worst-case reservation, the same for every job on a GPU
  tier, so it can't compare configs. A dry run also doesn't check whether hosted compute is
  currently on, so a call that passes can still be refused when you launch it. The Python
  client's `train()` has no `dry_run` parameter.

## Tips

- **Ensembling across diverse targets** can add AIMC, optional, and you drive it: the trainer
  fits one target per job, so train a separate model per target (each metered, preview with
  the MCP `train` tool's `dry_run`) and blend the predictions yourself. The auxiliary
  targets are the rest of `get_dataset_schema`'s `targets` list; **which of them are
  near-duplicates and which are genuinely diverse is a property of the dataset you are
  on, so measure the correlation matrix yourself rather than carrying numbers over from
  another event.** Near-duplicates add little together; a strongly negative pair are
  inverses of one signal, so never blend both raw. Whatever you train on, you still
  submit a single prediction column, scored on the graded target.
  [`notebooks/01_explore_the_data.ipynb`](notebooks/01_explore_the_data.ipynb) prints that
  correlation matrix for the dataset you are on.
- **Feature neutralization** can add AIMC the same way, by reducing a model's exposure to
  dominant feature groups: project those features out of your predictions **per exped**
  (neutralization is cross-sectional) at a proportion you sweep, and watch what it costs in
  CORR: a full neutralization that flattens CORR has removed the signal along with the
  exposure. The [`eiq-model-implementation`](.claude/skills/eiq-model-implementation/SKILL.md)
  skill carries the projection helper and the offline AIMC proxy that scores the sweep.
- Lower-turnover models tend to score better over time.
- **A negative score on the practice board is not a verdict on your model.** `validation` covers
  a later period than `train`, separated by a gap, so a sound model can score negative there and
  positive on a `train` holdout. That period is simply harder to predict: nothing is inverted or
  sign-flipped to catch you out, so do not price in a trap that is not there. Never respond by
  flipping the sign of your predictions: that
  fits the one period you can see and inverts on the next. Compare the terms instead, since raw
  CORR negative with **NCORR** near zero or positive means the loss is core-feature exposure
  rather than your signal, and neutralising that exposure is the legitimate fix. Optimise for a
  model that generalises across periods, because every round is scored on one you have not seen.

## Research skills

If your agent supports skills (e.g. Claude Code), [`.claude/skills/`](.claude/skills) holds a
research workflow you can load:

- [`eiq-research`](.claude/skills/eiq-research/SKILL.md), the orchestrator: sequences the others
  for any "try a new idea" request.
- [`eiq-experiment-design`](.claude/skills/eiq-experiment-design/SKILL.md). Plan and run
  scout→scale experiments in rounds.
- [`eiq-model-implementation`](.claude/skills/eiq-model-implementation/SKILL.md). Write a custom
  training script for serverless GPU compute.
- [`eiq-event-submission`](.claude/skills/eiq-event-submission/SKILL.md), go live: create a
  model, submit into the open round, verify, and (optionally) stake.
- [`eiq-report-research`](.claude/skills/eiq-report-research/SKILL.md). Write up results and
  generate the standard plots.
