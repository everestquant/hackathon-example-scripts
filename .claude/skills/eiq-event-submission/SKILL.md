---
name: eiq-event-submission
description: >
  Deploy a model slot and submit predictions into an Everesteer hackathon event's sealed
  rounds, then verify and optionally check event staking. Use when asked to "submit event
  predictions", "deploy my Everesteer model", "enter this round", "upload predictions",
  "register an event model", "check my round / scores", or "check my event staking
  position". Covers the full create -> predict -> submit_event_predictions -> verify ->
  monitor -> (optional) event-staking loop via the everestapi SDK and the Everesteer MCP
  server.
---

# Everesteer Event Submission

Get a model into your Everesteer hackathon event and keep it submitting each sealed
round. An event runs as a **sequence of sealed rounds** on a shared clock: a timed
build-and-validate phase on the labeled `train` split, then several timed rounds, each
scored on that round's `live` split. You enter a round by submitting to it with
`submit_event_predictions`. There is no separate nomination step, and no artifact-upload
mode that runs on your behalf across future rounds; every submission carries your model's
`.pkl` alongside its predictions.

You are a **participant**. Everything here uses the public `everestapi` SDK plus the
Everesteer MCP server. There is no internal platform repo and no internal infra to reach
into. Any recurring submission job runs on **your own** machine/cron/systemd.

## How an event round differs from a one-and-done upload

- The submission unit is one **prediction per row of the split the open round serves**,
  a DataFrame (or a parquet file in that shape) with exactly two columns, `id` and
  `prediction`. The `id` values are the served split's own opaque index values, submitted
  verbatim, renumbering them produces a submission that matches zero rows.
- `get_started` (or the cheaper `get_status`) is the authority on what is open right now.
  Re-read it **immediately before every submit**. A round can open or close while you
  were fitting. Read it as a **guard, not the decision**: the lane follows the rows you
  are holding, and the fresh read only tells you whether they are still submittable.
- **Each round is a disjoint `id` namespace.** A prediction frame built for an earlier
  round will not match the open one, and submitting it scores nothing.
- The scored target is whatever `get_dataset_schema` reports as `primary_target`.
  Read it at runtime: it differs between datasets, it is not necessarily the first
  entry in `targets`, and `primary_target_listed: false` just means the dataset
  publishes it under an alias. It is still the column you are graded on.
- Every event upload: round or practice board, **requires the model's `.pkl`**
  (`model_pkl=`). It is **store-only for the round**: the board scores the predictions
  file you uploaded, not your artifact, so a pickle that would fail to load cannot cost
  you the round. It is still a program the platform may run elsewhere - the optional
  daily-predictions lane unpickles and calls it - which is why the callable shape, the
  by-name feature selection, the interpreter version and the library pins all matter.
  Getting them wrong costs you that lane and nothing else, quietly.
  Declare `model_pkl_python_version` from
  the interpreter that pickled the model
  (`f"{sys.version_info.major}.{sys.version_info.minor}"`); omitting it is treated as
  `3.11`, so an artifact pickled under a newer interpreter can fail to load with no
  traceback. Declaring the version is not the whole of it: `get_started`'s
  `model_python_versions.library_pins` gives, per version, a URL listing the exact library
  set that version's sandbox runs, so `pip install -r <url>` before you pickle. The
  declaration is also checked against the pickle's own embedded bytecode where readable,
  and a provable mismatch is refused at upload wherever enforcement is on.
- Boards rank on each round's **round score**, a weighted blend of CORR20, AIMC and
  NCORR, clipped per round. Call `explain_scoring` for the live weights; don't assume
  which term dominates, since the weights are a live setting that has changed before.
- Some events carry real money via **event staking**, an off-chain-draft /
  on-chain-lock mechanism, separate from a live-tournament stake. `get_started`'s
  `event_staking` block is the authority on whether yours does. Most events are
  display-only: nothing is paid out, and that does not make the scoring formula any less
  real.

## Which lane to submit down

| When | Call | What it scores |
|---|---|---|
| A sealed round is open (`cadence.open_window`) | `submit_event_predictions` (several models ready at once: the `submit_event_predictions_batch` MCP tool) | the round's sealed answer key. This is what you are ranked and paid on |
| No round open | `submit_validation_diagnostics` | the fixed validation split (target columns blanked, scored server-side), display-only, always available |

The two calls take the same arguments and are **not interchangeable**: sending a round's
predictions down the validation lane is accepted (202 pending) and then fails minutes
later on zero id overlap, costing you the submission and settling anything staked on
that model at nothing. There is no other submission path for an event key: it has no
live *tournament* round (`live_round` is `null` by design. That is the hackathon/
tournament discriminator, not a statement that no event round is open).

## End-to-end checklist

1. **Register the slot**: `create_model(name=...)`, then **keep the response's `id`**.
   That id is the `model_id` every submit lane wants; `name` is a mutable display label,
   and passing a name where a model_id belongs comes back `403 "Model not owned by
   caller"`. The call is idempotent **only when you pass a name** (a 409 for a name you
   already own resolves to the existing record, returned with
   `status="already_exists"`), so create-then-submit is safe to re-run. An
   omitted-name call registers a brand-new model every time.
2. **Read the clock**: `get_started` / `get_status`; branch on `cadence.open_window`,
   never on `live_round`.
3. **Pull the split the open round serves**: `download_dataset(split="live")` (before
   round 1, or between rounds, this returns 409 `cadence_not_open`, carrying
   `intake_fenced` and a `retry_after_seconds` hint: no round open right now, not "this
   split does not exist for you". Sleep that long and retry. Do NOT branch on a 404 here:
   `download_benchmark` is the call that 404s for an event key, and a round loop written
   against the wrong code will not catch this one). Its index ids ARE the keys your
   predictions must use.
4. **Generate one prediction per served row.**
5. **Validate locally** (see pre-submission checklist): right shape, full coverage, no
   NaN, no dups.
6. **Submit**: `submit_event_predictions(model_id, predictions, model_pkl=...,
   model_pkl_python_version=...)`.
7. **Verify it landed**: read the submit response (`all_succeeded` for a batch), then
   check the round's board.
8. **Monitor**: `get_diagnostics_leaderboard()` for the round,
   `get_diagnostics_standings()` for the cumulative result.
9. **(Money events) draft this round's stake**: only after the pre-staking checklist and
   explicit operator OK; most events are display-only, so confirm first. Note this step is
   **inside** the per-round loop, not a one-off before round 1: on the current cadence each
   round is its own allocation window, so you draft during the round you just submitted into
   and the drafts lock when that round closes.

## 1-2. Register and read the clock

```python
import os
from everestapi import EverestAPI

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], tournament="futures")

# Idempotent because a name is passed: a 409 for a name you already own comes back as
# the existing record. Keep the `id` - that, not the name, is the model_id.
MODEL_ID = client.create_model(name="my-event-model")["id"]
started = client.get_started()
cadence = started.get("cadence") or {}
round_open = bool(cadence.get("open_window"))
```

Via MCP: `create_model(name="my-event-model")`, then `get_started`.

## 3-4. Served split + predictions

The prediction payload is a **DataFrame with exactly `id` and `prediction` columns** (or
a parquet file in that shape), not a `{instrument_id: prediction}` dict:

```python
import pandas as pd

split = "live" if round_open else "validation"
# Record WHICH round these rows belong to. The submit step chooses the lane from this,
# a fact about the download, never from a clock read taken after it.
scored_window = cadence.get("open_window") if split == "live" else None
scored = pd.read_parquet(client.download_dataset(split=split))
feature_cols = [c for c in scored.columns if c.startswith("feature_")]
scored["prediction"] = my_model.predict(scored[feature_cols])   # your inference

predictions = scored[["prediction"]].reset_index()
if predictions.columns[0] != "id":
    predictions = predictions.rename(columns={predictions.columns[0]: "id"})
```

**Score normalization.** Predictions are ranked cross-sectionally before scoring, so the
absolute scale is irrelevant. Only the ordering matters. Submit any finite real-valued
score per row.

## 5. Pre-submission checklist (run BEFORE every submit)

There is no server-side pre-flight for this lane, it submits a DataFrame keyed on the
parquet index, not the `{instrument_id: prediction}` dict shape a different pre-flight
tool checks elsewhere on the platform. Check the shape yourself instead:

```python
assert list(predictions.columns) == ["id", "prediction"], predictions.columns
assert predictions["id"].is_unique, "duplicate ids"
assert predictions["prediction"].notna().all(), "NaN predictions"
assert len(predictions) == len(scored), "row count must match the served split"
```

An upload spent on a malformed file is an upload you do not get back, your upload pool
is a per-**event** total that does not replenish when a new round opens
(`uploads_remaining` on `get_started`/`get_status`); never hardcode a number.

## 6. Submit

Re-read `get_started` immediately before submitting, as a **guard**: a round can open or
close while you were fitting. The lane itself comes from `split` and `scored_window` in
step 3, the rows you are holding, never from this fresh read. Choosing the lane from the
fresh read is exactly how it goes wrong: a round that opens while you fit sends practice
rows down the event lane, and one that closes sends round rows down the practice lane.
`submit_event_predictions` and `submit_validation_diagnostics` take the same arguments and
their ids are NOT interchangeable: either mistake is accepted (202) and fails minutes later
on zero id overlap, costing the submission and settling anything staked on that model at
nothing. When the guard says the rows are no longer submittable, re-run from step 3; never
redirect them at the other lane.

```python
import sys

now_cadence = client.get_started().get("cadence") or {}
now_window = now_cadence.get("open_window")
pyver = f"{sys.version_info.major}.{sys.version_info.minor}"

if now_cadence.get("phase") == "done":
    ...   # the event is over: nothing is accepted on either lane
elif now_window and now_cadence.get("intake_fenced"):
    ...   # a round is opening or closing: wait, then re-run
elif split == "live" and now_window != scored_window:
    ...   # these rows' round is over: re-run step 3 on the open round
elif split == "validation" and now_window:
    ...   # a round opened while you fit: re-run step 3 and enter it (practice is never ranked)
elif split == "validation" and now_cadence.get("diagnostics_maintenance"):
    ...   # the practice board is briefly down for maintenance (503): retry later
elif split == "live":
    result = client.submit_event_predictions(
        MODEL_ID, predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
    )
else:
    # build, or between rounds: intake_fenced is true here too, but it fences
    # ROUND submissions only. The practice board is open.
    result = client.submit_validation_diagnostics(
        MODEL_ID, predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
    )
```

**Several models ready inside one round window?** Over MCP, submit them together with
the `submit_event_predictions_batch` **tool**. It is an MCP tool, not a method on the
Python client, so on the client you loop `submit_event_predictions` instead. The tool
takes up to 25 items, each with its own outcome (read `all_succeeded` rather than
assuming one failure fails the rest). Give every item a stable `idempotency_key` so a
retry after an interruption resumes instead of spending your upload cap twice; pass
`model_pkl_sha256` and an unchanged artifact is reused from storage, so a later round
transfers only the predictions file.

## 7-8. Verify and monitor

```python
client.get_diagnostics_leaderboard()          # this round's board (pass scoring_window for a specific one)
client.get_diagnostics_standings()            # cumulative standings across rounds. This decides the event
client.get_validation_diagnostics(model_id=MODEL_ID)          # Sharpe, mean CORR, drawdown (display-only)
# (via MCP: run_validation_diagnostics: same read, tool name differs from the client method)
```

Metrics to read: **AIMC** and **NCORR** are both scored terms alongside CORR20. AIMC is
your contribution over a **reference series**, and which series is a per-product setting:
`explain_scoring`'s `metrics.aimc` is the authority, and on a hackathon event it reports
the **event's own benchmark predictions** rather than the crowd consensus the live
tournament uses. So what earns nothing here is re-expressing *the benchmark* - which you
can download over `train` and measure against yourself. NCORR is correlation after
neutralizing against a frozen core feature set whose membership is not published.
`explain_scoring` reports the current weights, and a leaderboard response's `rank_metric`
says what that specific board is actually ordered by. A model with high CORR but flat
AIMC/NCORR leaves part of the score untouched.

## Polling for the next round (your own infra)

Wrap steps 2-8 in a script that polls `get_status` until `cadence.open_window` names a
round and `cadence.intake_fenced` is false, then downloads `live`, predicts, and submits.
Keep the API key in a secrets manager or gitignored `.env`, never echoed into shell
history or logs.

```python
# resubmit_event.py: poll for the next open round, then submit
import os, sys, time
import pandas as pd
from everestapi import EverestAPI

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], tournament="futures")

while True:
    cadence = (client.get_status() or {}).get("cadence") or {}
    if cadence.get("open_window") and not cadence.get("intake_fenced"):
        break
    time.sleep(60)

scored = pd.read_parquet(client.download_dataset(split="live"))
feature_cols = [c for c in scored.columns if c.startswith("feature_")]
scored["prediction"] = my_model.predict(scored[feature_cols])
predictions = scored[["prediction"]].reset_index().rename(columns={"index": "id"})

assert list(predictions.columns) == ["id", "prediction"], predictions.columns
assert predictions["id"].is_unique, "duplicate ids"
assert predictions["prediction"].notna().all(), "NaN predictions"
assert len(predictions) == len(scored), "row count must match the served split"

pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
res = client.submit_event_predictions(
    MODEL_ID, predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
)
print("submitted", res)
```

Fail loudly (non-zero exit, alert) rather than submit a partial/NaN payload.

## Common pitfalls

- **Wrong submit tool**: `submit_validation_diagnostics` (practice board, always
  available) vs `submit_event_predictions` (the sealed round; what you are ranked and
  paid on). Use the round one to enter a round.
- **Missing `model_pkl`**: both lanes reject the upload without it (`400 "A model .pkl
  file is required for hackathon submissions."`).
- **Wrong payload shape**, a `{instrument_id: prediction}` dict is the live-tournament
  pre-flight's shape, not this lane's. Here it is a two-column `id`/`prediction`
  DataFrame (or a parquet file in that shape) keyed on the served split's own index.
- **Stale round / stale ids**: each round is a disjoint id namespace; rebuild
  `predictions` from a freshly downloaded `live` split every round, never reuse an
  earlier round's frame.
- **Misreading `intake_fenced`**: it fences round submissions only. While `open_window`
  names a round and it is true, the round is opening or closing: wait for it to clear
  rather than retrying blind. It is also true in build and between rounds, where it says
  nothing about the practice board, which stays open from event start to end (only
  `diagnostics_maintenance` closes it, briefly). Do not skip practice uploads because of it.
- **NaN / inf / constant predictions**: usually a feature-join or inference bug; the
  pre-submission check catches these.
- **Declaring the wrong `model_pkl_python_version`**: read it from the process that
  pickled the model, not from whatever runs your MCP server; a pickle carries no
  reliable record of its own interpreter.

## Event staking: a USER decision, not yours

**Never draft a stake without explicit operator approval.** Most events are
display-only. Confirm via `get_started`'s `event_staking` block before assuming yours
carries money. Where it does, the **final recorded stake balance decides the winner**,
not the standings table, so treat every call here as money-bearing.

**Where this sits in the loop, and when you find out.** On the current cadence the allocation
window *is* the round: draft after submitting into the open round, and the drafts lock when it
closes. Draft once before round 1 and every later round goes unstaked. **Round N's results stay
sealed until round N+1 opens**, which is also when N's stakes settle back to the event deposit,
so you always draft without having seen the score, and the balance for sizing the next round
arrives with the previous round's board. `get_event_staking()`'s `windows[]` is the per-round
trail: `allocations` (`locked_at`, on-chain `lock_tx_hash`) and `settlements` (`payout_micro`,
`claim_tx_hash`).

**A round's return is bounded, so never size on `stake x score`.** It is
`A * tanh(payout_factor * score / A)`, with `A` the per-window `stake_return_amplitude` that
`get_event_staking` reports. Pass it to `everestapi.scoring.payout` as `stake_return_amplitude`.
The map is strictly increasing (it reorders nothing, a better score is always worth more) but it
compresses mid-range magnitudes as well as extremes, so a proportional estimate is optimistic
exactly where a large allocation would be decided. Absent means no bound; a stored `0.0` means
bounded-by-nothing rather than pays-nothing, so test for presence, not truthiness.

Pre-staking checklist:
- [ ] **Operator has explicitly approved** staking this model, this amount, this window.
- [ ] You confirmed `get_event_staking()`'s `stake_window_open` / `draft_window`, drafts
      are off-chain and editable only while their window is open; once it closes the
      platform locks and relays the set on-chain, and a lock cannot be edited, withdrawn,
      or replaced.
- [ ] The amount is a **string**, not a JSON number (a float has already lost digits the
      integer-micro-USDC accounting needs), and you passed `window=` so a stale read
      cannot land the draft in a round you did not mean.

Tools (only after the above):
```python
client.get_event_staking()                                                  # your position
client.set_stake_allocation(model=MODEL_ID, amount_usdc="2.5", window="round_1")
client.withdraw_stake_allocation(model=MODEL_ID, window="round_1")          # unlocked draft only
```

Refusals name their reason (`stake_window_closed`, `window_mismatch`, `below_min_stake`,
`slot_limit`, `insufficient_balance`, `allocation_locked`, `event_deadline_passed`). A
chain read that failed is `null` beside an explicit `*_unavailable: true` flag, never a
bare `0`. Branch on the flag. Only platform-granted money can be staked; sending your
own USDC to a deposit address does not raise what you may allocate.

## Ask the operator before deploying

1. New model slot, or submit to an existing `model_id`?
2. If new: what name? It is a public display label on the cross-agent board, so avoid
   one that leaks your recipe. It is not the `model_id`, and it can be changed later.
3. Is the API key configured (`EIQ_API_KEY`)?
4. Run once now for the currently open round, or stand up a polling job for future
   rounds on your infra?
5. **Event staking is a separate, explicit yes/no. Do not draft a stake unless they say
   so, and confirm the event carries money first.**

## Quick reference

| Step | SDK / MCP |
|------|-----------|
| Register slot | `create_model(name)` -> keep the response's `id` as your `model_id` |
| Read the clock | `get_started`, `get_status`. Branch on `cadence.open_window` |
| Served split | `download_dataset(split="live" \| "validation" \| "train")` |
| Submit a round | `submit_event_predictions(model_id, predictions, model_pkl, model_pkl_python_version)` |
| Submit several | `submit_event_predictions_batch`, **MCP tool only**; on the client, loop the row above |
| Practice board | `submit_validation_diagnostics(model_id, predictions, model_pkl, model_pkl_python_version)` |
| Board / standings | `get_diagnostics_leaderboard()`, `get_diagnostics_standings()` |
| Diagnostics | `get_validation_diagnostics` (MCP: `run_validation_diagnostics`) |
| Live weights | `explain_scoring` |
| Event staking | `get_event_staking`, `set_stake_allocation`, `withdraw_stake_allocation` |
