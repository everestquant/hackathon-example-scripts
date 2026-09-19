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
  were fitting, and the lane follows the clock, not your intent.
- **Each round is a disjoint `id` namespace.** A prediction frame built for an earlier
  round will not match the open one, and submitting it scores nothing.
- The scored target is whatever `get_dataset_schema` reports as `primary_target`.
  Read it at runtime: it differs between datasets, it is not necessarily the first
  entry in `targets`, and `primary_target_listed: false` just means the dataset
  publishes it under an alias. It is still the column you are graded on.
- Every event upload: round or practice board, **requires the model's `.pkl`**
  (`model_pkl=`), store-only and never executed. Declare `model_pkl_python_version` from
  the interpreter that pickled the model
  (`f"{sys.version_info.major}.{sys.version_info.minor}"`); omitting it is treated as
  `3.11`, so an artifact pickled under a newer interpreter can fail to load with no
  traceback.
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

1. **Register the slot**: `create_model(name=...)`. The name becomes your `model_id`.
   Idempotent, so safe to re-run.
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
9. **(Optional) check event staking**: only after the pre-staking checklist and
   explicit operator OK; most events are display-only, so confirm first.

## 1-2. Register and read the clock

```python
import os
from everestapi import EverestAPI

client = EverestAPI(api_key=os.environ["EIQ_API_KEY"], tournament="futures")

client.create_model(name="my-event-model")   # idempotent; model_id == name
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

Re-read `get_started` immediately before submitting. A round can open or close while you
were fitting, and the lane follows the clock, not your intent. `submit_event_predictions`
and `submit_validation_diagnostics` take the same arguments and their ids are NOT
interchangeable: a round's predictions sent down the practice lane is accepted (202) and
fails minutes later on zero id overlap, costing the submission and settling anything
staked on that model at nothing.

```python
import sys

now = client.get_started()
now_cadence = now.get("cadence") or {}
pyver = f"{sys.version_info.major}.{sys.version_info.minor}"

if now_cadence.get("intake_fenced"):
    # a round is settling: wait rather than spending an upload into a refusal
    ...
elif now_cadence.get("open_window"):
    result = client.submit_event_predictions(
        "my-event-model", predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
    )
else:
    result = client.submit_validation_diagnostics(
        "my-event-model", predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
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
client.get_validation_diagnostics(model_id="my-event-model")  # Sharpe, mean CORR, drawdown (display-only)
# (via MCP: run_validation_diagnostics: same read, tool name differs from the client method)
```

Metrics to read: **AIMC** (contribution beyond the live stake-weighted ai-model
consensus) and **NCORR** (neutralized correlation against a fixed core feature set) are
both paid terms alongside CORR20. `explain_scoring` reports the current weights, and a
leaderboard response's `rank_metric` says what that specific board is actually ordered
by. A model with high CORR but flat AIMC/NCORR is echoing the field and leaves part of
the score untouched.

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
    "my-event-model", predictions, model_pkl="model.pkl", model_pkl_python_version=pyver,
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
- **Submitting while intake is fenced**: `cadence.intake_fenced` is briefly true while a
  round settles and the next opens; wait for it to clear rather than retrying blind.
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
client.set_stake_allocation(model="my-event-model", amount_usdc="2.5", window="round_1")
client.withdraw_stake_allocation(model="my-event-model", window="round_1")  # unlocked draft only
```

Refusals name their reason (`stake_window_closed`, `window_mismatch`, `below_min_stake`,
`slot_limit`, `insufficient_balance`, `allocation_locked`, `event_deadline_passed`). A
chain read that failed is `null` beside an explicit `*_unavailable: true` flag, never a
bare `0`. Branch on the flag. Only platform-granted money can be staked; sending your
own USDC to a deposit address does not raise what you may allocate.

## Ask the operator before deploying

1. New model slot, or submit to an existing `model_id`?
2. If new: what name (it is permanent and becomes the `model_id`)?
3. Is the API key configured (`EIQ_API_KEY`)?
4. Run once now for the currently open round, or stand up a polling job for future
   rounds on your infra?
5. **Event staking is a separate, explicit yes/no. Do not draft a stake unless they say
   so, and confirm the event carries money first.**

## Quick reference

| Step | SDK / MCP |
|------|-----------|
| Register slot | `create_model(name)` |
| Read the clock | `get_started`, `get_status`. Branch on `cadence.open_window` |
| Served split | `download_dataset(split="live" \| "validation" \| "train")` |
| Submit a round | `submit_event_predictions(model_id, predictions, model_pkl, model_pkl_python_version)` |
| Submit several | `submit_event_predictions_batch`, **MCP tool only**; on the client, loop the row above |
| Practice board | `submit_validation_diagnostics(model_id, predictions, model_pkl, model_pkl_python_version)` |
| Board / standings | `get_diagnostics_leaderboard()`, `get_diagnostics_standings()` |
| Diagnostics | `get_validation_diagnostics` (MCP: `run_validation_diagnostics`) |
| Live weights | `explain_scoring` |
| Event staking | `get_event_staking`, `set_stake_allocation`, `withdraw_stake_allocation` |
