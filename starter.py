#!/usr/bin/env python3
"""
Everesteer Hackathon, Starter Example

The hackathon is a different product from the Himalayas (futures) tournament, and
your key is scoped to it. The shape of an event:

  1. fit on `train`:      LABELED (features + target_* columns), the largest split
  2. predict `validation`, target columns BLANKED: the PRACTICE board that runs
                            before round 1. Answers stay server-side, always
  3. then N sealed rounds. Each open round is served on `live`. FOUR is one
                            event's configuration, not a rule: read the clock

The objective is to predict the held-out target better than the field.

This script runs the whole loop once: orient, download, fit a LightGBM baseline,
predict the split that is currently scored, and submit it down the correct lane.

It produces:
  - hackathon_model.pkl:        the fitted model (the event lane REQUIRES it)
  - hackathon_predictions.parquet, id + prediction

Usage:
    pip install "everestapi>=0.3.32" lightgbm scikit-learn pandas pyarrow cloudpickle
    export EIQ_API_KEY=...                  # from your event onboarding
    export EIQ_BASE_URL=https://hackathon.everesteer.ai
    python starter.py

Connecting: production needs only your API key. A staging or preview host also
sits behind Cloudflare Access and will bounce an API-key-only request at the edge
(a 302 to a login page, or error 1010). Which does not look like an auth failure.
The SDK handles it for you: set CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET in
the environment (or pass cf_access_client_id= / cf_access_client_secret= to
EverestAPI) and the service-token headers ride alongside your key. Interactive
`cloudflared access login <host>` also works.
"""

from __future__ import annotations

import os
import sys

import cloudpickle
import lightgbm as lgb
import pandas as pd
from everestapi import EverestAPI

client = EverestAPI(
    api_key=os.environ["EIQ_API_KEY"],
    base_url=os.environ.get("EIQ_BASE_URL", "https://hackathon.everesteer.ai"),
    tournament="futures",
)


def bail(step: str, exc: Exception) -> None:
    """Fail with something you can act on instead of a traceback."""
    print(f"\n{step} failed: {exc}")
    print("If this is a staging or preview host, check the Cloudflare Access note")
    print("in this file's docstring. An edge bounce is not an auth failure.")
    sys.exit(1)


# =====================================================================
# 1. Orient: get_started is the authority on what to do right now
# =====================================================================
# It is mode-aware: it answers for YOUR key. Two things to know about the
# hackathon payload:
#
#   * `live_round` is null BY DESIGN. It means "no live public tournament round"
#     and is the hackathon/tournament discriminator. Do NOT branch on it.
#     Branch on `cadence.open_window`, which is the round signal.
#   * `uploads_remaining` is how many uploads you have LEFT, not your cap. The
#     cap is a per-EVENT pool: it counts across every model and every round and
#     does NOT replenish when a new round opens. Only round submissions draw on
#     it; practice-board uploads are free. Never hardcode a number.
try:
    started = client.get_started()
except Exception as exc:  # noqa: BLE001, first network call, report it plainly
    bail("get_started", exc)

cadence = started.get("cadence") or {}
round_open = bool(cadence.get("open_window"))

print("Event orientation")
print(f"  phase              : {cadence.get('phase')}")
print(f"  phase ends at      : {cadence.get('phase_ends_at')}")
print(f"  round open now     : {round_open}")
print(f"  intake fenced      : {cadence.get('intake_fenced')}")
print(f"  uploads remaining  : {started.get('uploads_remaining')}")
print(f"  hosted train funded: {started.get('hosted_train_funded')}")

if cadence.get("phase") == "done":
    print("\nThe event is over (phase 'done'): no round or practice upload is accepted now.")
    sys.exit(0)

# =====================================================================
# 2. Schema: read the target and features, never hardcode them
# =====================================================================
try:
    schema = client.get_dataset_schema()
except Exception as exc:  # noqa: BLE001
    bail("get_dataset_schema", exc)

targets = schema.get("targets") or []
# The graded column is `primary_target` and only `primary_target`. It is NOT
# targets[0]: `targets` is the dataset's own target block in its own order, so
# the first entry is just whichever name happens to sort first, which is a
# different column on most datasets. Fitting that one instead fails silently.
target_col = schema.get("primary_target")
if not target_col:
    bail(
        "get_dataset_schema",
        RuntimeError("schema declares no primary_target - refusing to guess"),
    )
print(f"\nTarget: {target_col}   (schema advertises {len(targets)} target(s))")

# The missing sentinel is a dataset fact too. `feature_encoding` declares the
# bin count, the value range and the sentinel; read it rather than assuming -1.
# It is absent on a tree that does not declare its encoding, which is not
# permission to assume a default either, so fall back only as a last resort.
encoding = schema.get("feature_encoding") or {}
MISSING = encoding.get("missing")
if MISSING is None:            # absent OR published as null; neither declares a sentinel
    MISSING = -1.0
print(f"Missing sentinel: {MISSING!r}   (from schema['feature_encoding'])")
if schema.get("primary_target_listed") is False:
    # Expected on a dataset that publishes the graded column under an alias:
    # it is served and scored either way. Predict it, do not substitute.
    print("  note: graded column not listed among 'targets' - predict it anyway")

# =====================================================================
# 3. Download: train, plus whichever split is scored right now
# =====================================================================
# A round open means `live` is the scored split. Between rounds (and before
# round 1 is published) `live` returns 409 cadence_not_open, carrying
# intake_fenced and a retry_after_seconds hint: that means "no round open right
# now", never "this split does not exist for my key". download_benchmark is the
# call that genuinely 404s for an event key; the two refuse for different
# reasons, so do not branch on one expecting the other.
try:
    train_path = client.download_dataset(split="train")
except Exception as exc:  # noqa: BLE001
    bail("download_dataset(split='train')", exc)

# `scored_window` records WHICH round these rows came from. Step 8 needs it: the
# lane is decided by the rows in hand, and "which rows are these" is a fact about
# the download, not about whatever the clock says some minutes later.
scored_split = "live" if round_open else "validation"
scored_window = cadence.get("open_window") if round_open else None
try:
    scored_path = client.download_dataset(split=scored_split)
except Exception as exc:  # noqa: BLE001
    if scored_split == "live":
        print(f"\nlive is not being served right now ({exc}).")
        print("No round is open, practising on the validation board instead.")
        scored_split = "validation"
        scored_window = None
        try:
            scored_path = client.download_dataset(split="validation")
        except Exception as inner:  # noqa: BLE001
            bail("download_dataset(split='validation')", inner)
    else:
        bail(f"download_dataset(split='{scored_split}')", exc)

train = pd.read_parquet(train_path)
scored = pd.read_parquet(scored_path)
print(f"\nTrain : {len(train):>8,} rows   ({train_path})")
print(f"Scored: {len(scored):>8,} rows   split={scored_split}  ({scored_path})")

# =====================================================================
# 4. Fit a LightGBM baseline
# =====================================================================
# Feature values are ENCODED into bins. The bin count differs between datasets,
# so read `schema["feature_encoding"]` rather than assuming one: it declares
# the bin count, the value range and the `missing` sentinel (commonly -1.0, set
# where the source was not onboarded for that instrument/date). Treat missing as
# NaN or as its own category, NEVER as an ordinal below the lowest real bin.
# A NaN target means the row was uncomputable; it is never imputed, so drop
# those rows rather than filling them.
def features_matrix(df, cols):
    """Bin codes as float, with the missing sentinel turned into real NaN.

    LightGBM handles NaN natively as "missing". `.fillna(0.0)` does NOT do this
    job and is worse than it looks: the served split carries no NaNs at all, so
    the fill never fires, and every sentinel reaches the model as an ordinal
    below the lowest real bin, which is precisely what the note above forbids.
    """
    x = df[cols].astype("float32")
    return x.mask(x == MISSING)


feat_cols = sorted(c for c in train.columns if c.startswith("feature_"))
fit = train.dropna(subset=[target_col])
print(f"\nFitting on {len(fit):,} labeled rows over {len(feat_cols)} features...")

model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
model.fit(features_matrix(fit, feat_cols), fit[target_col])

# =====================================================================
# 5. Predict: the id is the PARQUET INDEX, not a column
# =====================================================================
# The downloaded parquet has NO 'id' column. The id every submit lane wants is
# the index (its name is 'id'), and its values are opaque strings like
# 'eiq_559073fecf705ae5'. Submit them VERBATIM. Renumbering them 0..N-1 produces
# a submission that matches ZERO rows.
predictions = pd.DataFrame(
    {"prediction": model.predict(features_matrix(scored, feat_cols))},
    index=scored.index,
).reset_index()
if predictions.columns[0] != "id":
    predictions = predictions.rename(columns={predictions.columns[0]: "id"})
print(f"Predicted {len(predictions):,} rows; first id: {predictions['id'].iloc[0]!r}")

pred_path = "hackathon_predictions.parquet"
predictions.to_parquet(pred_path, index=False)


# =====================================================================
# 6. Pickle the model: ONE artifact shape is accepted
# =====================================================================
# Everesteer runs exactly one shape: a CLOUDPICKLED CALLABLE. Pickle a
# `predict(live_features)` function -- or `predict(live_features,
# live_benchmark_models)` to also receive the published live benchmark
# series -- returning a SINGLE-COLUMN pandas DataFrame indexed by
# instrument id. A bare estimator, or a dict wrapping one, is refused:
#   400 "Everesteer runs one model shape: a cloudpickled callable."
# So use cloudpickle.dump, never pickle.dump.
#
# Select the features BY NAME inside predict. The artifact then survives a
# change to the served column set, instead of silently mispredicting on a
# positional array whose columns have shifted underneath it.
def build_predict(fitted, columns, missing):
    # `missing` is closed over by value so the pickled artifact carries the
    # sentinel with it, instead of depending on a global that only exists here.
    def predict(live_features, live_benchmark_models=None):
        x = live_features.reindex(columns=columns).astype("float32")
        # A column the served split does not carry reindexes to real NaN, which
        # is the honest encoding for "absent" and what LightGBM already expects.
        x = x.mask(x == missing)
        return pd.DataFrame({"prediction": fitted.predict(x)}, index=live_features.index)

    return predict


model_path = "hackathon_model.pkl"
with open(model_path, "wb") as fh:
    cloudpickle.dump(build_predict(model, feat_cols, MISSING), fh)
print(f"Wrote {pred_path} and {model_path}")

# =====================================================================
# 7. Sanity-check the payload before spending an upload
# =====================================================================
# Note: client.validate_submission() is the TOURNAMENT pre-flight: it takes a
# {instrument_id: prediction} dict and checks a futures submission. It does not
# apply to the hackathon lanes, which take a DataFrame keyed on the parquet
# index. So check the shape yourself; an upload spent on a malformed file is an
# upload you do not get back.
assert list(predictions.columns) == ["id", "prediction"], predictions.columns
assert predictions["id"].is_unique, "duplicate ids"
assert predictions["prediction"].notna().all(), "NaN predictions"
assert len(predictions) == len(scored), "row count must match the served split"
print(f"\nPayload OK: {len(predictions):,} unique ids, no NaNs.")

# =====================================================================
# 8. Submit down the CORRECT lane
# =====================================================================
# The lane follows the ROWS YOU ARE HOLDING -- `scored_split` above -- never a
# clock read taken after they were downloaded. Those are two different questions,
# and the gap between them is exactly where this goes wrong:
#
#   * a fence that clears while you fit turns "no round open" into "round open",
#     and choosing the lane from that fresh read sends PRACTICE rows down the
#     event lane;
#   * a round that closes while you fit does the reverse, and sends ROUND rows
#     down the practice lane.
#
# Either way the upload is accepted (202 pending) and fails minutes later on zero
# id overlap, because the splits are disjoint id namespaces. You lose the upload,
# the minutes, and anything staked on that model settles at nothing.
#
# So re-read get_started IMMEDIATELY before submitting, but read it as a GUARD,
# not as the decision: it answers "are the rows I am holding still submittable?".
# When they are not, the move is to re-run, never to redirect them at the other
# lane.
#
# From the SDK's own warning on these two calls:
#   "The two take the same arguments and their ids are NOT interchangeable:
#    sending a round's predictions down the validation lane is accepted (202)
#    and then fails minutes later on zero id overlap, costing you the
#    submission and settling anything staked on that model at nothing."
#
# submit_futures_predictions is the TOURNAMENT lane and never applies here.
# A model must EXIST before you can submit predictions or upload a .pkl for it,
# the platform never auto-creates one, and submitting to a name it does not know
# comes back as a 404 telling you to create it first. Reuse the same model across
# rounds so its board history stays on one entry.
# create_model(name=...) is itself idempotent -- a 409 for a name you already own comes
# back as the existing record with status="already_exists" -- so the lookup below is not
# strictly required. It is here because reusing the SAME model across rounds is what keeps
# your board history on one entry, and a lookup makes that explicit.
MODEL_NAME = os.environ.get("EIQ_MODEL_ID", "hackathon-baseline")
try:
    listed = client.get_models() or {}
    existing = {m.get("name"): m.get("id") for m in (listed.get("models") or [])}
except Exception:  # noqa: BLE001, fall through to create
    existing = {}

if MODEL_NAME in existing:
    MODEL_ID = existing[MODEL_NAME]
    print(f"\nUsing existing model {MODEL_NAME!r} ({MODEL_ID})")
else:
    try:
        created = client.create_model(name=MODEL_NAME, description="hackathon starter baseline")
        MODEL_ID = created["id"]
        print(f"\nCreated model {MODEL_NAME!r} ({MODEL_ID})")
    except Exception as exc:  # noqa: BLE001
        bail("create_model", exc)

try:
    now = client.get_started()
except Exception as exc:  # noqa: BLE001
    bail("get_started (pre-submit re-read)", exc)

now_cadence = now.get("cadence") or {}
now_window = now_cadence.get("open_window")

if now_cadence.get("phase") == "done":
    print("\nThe event ended while this was fitting: nothing is accepted any more.")
    sys.exit(0)

# `intake_fenced` fences ROUND submissions only. It is also true in build, in stake
# and between rounds, where no round's data is open; the practice board takes
# uploads there regardless. So it only means "wait" while a round is named.
if now_window and now_cadence.get("intake_fenced"):
    print(f"\nRound {now_window!r} is opening or closing and not taking predictions.")
    print("Wait for the next phase, then re-run.")
    sys.exit(0)

if scored_split == "live" and now_window != scored_window:
    # The round these ids belong to is over. Nothing scores them now: the event
    # lane has moved to a different id namespace, and the practice board never
    # shared this one. Predict the open round rather than redirect these.
    print(f"\nThese rows are round {scored_window!r}; the open round is now {now_window!r}.")
    print("Re-run: download `live` again for the open round and predict on it.")
    sys.exit(1)

if scored_split == "validation" and now_window:
    # A round opened while this was fitting. The practice board WOULD take these
    # rows -- it takes them in every phase, and practice uploads are free -- but
    # it is display-only: nothing there is ranked or paid. Re-run, and put these
    # predictions into the round instead.
    print(f"\nRound {now_window!r} opened while this was fitting; these are practice rows.")
    print("Re-run to enter the round: the practice board is display-only and is never ranked.")
    sys.exit(1)

if scored_split == "validation" and now_cadence.get("diagnostics_maintenance"):
    # The one state that closes the practice board, and it is temporary.
    print("\nThe practice board is briefly down for maintenance. Retry in a few minutes.")
    sys.exit(0)

try:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Name the graded column on every upload. The SDK otherwise sends its own
    # default, target_everest_20, a column this dataset does not have.
    if scored_split == "live":
        print(f"\nRound {scored_window} is open and these are its rows: the event lane.")
        result = client.submit_event_predictions(
            MODEL_ID, predictions, target=target_col, model_pkl=model_path,
            model_pkl_python_version=pyver,
        )
    else:
        # BOTH hackathon lanes require the .pkl. The SDK docstring calls model_pkl
        # the thing that distinguishes the event lane, but a hackathon key is
        # refused on the practice board without it too:
        #   400 "A model .pkl file is required for hackathon submissions."
        print("\nThese are practice-board rows: the validation lane.")
        result = client.submit_validation_diagnostics(
            MODEL_ID, predictions, target=target_col, model_pkl=model_path,
            model_pkl_python_version=pyver,
        )
except Exception as exc:  # noqa: BLE001
    if scored_split == "validation" and getattr(exc, "status_code", None) == 503:
        # Maintenance began after the re-read above. Nothing was spent: retry later.
        print(f"\nThe practice board is briefly down for maintenance ({exc}). Retry later.")
        sys.exit(0)
    bail("submit", exc)

print(f"Accepted: {result}")

# =====================================================================
# 9. Money: read your staking position. THIS READS, IT NEVER STAKES
# =====================================================================
# Most events are display-only. Where get_started's event_staking block reports
# money_event, the FINAL RECORDED STAKE BALANCE is the result -- not the
# standings table -- and each round is its OWN allocation window: you draft
# during the round you just submitted into, and the drafts lock when that round
# closes. Which is why this sits here, right after the submit, and not once
# before round 1: draft only there and rounds 2..N go unstaked.
#
# Round N's results stay sealed until round N+1 opens, and that is also when N's
# stakes settle back to your deposit. So you always draft without having seen
# the score, and the balance you size the next round from arrives with the
# previous round's board.
#
# This script only READS. Drafting spends real money, so it stays an explicit
# decision you make, not something a starter does on your behalf:
#
#   client.set_stake_allocation(model=MODEL_ID, amount_usdc="2.5", window=<round>)
#
# with the amount as a STRING -- a JSON number is refused rather than rounded,
# because a binary float has already lost the digits the 6-decimal rule exists
# to keep -- and `window` passed so a stale read cannot land a draft in a round
# you did not mean.
if (now.get("event_staking") or {}).get("money_event"):
    try:
        pos = client.get_event_staking()
    except Exception as exc:  # noqa: BLE001, a read: report it and carry on
        print(f"\nget_event_staking failed: {exc}")
    else:
        # A chain read that failed comes back as null beside an explicit
        # *_unavailable flag, never as 0. "Could not read" is not "you have
        # nothing", so branch on the flag rather than on the number.
        print("\nEvent staking (this event carries money)")
        if pos.get("balance_unavailable"):
            print("  balance      : UNAVAILABLE (chain read failed, not zero)")
        else:
            print(f"  balance      : {pos.get('net_usdc')} USDC"
                  f"  (principal {pos.get('principal_usdc')}, settled {pos.get('settled_usdc')})")
        print(f"  slots/round  : {pos.get('max_slots')}   min stake: {pos.get('min_stake_usdc')} USDC")

        draft_window = pos.get("draft_window")
        if draft_window:
            # The return is BOUNDED, so it is not proportional to the score: it is
            # A * tanh(payout_factor * score / A). Size against that, not against
            # stake x score. Absent means no bound; a stored 0.0 means
            # bounded-by-nothing, not pays-nothing, so test presence not truth.
            window = next((w for w in (pos.get("windows") or [])
                           if w.get("window") == draft_window), {})
            if "stake_return_amplitude" in window:
                print(f"  return bound : A={window['stake_return_amplitude']} "
                      "(pass to everestapi.scoring.payout as stake_return_amplitude)")
            print(f"  DRAFTING IS OPEN for {draft_window}; drafts lock when it closes.")
        else:
            print("  No window is draftable right now.")

# =====================================================================
# 10. Where your score shows up
# =====================================================================
print("\nNext:")
print("  client.get_diagnostics_leaderboard():  the board for a round")
print("     (pass scoring_window to read a specific round's board)")
print("  client.get_diagnostics_standings():    cumulative standings across rounds")
print("  client.get_event_staking():            the authority on whether this")
print("     event carries money, and whether staking is enabled for you")
print("\nBoards rank on the round score: a weighted blend of CORR, AIMC and NCORR,")
print("clipped per round. Call explain_scoring for the live weights and do not")
print("assume which term dominates.")
