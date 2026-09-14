#!/usr/bin/env python3
"""
Everesteer Hackathon — Starter Example

The hackathon is a different product from the Himalayas (futures) tournament, and
your key is scoped to it. The shape of an event:

  1. fit on `train`      — LABELED (features + target_* columns), the largest split
  2. predict `validation` — target columns BLANKED: the PRACTICE board that runs
                            before round 1. Answers stay server-side, always
  3. then N sealed rounds — each open round is served on `live`. FOUR is one
                            event's configuration, not a rule: read the clock

The objective is to predict the held-out target better than the field.

This script runs the whole loop once: orient, download, fit a LightGBM baseline,
predict the split that is currently scored, and submit it down the correct lane.

It produces:
  - hackathon_model.pkl        — the fitted model (the event lane REQUIRES it)
  - hackathon_predictions.parquet — id + prediction

Usage:
    pip install "everestapi>=0.3.32" lightgbm scikit-learn pandas pyarrow
    export EIQ_API_KEY=...                  # from your event onboarding
    export EIQ_BASE_URL=https://app.everesteer.ai
    python starter.py

Connecting: production needs only your API key. A staging or preview host also
sits behind Cloudflare Access and will bounce an API-key-only request at the edge
(a 302 to a login page, or error 1010) — which does not look like an auth failure.
The SDK handles it for you: set CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET in
the environment (or pass cf_access_client_id= / cf_access_client_secret= to
EverestAPI) and the service-token headers ride alongside your key. Interactive
`cloudflared access login <host>` also works.
"""

from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import pandas as pd
from everestapi import EverestAPI

client = EverestAPI(
    api_key=os.environ["EIQ_API_KEY"],
    base_url=os.environ.get("EIQ_BASE_URL", "https://app.everesteer.ai"),
    tournament="futures",
)


def bail(step: str, exc: Exception) -> None:
    """Fail with something you can act on instead of a traceback."""
    print(f"\n{step} failed: {exc}")
    print("If this is a staging or preview host, check the Cloudflare Access note")
    print("in this file's docstring — an edge bounce is not an auth failure.")
    sys.exit(1)


# =====================================================================
# 1. Orient — get_started is the authority on what to do right now
# =====================================================================
# It is mode-aware: it answers for YOUR key. Two things to know about the
# hackathon payload:
#
#   * `live_round` is null BY DESIGN. It means "no live public tournament round"
#     and is the hackathon/tournament discriminator. Do NOT branch on it.
#     Branch on `cadence.open_window`, which is the round signal.
#   * `uploads_remaining` is how many uploads you have LEFT, not your cap. The
#     cap is a per-EVENT pool: it counts across every model and every round and
#     does NOT replenish when a new round opens. Never hardcode a number.
try:
    started = client.get_started()
except Exception as exc:  # noqa: BLE001 — first network call, report it plainly
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

# =====================================================================
# 2. Schema — read the target and features, never hardcode them
# =====================================================================
try:
    schema = client.get_dataset_schema()
except Exception as exc:  # noqa: BLE001
    bail("get_dataset_schema", exc)

targets = schema.get("targets") or []
target_col = targets[0] if targets else "target_everest_20"
print(f"\nTarget: {target_col}   (schema advertises {len(targets)} target(s))")

# =====================================================================
# 3. Download — train, plus whichever split is scored right now
# =====================================================================
# A round open means `live` is the scored split. Between rounds (and before
# round 1 is published) `live` 404s CLEANLY: that means "no round open right
# now", never "this split does not exist for my key".
try:
    train_path = client.download_dataset(split="train")
except Exception as exc:  # noqa: BLE001
    bail("download_dataset(split='train')", exc)

scored_split = "live" if round_open else "validation"
try:
    scored_path = client.download_dataset(split=scored_split)
except Exception as exc:  # noqa: BLE001
    if scored_split == "live":
        print(f"\nlive is not being served right now ({exc}).")
        print("No round is open — practising on the validation board instead.")
        scored_split = "validation"
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
# Feature values are ENCODED: cross-sectional quintile bins 0-4. A value of
# -1.0 means MISSING (the source was not onboarded for that instrument/date) —
# treat it as NaN or as its own category, NEVER as an ordinal below 0.
# A NaN target means the row was uncomputable; it is never imputed, so drop
# those rows rather than filling them.
feat_cols = sorted(c for c in train.columns if c.startswith("feature_"))
fit = train.dropna(subset=[target_col])
print(f"\nFitting on {len(fit):,} labeled rows over {len(feat_cols)} features...")

model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)
model.fit(fit[feat_cols].fillna(0.0), fit[target_col])

# =====================================================================
# 5. Predict — the id is the PARQUET INDEX, not a column
# =====================================================================
# The downloaded parquet has NO 'id' column. The id every submit lane wants is
# the index (its name is 'id'), and its values are opaque strings like
# 'eiq_559073fecf705ae5'. Submit them VERBATIM. Renumbering them 0..N-1 produces
# a submission that matches ZERO rows.
predictions = pd.DataFrame(
    {"prediction": model.predict(scored[feat_cols].fillna(0.0))},
    index=scored.index,
).reset_index()
if predictions.columns[0] != "id":
    predictions = predictions.rename(columns={predictions.columns[0]: "id"})
print(f"Predicted {len(predictions):,} rows; first id: {predictions['id'].iloc[0]!r}")

pred_path = "hackathon_predictions.parquet"
predictions.to_parquet(pred_path, index=False)

# =====================================================================
# 6. Pickle the model — the event lane requires it
# =====================================================================
model_path = "hackathon_model.pkl"
with open(model_path, "wb") as fh:
    pickle.dump(model, fh)
print(f"Wrote {pred_path} and {model_path}")

# =====================================================================
# 7. Sanity-check the payload before spending an upload
# =====================================================================
# Note: client.validate_submission() is the TOURNAMENT pre-flight — it takes a
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
# Re-read get_started IMMEDIATELY before submitting. A round can open or close
# while you were fitting, and the lane follows the clock, not your intent.
#
# From the SDK's own warning on these two calls:
#   "The two take the same arguments and their ids are NOT interchangeable:
#    sending a round's predictions down the validation lane is accepted (202)
#    and then fails minutes later on zero id overlap, costing you the
#    submission and settling anything staked on that model at nothing."
#
# submit_futures_predictions is the TOURNAMENT lane and never applies here.
# A model must EXIST before you can submit predictions or upload a .pkl for it —
# the platform never auto-creates one, and submitting to a name it does not know
# comes back as a 404 telling you to create it first. Reuse the same model across
# rounds so its board history stays on one entry.
MODEL_NAME = os.environ.get("EIQ_MODEL_ID", "hackathon-baseline")
try:
    listed = client.get_models() or {}
    existing = {m.get("name"): m.get("id") for m in (listed.get("models") or [])}
except Exception:  # noqa: BLE001 — fall through to create
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
if now_cadence.get("intake_fenced"):
    print("\nIntake is fenced around a round boundary — wait for the next phase.")
    sys.exit(0)

try:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if now_cadence.get("open_window"):
        print("\nA sealed round is OPEN — submitting to the event lane.")
        result = client.submit_event_predictions(
            MODEL_ID, predictions, model_pkl=model_path, model_pkl_python_version=pyver
        )
    else:
        # BOTH hackathon lanes require the .pkl. The SDK docstring calls model_pkl
        # the thing that distinguishes the event lane, but a hackathon key is
        # refused on the practice board without it too:
        #   400 "A model .pkl file is required for hackathon submissions."
        print("\nNo round open — submitting to the validation practice board.")
        result = client.submit_validation_diagnostics(
            MODEL_ID, predictions, model_pkl=model_path, model_pkl_python_version=pyver
        )
except Exception as exc:  # noqa: BLE001
    bail("submit", exc)

print(f"Accepted: {result}")

# =====================================================================
# 9. Where your score shows up
# =====================================================================
print("\nNext:")
print("  client.get_diagnostics_leaderboard()  — the board for a round")
print("     (pass scoring_window to read a specific round's board)")
print("  client.get_diagnostics_standings()    — cumulative standings across rounds")
print("  client.get_event_staking()            — the authority on whether this")
print("     event carries money, and whether staking is enabled for you")
print("\nBoards rank on the round score: a weighted blend of CORR, AIMC and NCORR,")
print("clipped per round. Call explain_scoring for the live weights and do not")
print("assume which term dominates.")
