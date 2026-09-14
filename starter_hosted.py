#!/usr/bin/env python3
"""
Everesteer Hackathon — Hosted-Training Starter

A LightGBM baseline trained on Everesteer's hosted compute via the platform
`train` tool: the platform loads the data, runs exped-purged/embargoed
cross-validation, computes canonical tournament metrics (CORR / AIMC estimate
/ NCORR), and returns the model .pkl + a feature manifest. In a hackathon
your event compute grant is already spendable (`get_started` reports
`hosted_train_funded`) — a CPU LightGBM baseline holds well under $1.

Submission uses the always-correct path: download the .pkl, predict LOCALLY
on whichever split the platform is currently scoring, and submit that.
(The job's own predictions artifact is scored against the tree the trainer
read — predicting on the served split yourself can never id-mismatch.)

This produces:
  - hosted_model.pkl              — the platform-trained model
  - hosted_predictions.parquet    — predictions file (id + prediction)

Usage:
    pip install "everestapi>=0.3.32" lightgbm pandas pyarrow
    export EIQ_API_KEY=...                 # from onboarding
    export EIQ_BASE_URL=https://app.everesteer.ai
    python starter_hosted.py
"""

from __future__ import annotations

import os
import pickle
import sys

import pandas as pd
from everestapi import EverestAPI

client = EverestAPI(
    api_key=os.environ["EIQ_API_KEY"],
    base_url=os.environ.get("EIQ_BASE_URL", "https://app.everesteer.ai"),
    tournament="futures",
)

# =====================================================================
# 1. Check the budget (hackathon grants are spendable immediately)
# =====================================================================
credits = client.get_compute_credits()
available = credits.get("available_cents", 0)
print(f"Compute available: ${available / 100:.2f}")
if available <= 0:
    raise SystemExit(
        "No spendable compute balance — in a hackathon the event grant funds this; "
        "outside one, top up your compute credits before running this again."
    )

# =====================================================================
# 2. Launch the hosted training job
# =====================================================================
# Mirrors the local starter's LightGBM config. gpu="CPU" is the cheapest
# tier and the right choice for tree models; the platform runs exped-purged
# CV with the tournament embargo — no leakage bookkeeping on your side.
print("Launching hosted train (LightGBM, CPU)...")
job = client.train(
    model="lightgbm",
    features="all",
    target="target_everest_20",
    gpu="CPU",
    params={
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "num_leaves": 64,
        "colsample_bytree": 0.10,
        "subsample": 0.80,
        "min_child_samples": 500,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    },
)
job_id = job["job_id"] if "job_id" in job else job["id"]
print(f"  Job {job_id} submitted — polling (a CPU baseline takes a few minutes)...")

status = client.wait_for_job(job_id, timeout=3600)
if status["status"] != "completed":
    raise SystemExit(
        f"Job ended {status['status']}: {status.get('error_message')} "
        f"— see client.get_job_log({job_id!r}) for the lifecycle trail."
    )

# =====================================================================
# 3. Read the platform-computed CV metrics
# =====================================================================
metrics = status.get("metrics") or {}
cv = metrics.get("cv") or {}
for name in ("corr", "aimc_estimate", "ncorr"):
    panel = cv.get(name) or {}
    if "mean" in panel:
        print(f"  CV {name:>14}: {panel['mean']:+.4f} (std {panel.get('std', 0):.4f})")
# aimc_estimate is a pre-submission estimate vs a consensus proxy — the real
# AIMC comes from the diagnostics scoring after you submit.

# =====================================================================
# 4. Download the model and predict on the SERVED scored split
# =====================================================================
pkl_path = client.download_model(job_id, "hosted_model.pkl")
# Safe to unpickle: this is YOUR OWN model artifact from YOUR authenticated
# train job (presigned, ownership-gated download) — not third-party data.
with open(pkl_path, "rb") as f:
    model = pickle.load(f)

manifest = status.get("feature_manifest") or {}
feature_order = manifest["features"]

# Predict on the split the platform is CURRENTLY scoring. In a multi-round
# cadence event the open round is served as `live` (get_started's cadence
# object names it); before round 1 — and in a single-window event — the
# scored split is `validation`. Tournament keys have no cadence object and
# also land on `validation` here (this script's submit half is the
# diagnostics loop, not a live-round submission).
cadence = (client.get_started() or {}).get("cadence") or {}
split = "live" if cadence.get("open_window") else "validation"
# Uploads are refused (409) while a round settles and the next opens. Say so
# rather than letting the submit at the end fail for a reason that looks like a
# bad file -- cadence.intake_fenced is the platform telling you to wait.
if cadence.get("intake_fenced"):
    print(
        f"  NOTE: intake is fenced right now (phase {cadence.get('phase')!r}) -- a round "
        "is settling. Predictions are still worth building, but the upload will be "
        "refused until cadence.intake_fenced goes false: poll client.get_status() "
        "and re-run the submit."
    )
print(f"Downloading the served {split} split and predicting locally...")
val = pd.read_parquet(client.download_dataset(split=split))
# The .pkl is a bare estimator trained on a positional array, so the manifest's
# column order is authoritative — a different order returns plausible-but-wrong
# predictions rather than an error.
#
# The trainer applies NO missing-value transform: features are quintile bins
# 0..4 and -1 IS the missing bin, so missingness is bin-coded and the served
# split carries no NaNs at all. The fill below is therefore a no-op today and
# exists only so a malformed frame fails predictably. It fills with -1, NOT
# 0.0: 0 is a real quintile, so filling with it would silently recode "missing"
# as "lowest bin" the moment a NaN did appear.
x = val[feature_order].fillna(-1.0).to_numpy("float32")
val_ids = val["id"] if "id" in val.columns else val.index
submission = pd.DataFrame({"prediction": model.predict(x)}, index=pd.Index(val_ids, name="id"))
submission.to_parquet("hosted_predictions.parquet")
print(f"  {len(submission):,} predictions written to hosted_predictions.parquet")

# =====================================================================
# 5. Submit down the lane the clock says is open. A round sent down the
#    practice lane matches none of a round's ids and settles at $0, so
#    the cadence below is re-read rather than trusted from step 4.
#    Several models: submit_event_predictions_batch.
# =====================================================================
MODEL_NAME = "hosted-lgbm-baseline"
client.create_model(MODEL_NAME)  # idempotent — 409 "already exists" is fine

# Re-read the clock. Minutes passed while the job trained and the split
# downloaded, and the lane follows the clock, not the read you took back then.
now_cadence = (client.get_started() or {}).get("cadence") or {}
pyver = f"{sys.version_info.major}.{sys.version_info.minor}"

if now_cadence.get("open_window"):
    print("A sealed round is OPEN — submitting to the event lane.")
    result = client.submit_event_predictions(
        MODEL_NAME,
        "hosted_predictions.parquet",
        model_pkl=pkl_path,  # required on both lanes (store-only, never executed)
        model_pkl_python_version=pyver,
    )
else:
    print("No round open — submitting to the validation practice board.")
    result = client.submit_validation_diagnostics(
        MODEL_NAME,
        "hosted_predictions.parquet",
        model_pkl=pkl_path,
        model_pkl_python_version=pyver,
        wait=True,
    )
print(f"Scored: {result}")
print("Check your standing with client.get_diagnostics_leaderboard().")
