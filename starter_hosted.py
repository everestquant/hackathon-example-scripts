#!/usr/bin/env python3
"""
Everesteer Hackathon, Hosted-Training Starter

A LightGBM baseline fitted on Everesteer's servers with the platform `train`
call, then checked and packaged by THIS script. The server is used as the
engine that does the fitting, and nothing else: every number you act on and
every file you upload is produced here, from your own code.

  - The fit stops before your holdout. A `train` job fits the whole train
    split unless told otherwise, which would put your holdout inside the fit.
    `train_filter` ends it at the first exped of the embargo instead.
  - The score is computed here, on that holdout, not read off the job. The
    job's own CV numbers are measured differently and its AIMC is an estimate
    against a proxy, so they are not what the board will say.
  - The upload is wrapped here. What a job hands back has not always been the
    one shape the upload gate accepts, so this script never uploads it as-is.

In a hackathon your event compute grant is already spendable (`get_started`
reports `hosted_train_funded`). A CPU LightGBM job costs a few cents.

This produces:
  - hosted_model.pkl:              the model file the job returned, as downloaded
  - hosted_model_callable.pkl:     the wrapped predict() callable that gets uploaded
  - hosted_predictions.parquet:    predictions file (id + prediction)

Usage:
    pip install "everestapi>=0.3.32" lightgbm pandas pyarrow cloudpickle
    export EIQ_API_KEY=...                 # from onboarding
    export EIQ_BASE_URL=https://app.everesteer.ai
    python starter_hosted.py
"""

from __future__ import annotations

import os
import pickle
import sys

import cloudpickle
import pandas as pd
from everestapi import EverestAPI

client = EverestAPI(
    api_key=os.environ["EIQ_API_KEY"],
    base_url=os.environ.get("EIQ_BASE_URL", "https://app.everesteer.ai"),
    tournament="futures",
)

HOLDOUT_EXPEDS = 100   # the tail of train, kept out of the fit and scored here
EMBARGO = 20           # expeds dropped in front of the holdout. A wide round number:
                       # the target's horizon is not published, so err wide.
EXPED = "exped"

# =====================================================================
# 1. Check the budget (hackathon grants are spendable immediately)
# =====================================================================
credits = client.get_compute_credits()
available = credits.get("available_cents", 0)
print(f"Compute available: ${available / 100:.2f}")
if available <= 0:
    raise SystemExit(
        "No spendable compute balance, in a hackathon the event grant funds this; "
        "outside one, top up your compute credits before running this again."
    )

# =====================================================================
# 2. Carve the holdout, and work out where the fit has to stop
# =====================================================================
# The graded column comes from the schema: it differs between datasets and is
# not necessarily the first entry in `targets`.
schema = client.get_dataset_schema() or {}
TARGET = schema.get("primary_target")
if not TARGET:
    raise SystemExit("The schema declares no primary_target - refusing to guess.")

train = pd.read_parquet(client.download_dataset(split="train"))
# train's exped tokens are zero-padded, so their string order is their time order,
# which is also the order the server's cutoff compares on.
ordered = sorted(train[EXPED].unique())
holdout_expeds = set(ordered[-HOLDOUT_EXPEDS:])
FIRST_EMBARGO_EXPED = ordered[-(HOLDOUT_EXPEDS + EMBARGO)]
holdout = train[train[EXPED].isin(holdout_expeds)].dropna(subset=[TARGET])
print(f"Holdout: last {HOLDOUT_EXPEDS} expeds ({len(holdout):,} rows). "
      f"The fit stops before {FIRST_EMBARGO_EXPED}, leaving a {EMBARGO}-exped embargo.")

# =====================================================================
# 3. Launch the hosted training job, fenced off from the holdout
# =====================================================================
# gpu="CPU" is the cheapest tier and the right one for tree models; the default
# is T4. features="all" is passed explicitly because the default is "small",
# which on a dataset that publishes a single set is an alphabetical prefix of
# the feature list rather than a curated subset.
#
# No `target=`: the platform trains on whichever column the dataset declares as
# graded. Pass `target=` only when you deliberately want an AUXILIARY target.
print("Launching hosted train (LightGBM, CPU)...")
job = client.train(
    model="lightgbm",
    features="all",
    gpu="CPU",
    train_filter={"exped": {"cutoff_lt": FIRST_EMBARGO_EXPED}},
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
print(f"  Job {job_id} submitted. Polling (a CPU baseline takes a few minutes)...")

status = client.wait_for_job(job_id, timeout=3600)
if status["status"] != "completed":
    raise SystemExit(
        f"Job ended {status['status']}: {status.get('error_message')} "
        f". See client.get_job_log({job_id!r}) for the lifecycle trail."
    )
# The job also reports its own CV metrics under status["metrics"]. They are not
# used here: they are measured inside the job's folds, and its AIMC is an
# estimate against a proxy. The score that decides anything is step 5's.

# =====================================================================
# 4. Download the model, and make one function that scores a frame
# =====================================================================
pkl_path = client.download_model(job_id, "hosted_model.pkl")
# Safe to unpickle: this is YOUR OWN model artifact from YOUR authenticated
# train job (presigned, ownership-gated download): not third-party data.
with open(pkl_path, "rb") as f:
    fitted = pickle.load(f)

# The manifest is the record of what the model was fit on: the ordered feature
# columns and the preprocessing the trainer applied. Reproduce that preprocessing
# exactly. In particular, do NOT turn the missing value (-1) into NaN here: the
# trainer fed -1 to the model as an ordinary value, so the model learned it that
# way, and changing it at predict time would hand the model inputs it never saw.
manifest = status.get("feature_manifest") or {}
FEATURES = manifest["features"]
FILL = float(manifest.get("fill_value", 0.0))


def make_scorer(model, columns, fill):
    """Score a frame, whichever shape of artifact the job returned.

    A bare estimator takes a positional array, so columns are selected by name
    and in the manifest's order. A callable takes the frame and returns a
    one-column frame. `columns` and `fill` are bound by value so the pickled
    wrapper in step 7 carries them with it.
    """
    if hasattr(model, "predict"):
        def score(frame):
            arr = frame[columns].fillna(fill).to_numpy("float32")
            return pd.Series(model.predict(arr), index=frame.index)
    elif callable(model):
        def score(frame):
            out = model(frame)
            out = out.iloc[:, 0] if isinstance(out, pd.DataFrame) else pd.Series(out)
            return pd.Series(out.to_numpy(), index=frame.index)
    else:
        raise SystemExit(f"Unrecognised artifact type: {type(model).__qualname__}")
    return score


score = make_scorer(fitted, FEATURES, FILL)
print(f"Artifact: {type(fitted).__qualname__} "
      f"({'estimator' if hasattr(fitted, 'predict') else 'callable'}), {len(FEATURES)} features")

# =====================================================================
# 5. Score it on the holdout, here
# =====================================================================
# CORR is the rank correlation between predictions and the target, computed
# within each exped. Pearson on ranks is Spearman.
def per_exped_corr(frame, pred_col):
    return frame.groupby(EXPED)[[pred_col, TARGET]].apply(
        lambda g: g[pred_col].rank().corr(g[TARGET].rank())
    ).dropna()


holdout = holdout.assign(prediction=score(holdout))
corr = per_exped_corr(holdout, "prediction")
print(f"Holdout CORR {corr.mean():+.4f} | std {corr.std():.4f} | "
      f"sharpe {corr.mean() / corr.std():.2f} | {(corr > 0).mean():.0%} of expeds positive")

# The benchmark on the same rows, scored the same way, so the comparison is
# like-for-like. Only its `train` split is served during an event. A row is the
# same row only when its id AND its exped agree: a benchmark built from an older
# train file can share ids with this one on different expeds, and an id-only
# join would quietly score it against the wrong rows.
try:
    bench = pd.read_parquet(client.download_benchmark("futures", "train"))
    bench_mean = pd.DataFrame({"benchmark": bench.drop(columns=[EXPED]).mean(axis=1),
                               "bench_exped": bench[EXPED]})
    rows = holdout.join(bench_mean, how="inner")
    rows = rows[rows["bench_exped"] == rows[EXPED]]
    if len(rows) < 0.5 * len(holdout):
        print(f"Benchmark comparison skipped: only {len(rows):,} of {len(holdout):,} holdout "
              "rows match on id and exped, so the benchmark was built from a different "
              "train file than the one served now.")
    else:
        print(f"Benchmark    {per_exped_corr(rows, 'benchmark').mean():+.4f} "
              f"on {len(rows):,} of {len(holdout):,} holdout rows")
except Exception as exc:  # noqa: BLE001, a comparison is useful but not required
    print(f"Benchmark comparison unavailable: {exc}")

# This model never saw the embargo or the holdout. That is the price of an
# honest score: about 2% of the history. To submit a model fit on all of it,
# launch a second job without train_filter once you are happy with this one.

# =====================================================================
# 6. Predict on the SERVED scored split
# =====================================================================
# In a multi-round cadence event the open round is served as `live`
# (get_started's cadence object names it); before round 1, the scored split is
# `validation`. `split_window` records WHICH round these rows came from. Step 8
# needs it: the submit call is decided by the rows in hand, and that is a fact
# about this download, not about whatever the clock says once they are ready.
cadence = (client.get_started() or {}).get("cadence") or {}
split = "live" if cadence.get("open_window") else "validation"
split_window = cadence.get("open_window") if split == "live" else None
if cadence.get("phase") == "done":
    raise SystemExit("The event is over (phase 'done'): no round or practice upload is accepted now.")
# cadence.intake_fenced fences ROUND submissions only (409 while a round opens or
# closes). It is also true in build, in stake and between rounds, where the
# practice board takes uploads regardless, so it only matters while a round is
# named, and then `live` itself is not served yet (409 cadence_not_open). Say so
# rather than failing on the download for a reason that looks like a bug.
if split == "live" and cadence.get("intake_fenced"):
    raise SystemExit(
        f"Round {split_window!r} is not taking predictions right now (phase "
        f"{cadence.get('phase')!r}): `live` is served and uploads are accepted once "
        "cadence.intake_fenced goes false. Poll client.get_status() and re-run."
    )
if split == "validation" and cadence.get("diagnostics_maintenance"):
    print("  NOTE: the practice board is briefly down for maintenance; retry the submit later.")
print(f"Downloading the served {split} split and predicting locally...")
served = pd.read_parquet(client.download_dataset(split=split))
served_ids = served["id"] if "id" in served.columns else served.index   # the id IS the index
submission = pd.DataFrame({"prediction": score(served).to_numpy()},
                          index=pd.Index(served_ids, name="id"))
submission.to_parquet("hosted_predictions.parquet")
print(f"  {len(submission):,} predictions written to hosted_predictions.parquet")


# =====================================================================
# 7. Wrap the model in the one shape the upload gate accepts
# =====================================================================
# Everesteer runs one model shape: a cloudpickled predict() callable returning a
# single-column DataFrame indexed by instrument id. Anything else is refused with
#   400 "Everesteer runs one model shape: a cloudpickled callable."
# The wrapper reuses step 4's scorer, so it selects features by name and applies
# the same preprocessing the model was fit with, whatever the job returned.
def build_predict(scorer):
    def predict(live_features, live_benchmark_models=None):
        return pd.DataFrame({"prediction": scorer(live_features).to_numpy()},
                            index=live_features.index)

    return predict


upload_pkl = "hosted_model_callable.pkl"
with open(upload_pkl, "wb") as f:
    cloudpickle.dump(build_predict(score), f)

# =====================================================================
# 8. Submit with the call THESE ROWS belong to. A round sent to the
#    practice board matches none of that round's ids and settles at $0,
#    and so does the reverse, so the call follows `split` from step 6 and
#    the clock re-read below is only a guard on whether it is still valid.
#    Several models at once: the submit_event_predictions_batch MCP TOOL
#    (there is no batch method on the Python client - loop this call).
# =====================================================================
# create_model is idempotent WHEN YOU PASS A NAME: a 409 for a name you already
# own resolves to the existing record with status="already_exists". Either way
# read the response's `id`. That is the stable model_id every submit call wants;
# `name` is a mutable display label, and passing it where a model_id belongs is
# a 403 "Model not owned by caller".
MODEL_NAME = "hosted-lgbm-baseline"
MODEL_ID = client.create_model(name=MODEL_NAME)["id"]
print(f"Model {MODEL_NAME!r} -> {MODEL_ID}")

# Re-read the clock. Minutes passed while the job trained and the split
# downloaded. This does NOT choose the call -- `split` already did, back when the
# rows were fetched -- it only answers whether those rows are still submittable.
now_cadence = (client.get_started() or {}).get("cadence") or {}
now_window = now_cadence.get("open_window")
pyver = f"{sys.version_info.major}.{sys.version_info.minor}"

if now_cadence.get("phase") == "done":
    raise SystemExit("The event ended while this ran: nothing is accepted any more.")
if now_window and now_cadence.get("intake_fenced"):
    raise SystemExit(
        f"Round {now_window!r} is opening or closing (phase {now_cadence.get('phase')!r}) "
        "and not taking predictions. Poll client.get_status() and re-run the submit."
    )
if split == "live" and now_window != split_window:
    raise SystemExit(
        f"These rows are round {split_window!r}; the open round is now {now_window!r}. "
        "Re-run: download `live` again for the open round and predict on it. Sending them "
        "to the practice board instead would match zero ids and score nothing."
    )
if split == "validation" and now_window:
    raise SystemExit(
        f"Round {now_window!r} opened while this ran; these are practice-board rows. "
        "Re-run to enter the round: the practice board is display-only and is never ranked."
    )
if split == "validation" and now_cadence.get("diagnostics_maintenance"):
    raise SystemExit(
        "The practice board is briefly down for maintenance. Nothing was uploaded: "
        "re-run the submit in a few minutes."
    )

if split == "live":
    print(f"Round {split_window} is open and these are its rows: submitting to the round.")
    result = client.submit_event_predictions(
        MODEL_ID,
        "hosted_predictions.parquet",
        target=TARGET,  # the SDK default, target_everest_20, is not this dataset's column
        model_pkl=upload_pkl,  # required: a CLOUDPICKLED predict() callable
        model_pkl_python_version=pyver,
    )
else:
    print("These are practice-board rows: submitting to the practice board.")
    try:
        result = client.submit_validation_diagnostics(
            MODEL_ID,
            "hosted_predictions.parquet",
            target=TARGET,
            model_pkl=upload_pkl,
            model_pkl_python_version=pyver,
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "status_code", None) != 503:
            raise
        raise SystemExit(
            f"The practice board is briefly down for maintenance ({exc}). "
            "Re-run the submit in a few minutes."
        ) from exc
print(f"Scored: {result}")
print("Check your standing with client.get_diagnostics_leaderboard().")
