"""
DIMER Fine-tuner — Mitra tabular classification (Custom / Other task type)
==========================================================================
Fits AutoGluon's Mitra on the validated tabular dataset, evaluates on a holdout,
saves the predictor as the model artifact, and writes result.json.

Design points:
  - single model, no ensembling  (fit_weighted_ensemble=False, hyperparameters={"MITRA": {}})
  - infer binary vs multiclass from the target's cardinality so AutoGluon loads the
    classifier checkpoint; enforce Mitra's 10-class ceiling
  - assert the requested model actually trained (AutoGluon can skip a model and still
    report generic success)
  - seed every RNG (Mitra's fit is stochastic; a fixed seed makes runs reproducible)
  - 10,000-row training ceiling (an upstream limit of Mitra)

DIMER contract:
  - dataset at DIMER_DATASET_DIR (raw zip or dir)
  - write the trained model under DIMER_OUTPUT_DIR
  - write result.json to DIMER_RESULT_PATH
  - POST DIMER_DONE_CALLBACK when done
  - DIMER_TRAIN_DEVICE = "cuda:0" | "cpu"
  - DIMER_HYPERPARAMETERS_JSON / DIMER_PREPROCESSING_ARGS_JSON = the dimer-pipeline.json fields
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

TEMPLATE_NAME = "mitra-classifier-finetuner"

# Custom / Other normalizes taskType to object_detection upstream — bake our own default
# into the image and treat the pipeline value as an override, never as the truth.
DEFAULT_TASK_TYPE = os.getenv("DIMER_TASK_TYPE", "tabular_classification")

DATASET_DIR = Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
OUTPUT_DIR = Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output"))
RESULT_PATH = Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json"))
DONE_CALLBACK = os.getenv("DIMER_DONE_CALLBACK", "").strip()
CALLBACK_TIMEOUT_SECONDS = float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10"))
TRAIN_DEVICE = os.getenv("DIMER_TRAIN_DEVICE", "cuda:0").strip()

PIPELINE_METADATA = json.loads(os.getenv("DIMER_PIPELINE_METADATA_JSON", "{}") or "{}")
HYPERPARAMS = json.loads(os.getenv("DIMER_HYPERPARAMETERS_JSON", "{}") or "{}")
PREPROCESSING = json.loads(os.getenv("DIMER_PREPROCESSING_ARGS_JSON", "{}") or "{}")

# Preprocessing fields (see dimer-pipeline.json)
TARGET_COLUMN = str(PREPROCESSING.get("target_column") or "target").strip()
DROP_COLUMNS = [
    c.strip() for c in str(PREPROCESSING.get("drop_columns") or "").split(",") if c.strip()
]
MAX_TRAIN_ROWS = int(PREPROCESSING.get("max_train_rows") or 10_000)
VALIDATION_SPLIT = float(PREPROCESSING.get("validation_split") or 0.2)

# Hyperparameter fields
TIME_LIMIT = int(HYPERPARAMS.get("time_limit_seconds") or 600)
SEED = int(HYPERPARAMS.get("seed") or 0)
EVAL_METRIC = str(HYPERPARAMS.get("eval_metric") or "accuracy").strip()
_ft = HYPERPARAMS.get("fine_tune", True)
FINE_TUNE = _ft if isinstance(_ft, bool) else str(_ft).strip().lower() in ("true", "1", "yes")
FINE_TUNE_STEPS = int(HYPERPARAMS.get("fine_tune_steps") or 0)  # 0 = AutoGluon default

MITRA_MODEL_KEY = "MITRA"
MITRA_ROW_LIMIT = 10_000  # hard upstream ceiling
MITRA_CLASS_LIMIT = 10    # Mitra classifies at most 10 classes
MIN_ROWS_FOR_SPLIT = 20   # below this, don't carve a holdout out of train

BASE_MODEL = "autogluon/mitra-classifier"
# Pinned weights revision (also baked in Dockerfile Option A). The revision actually used is
# resolved at runtime and recorded next to this expected value so any drift is visible.
PINNED_MITRA_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    RESULT_PATH.write_text(content, encoding="utf-8")


def notify_done_callback() -> dict[str, Any]:
    if not DONE_CALLBACK:
        return {"attempted": False, "message": "DIMER_DONE_CALLBACK not set; skipping."}
    parsed = urlparse(DONE_CALLBACK)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "message": f"Unsupported scheme: {parsed.scheme}"}
    try:
        response = requests.post(DONE_CALLBACK, timeout=CALLBACK_TIMEOUT_SECONDS)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _read_csv_from_dataset(stem: str) -> pd.DataFrame | None:
    """Read <stem>.csv from the raw zip or an unzipped directory."""
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as zf:
            for member in zf.namelist():
                p = Path(member.lstrip("./"))
                if p.suffix.lower() == ".csv" and p.stem.lower() == stem:
                    with zf.open(member) as handle:
                        return pd.read_csv(io.BytesIO(handle.read()))
        return None
    for path in sorted(DATASET_DIR.rglob("*.csv")):
        if path.stem.lower() == stem:
            return pd.read_csv(path)
    return None


def _prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, val) with target present, drop_columns removed, capped to the ceiling."""
    train = _read_csv_from_dataset("train")
    if train is None:
        raise FileNotFoundError("no train.csv in the dataset (validator should have caught this)")
    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"target column '{TARGET_COLUMN}' not in {list(train.columns)}")

    drop = [c for c in DROP_COLUMNS if c in train.columns]
    train = train.drop(columns=drop)
    train = train.dropna(subset=[TARGET_COLUMN])

    val = _read_csv_from_dataset("val")
    if val is not None:
        val = val.drop(columns=[c for c in DROP_COLUMNS if c in val.columns])
        val = val.dropna(subset=[TARGET_COLUMN])
    else:
        # Deterministic holdout split; report it (never train with no validation silently).
        val_frac = min(max(VALIDATION_SPLIT, 0.0), 0.4)
        if val_frac > 0 and len(train) > MIN_ROWS_FOR_SPLIT:
            val = train.sample(frac=val_frac, random_state=SEED)
            train = train.drop(index=val.index)
        else:
            val = pd.DataFrame(columns=train.columns)

    ceiling = min(MAX_TRAIN_ROWS, MITRA_ROW_LIMIT)
    if len(train) > ceiling:
        log(f"Sampling train {len(train)} -> {ceiling} rows (seed={SEED}).")
        train = train.sample(n=ceiling, random_state=SEED)

    return train.reset_index(drop=True), val.reset_index(drop=True)


def _fit_and_evaluate(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
    # Resolve the effective device. One image runs on GPU or CPU: use the GPU only when the
    # platform did not ask for CPU and a GPU is actually present at runtime. An explicit CPU
    # request hides the GPU before torch initializes.
    requested_cpu = TRAIN_DEVICE.lower() == "cpu"
    if requested_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _seed_everything(SEED)
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False
    use_gpu = gpu_available and not requested_cpu

    # Fine-tuning Mitra requires a GPU: on CPU the backward pass uses a low-precision path
    # many CPUs do not support. Without a usable GPU, run zero-shot in-context inference,
    # which is CPU-safe.
    fine_tune = FINE_TUNE
    if not use_gpu and fine_tune:
        why = "no GPU is available" if not gpu_available else "CPU was requested"
        log(f"Running zero-shot (fine_tune=False): {why}; Mitra fine-tuning requires a GPU.")
        fine_tune = False
    mitra_hp: dict[str, Any] = {"fine_tune": fine_tune}
    if fine_tune and FINE_TUNE_STEPS:
        mitra_hp["fine_tune_steps"] = FINE_TUNE_STEPS

    from autogluon.tabular import TabularPredictor

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor_path = OUTPUT_DIR / "mitra_predictor"

    # Infer the problem type from the target's cardinality. AutoGluon selects the Mitra
    # classifier checkpoint for binary/multiclass; enforce Mitra's 10-class ceiling here so
    # an over-wide target fails fast with a clear message rather than deep in the fit.
    num_classes = int(train[TARGET_COLUMN].nunique(dropna=True))
    if num_classes < 2:
        raise ValueError(
            f"target '{TARGET_COLUMN}' has {num_classes} distinct class(es); classification "
            f"needs at least 2."
        )
    if num_classes > MITRA_CLASS_LIMIT:
        raise ValueError(
            f"target '{TARGET_COLUMN}' has {num_classes} classes; Mitra supports at most "
            f"{MITRA_CLASS_LIMIT}. Reduce the number of classes (e.g. merge rare labels)."
        )
    problem_type = "binary" if num_classes == 2 else "multiclass"

    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type=problem_type,
        eval_metric=EVAL_METRIC,
        path=str(predictor_path),
        verbosity=2,
    )
    predictor.fit(
        train,
        hyperparameters={MITRA_MODEL_KEY: mitra_hp},
        fit_weighted_ensemble=False,
        time_limit=TIME_LIMIT,
    )

    # Silent-substitution guard. AutoGluon can skip a model (for example when its projected
    # memory exceeds the guard threshold), log a warning, and still report generic success.
    # Refuse to report a result for a model that is not the one requested.
    trained = list(predictor.model_names())
    if not trained:
        raise RuntimeError(
            f"{MITRA_MODEL_KEY} did not train. A common cause is AutoGluon's memory guard "
            f"(it needs the projected footprint under the available-RAM threshold). Request "
            f"a larger GPU/memory profile for this pipeline, then re-run. Check the fit log "
            f"for 'Not enough memory to safely train model'."
        )
    if not any(MITRA_MODEL_KEY.split("-")[0].lower() in m.lower() for m in trained):
        raise RuntimeError(
            f"expected {MITRA_MODEL_KEY} but AutoGluon trained {trained} — refusing to "
            f"report a result for a model that was not the one requested"
        )

    metrics: dict[str, Any] = {
        "trainedModels": trained,
        "trainRows": int(len(train)),
        "numClasses": num_classes,
        "problemType": problem_type,
    }
    if len(val) > 0:
        # Compare as strings so a numeric-vs-object label dtype mismatch never inflates error.
        y_true = val[TARGET_COLUMN].astype(str).to_numpy()
        y_pred = np.asarray(predictor.predict(val.drop(columns=[TARGET_COLUMN]))).astype(str)
        accuracy = float(np.mean(y_true == y_pred))
        metrics.update({"valRows": int(len(val)), "accuracy": accuracy})
        try:
            evaluation = predictor.evaluate(val, silent=True)
            metrics["evaluation"] = {k: float(v) for k, v in evaluation.items()}
        except Exception as exc:  # noqa: BLE001 - full evaluation is best-effort
            metrics["evaluationError"] = str(exc)
        log(f"Holdout accuracy={accuracy:.4f} on {len(val)} rows ({num_classes} classes).")
    else:
        metrics["valRows"] = 0
        metrics["note"] = "No validation rows available; trained on all rows without holdout."

    metrics["artifactPath"] = str(predictor_path)
    metrics["mode"] = "fine-tune" if fine_tune else "zero-shot"
    metrics["device"] = "cuda" if use_gpu else "cpu"
    return metrics


def _dataset_sha256() -> dict[str, Any] | None:
    """SHA-256 of the uploaded dataset — the zip if present, else the CSVs in sorted order."""
    h = hashlib.sha256()
    zips = sorted(DATASET_DIR.glob("*.zip"))
    if zips:
        with open(zips[0], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"file": zips[0].name, "sha256": h.hexdigest()}
    csvs = sorted(DATASET_DIR.rglob("*.csv"))
    if not csvs:
        return None
    for p in csvs:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return {"files": [p.name for p in csvs], "sha256": h.hexdigest()}


def _resolve_base_model_revision() -> str | None:
    """The commit hash of the Mitra weights actually used: an explicit override, else the
    revision resolved from the local Hugging Face cache."""
    override = os.getenv("DIMER_MITRA_REVISION", "").strip()
    if override:
        return override
    cache = os.getenv("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    repo_dir = os.path.join(cache, "hub", "models--" + BASE_MODEL.replace("/", "--"))
    snapshots = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")))
    return os.path.basename(snapshots[-1]) if snapshots else None


def _provenance() -> dict[str, Any]:
    """Lineage record for result.json: base-model revision (resolved vs expected), a hash of
    the uploaded dataset, and the AutoGluon version. Never raises — provenance must not fail
    a run."""
    prov: dict[str, Any] = {
        "baseModel": BASE_MODEL,
        "baseModelRevisionExpected": PINNED_MITRA_REVISION,
    }
    try:
        prov["baseModelRevision"] = _resolve_base_model_revision()
    except Exception as exc:  # noqa: BLE001
        prov["baseModelRevision"] = None
        prov["baseModelRevisionError"] = str(exc)
    try:
        prov["dataset"] = _dataset_sha256()
    except Exception as exc:  # noqa: BLE001
        prov["dataset"] = {"error": str(exc)}
    ag = sys.modules.get("autogluon.tabular")
    prov["autogluonVersion"] = getattr(ag, "__version__", None)
    return prov


def run() -> int:
    train, val = _prepare_frames()
    metrics = _fit_and_evaluate(train, val)
    payload = {
        "successful": True,
        "message": (
            f"Mitra fine-tuning succeeded on {metrics['trainRows']} rows"
            + (f"; holdout accuracy {metrics['accuracy']:.4f}." if "accuracy" in metrics else ".")
        ),
        "metrics": metrics,
        "artifacts": {"modelDir": str(OUTPUT_DIR / "mitra_predictor")},
        "provenance": _provenance(),
        "metadata": {
            "template": TEMPLATE_NAME,
            "taskType": DEFAULT_TASK_TYPE,
            "baseModel": BASE_MODEL,
            "targetColumn": TARGET_COLUMN,
            "dropColumns": DROP_COLUMNS,
            "seed": SEED,
            "timeLimitSeconds": TIME_LIMIT,
            "evalMetric": EVAL_METRIC,
            "trainDevice": TRAIN_DEVICE,
        },
    }
    write_result(payload)
    log(f"Callback: {json.dumps(notify_done_callback(), sort_keys=True)}")
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": f"Mitra fine-tuning failed: {exc}",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "provenance": _provenance(),
            "metadata": {"template": TEMPLATE_NAME, "taskType": DEFAULT_TASK_TYPE},
        }
        try:
            write_result(payload)
            notify_done_callback()
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
