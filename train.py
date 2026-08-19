"""
DIMER Fine-tuner — Mitra tabular classification (Custom / Other task type)
==========================================================================
Fits AutoGluon's Mitra on the validated tabular dataset, evaluates on a holdout
(and on test.csv when present), saves the predictor, and writes result.json.

Design points:
  - single model, no ensembling (fit_weighted_ensemble=False, hyperparameters={"MITRA": {}})
  - infer binary vs multiclass from the target's cardinality; enforce Mitra's 10-class ceiling
  - class-preserving row cap and a stratified holdout split, so validation guarantees hold at
    training time (no class is lost to sampling or the split)
  - resolve and checksum-verify the base weights before fitting; refuse unexpected weights
  - assert the requested model actually trained; seed every RNG

DIMER contract:
  - dataset at DIMER_DATASET_DIR (raw zip or dir); write model under DIMER_OUTPUT_DIR;
    write result.json to DIMER_RESULT_PATH; POST DIMER_DONE_CALLBACK when done
  - DIMER_TRAIN_DEVICE = "cuda:0" | "cpu"
  - DIMER_HYPERPARAMETERS_JSON / DIMER_PREPROCESSING_ARGS_JSON = the dimer-pipeline.json fields
  - DIMER_MODEL_DIR (optional) = a directory holding uploaded model.safetensors + config.json
  - DIMER_MITRA_REVISION (optional) = required base-model revision (defaults to the pinned one)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

TEMPLATE_NAME = "mitra-classifier-finetuner"
BASE_MODEL = "autogluon/mitra-classifier"
# Pinned weights revision and its model.safetensors SHA-256. The loaded weights are verified
# against this checksum before fitting; a mismatch fails the run.
PINNED_MITRA_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
EXPECTED_WEIGHTS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
# config.json is checksum-enforced too: it carries the architecture Mitra builds before loading
# the weights, so a drifted config with matching weights would still change the model.
EXPECTED_CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"

MITRA_MODEL_KEY = "MITRA"
MITRA_ROW_LIMIT = 10_000  # hard upstream ceiling
MITRA_CLASS_LIMIT = 10    # Mitra classifies at most 10 classes
MIN_ROWS_FOR_SPLIT = 20   # below this, don't carve a holdout out of train

# ============================================================================
# CANONICAL DATASET RESOLUTION + ARCHIVE SAFETY
# Keep this block byte-identical across the validator and finetuner containers.
# The CI parity check (scripts/check_shared.py) enforces it.
# ============================================================================
def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


MAX_TOTAL_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_UNCOMPRESSED_BYTES"), 4 * 1024**3)
MAX_COMPRESSION_RATIO = _safe_float(os.getenv("DIMER_MAX_COMPRESSION_RATIO"), 200.0)
# Per-file uncompressed-byte ceiling and a full-read row ceiling. These bound memory BEFORE
# pandas materializes a table, and — unlike the zip-bomb guard — also apply to directory-mode
# inputs. An over-limit table is rejected, never silently truncated (truncation would corrupt
# the validator's row/usable counts and the finetuner's class-preservation guarantees).
MAX_MEMBER_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_MEMBER_BYTES"), 1 * 1024**3)
MAX_CSV_ROWS = _safe_int(os.getenv("DIMER_MAX_CSV_ROWS"), 5_000_000)
CSV_READ_CHUNK_ROWS = _safe_int(os.getenv("DIMER_CSV_CHUNK_ROWS"), 200_000)
_DATASET_DIR_ALIASES = {"dataset", "datasets"}


def _normalize_member(name: str) -> str | None:
    """Normalize an archive member or relative path to a canonical form, or None
    for directories/junk. Strips a single leading dataset/ or datasets/ wrapper."""
    if not name or name.endswith("/"):
        return None
    cleaned = name.replace("\\", "/").lstrip("./")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        return None
    if len(parts) > 1 and parts[0].lower() in _DATASET_DIR_ALIASES:
        parts = parts[1:]
    return "/".join(parts)


def _assert_zip_safe(zf: zipfile.ZipFile) -> None:
    """Reject pathological archives (zip bombs, oversized expansion) before any read."""
    total = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        total += info.file_size
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive member {info.filename!r} is {info.file_size:,} uncompressed bytes "
                f"(> {MAX_MEMBER_UNCOMPRESSED_BYTES:,}); refusing (per-file guard)"
            )
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                raise ValueError(
                    f"archive member {info.filename!r} expands {ratio:.0f}x "
                    f"(> {MAX_COMPRESSION_RATIO:.0f}); refusing (zip-bomb guard)"
                )
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"archive expands to {total:,} bytes (> {MAX_TOTAL_UNCOMPRESSED_BYTES:,}); "
            f"refusing to load"
        )


class DatasetSource:
    """Canonical dataset reader: a zip (preferred) or an unzipped directory.

    Resolves train/val/test.csv deterministically and rejects ambiguous archives
    (two members that resolve to the same table). Streams CSV members straight into
    pandas rather than materializing whole members in memory.
    """

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.archive_name: str | None = None
        self.source_type = "directory"
        self._zip: zipfile.ZipFile | None = None
        self._members: dict[str, list[str]] = {}  # normalized -> [raw member/path, ...]

        zips = sorted(dataset_dir.glob("*.zip"))
        if zips:
            self.archive_name = zips[0].name
            self.source_type = "zip"
            self._zip = zipfile.ZipFile(zips[0])
            _assert_zip_safe(self._zip)
            for raw in self._zip.namelist():
                nm = _normalize_member(raw)
                if nm:
                    self._members.setdefault(nm, []).append(raw)
        else:
            for p in sorted(dataset_dir.rglob("*")):
                if p.is_file():
                    key = str(p.relative_to(dataset_dir)).replace("\\", "/")
                    self._members.setdefault(key, []).append(str(p))

    @property
    def files(self) -> list[str]:
        return sorted(self._members)

    def has_nested_zip(self) -> bool:
        return any(f.lower().endswith(".zip") for f in self._members)

    def candidates(self, stem: str) -> list[str]:
        return [
            f for f in self.files
            if f.lower().endswith(".csv") and Path(f).stem.lower() == stem
        ]

    def duplicate_raw_count(self, stem: str) -> int:
        """Total raw members that resolve to <stem>.csv (across normalized names)."""
        return sum(len(self._members[nm]) for nm in self.candidates(stem))

    def resolve_single(self, stem: str) -> str | None:
        """Return the one normalized <stem>.csv, or None. Raise if ambiguous."""
        cands = self.candidates(stem)
        raw_total = self.duplicate_raw_count(stem)
        if len(cands) > 1 or raw_total > 1:
            raise ValueError(
                f"ambiguous dataset: multiple {stem}.csv candidates "
                f"({cands or raw_total} raw members). Put exactly one {stem}.csv at the "
                f"archive root."
            )
        return cands[0] if cands else None

    def open(self, normalized: str):
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.open(raw)  # ZipExtFile — streams, no full-member read
        return open(raw, "rb")

    def _member_bytes(self, normalized: str) -> int:
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.getinfo(raw).file_size
        return os.path.getsize(raw)

    def _guard_size(self, normalized: str) -> None:
        """Reject an oversized member/file before pandas materializes it. Covers directory-mode
        inputs too (the zip-bomb guard only sees archives)."""
        size = self._member_bytes(normalized)
        if size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{normalized}: {size:,} uncompressed bytes exceeds the per-file limit "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES:,} (DIMER_MAX_MEMBER_BYTES); refusing to load."
            )

    def read_csv(self, normalized: str, nrows: int | None = None) -> pd.DataFrame:
        """Read a CSV member with memory bounded before materialization: a raw-byte ceiling per
        file, and — for a full read — a chunked parse that refuses to build a frame past
        MAX_CSV_ROWS rather than OOM on a hostile or accidental giant table. Rows are never
        silently dropped: an over-limit table is rejected, not truncated."""
        self._guard_size(normalized)
        if nrows is not None:
            with self.open(normalized) as handle:
                return pd.read_csv(handle, nrows=nrows)
        with self.open(normalized) as handle:
            chunks: list[pd.DataFrame] = []
            rows = 0
            for chunk in pd.read_csv(handle, chunksize=CSV_READ_CHUNK_ROWS):
                rows += len(chunk)
                if rows > MAX_CSV_ROWS:
                    raise ValueError(
                        f"{normalized}: exceeds the {MAX_CSV_ROWS:,}-row read ceiling "
                        f"(DIMER_MAX_CSV_ROWS); refusing to load the whole table into memory."
                    )
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
# ============================================================================
# END shared block
# ============================================================================


@dataclass
class Config:
    dataset_dir: Path
    output_dir: Path
    result_path: Path
    done_callback: str
    callback_timeout: float
    train_device: str
    default_task_type: str
    pipeline_metadata: dict[str, Any]
    target_column: str
    drop_columns: list[str]
    max_train_rows: int
    validation_split: float
    time_limit: int
    seed: int
    eval_metric: str
    fine_tune: bool
    fine_tune_steps: int
    model_dir: Path | None
    required_revision: str
    max_eval_rows: int


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes")


def load_config() -> Config:
    """Parse the DIMER_* environment. Called inside main()'s try so malformed input
    produces a structured failure result rather than an import-time crash."""
    hp = json.loads(os.getenv("DIMER_HYPERPARAMETERS_JSON", "{}") or "{}")
    pre = json.loads(os.getenv("DIMER_PREPROCESSING_ARGS_JSON", "{}") or "{}")
    model_dir = os.getenv("DIMER_MODEL_DIR", "").strip()
    return Config(
        dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
        output_dir=Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output")),
        result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json")),
        done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
        callback_timeout=float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10")),
        train_device=os.getenv("DIMER_TRAIN_DEVICE", "cuda:0").strip(),
        default_task_type=os.getenv("DIMER_TASK_TYPE", "tabular_classification"),
        pipeline_metadata=json.loads(os.getenv("DIMER_PIPELINE_METADATA_JSON", "{}") or "{}"),
        target_column=str(pre.get("target_column") or "target").strip(),
        drop_columns=[c.strip() for c in str(pre.get("drop_columns") or "").split(",") if c.strip()],
        max_train_rows=int(pre.get("max_train_rows") or MITRA_ROW_LIMIT),
        validation_split=float(pre.get("validation_split") if pre.get("validation_split") is not None else 0.2),
        time_limit=int(hp.get("time_limit_seconds") or 600),
        seed=int(hp.get("seed") or 0),
        eval_metric=str(hp.get("eval_metric") or "accuracy").strip(),
        fine_tune=_as_bool(hp.get("fine_tune", True), True),
        fine_tune_steps=int(hp.get("fine_tune_steps") or 0),
        model_dir=Path(model_dir) if model_dir else None,
        required_revision=os.getenv("DIMER_MITRA_REVISION", "").strip() or PINNED_MITRA_REVISION,
        max_eval_rows=int(os.getenv("DIMER_MAX_EVAL_ROWS", "50000")),
    )


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def write_result(cfg: Config, payload: dict[str, Any]) -> None:
    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    cfg.result_path.write_text(content, encoding="utf-8")


def notify_done_callback(cfg: Config) -> dict[str, Any]:
    if not cfg.done_callback:
        return {"attempted": False, "message": "DIMER_DONE_CALLBACK not set; skipping."}
    parsed = urlparse(cfg.done_callback)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "message": f"Unsupported scheme: {parsed.scheme}"}
    try:
        response = requests.post(cfg.done_callback, timeout=cfg.callback_timeout)
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


# --- Weights resolution + provenance (findings: revision enforcement, uploaded weights) ---

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_commit_from_path(path: str) -> str | None:
    m = re.search(r"/snapshots/([0-9a-fA-F]{7,64})/", path.replace("\\", "/"))
    return m.group(1) if m else None


def _hf_hub_dir() -> Path:
    home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def _install_uploaded_weights(model_dir: Path) -> tuple[str, str]:
    """Materialize an uploaded checkpoint into the HF cache for BASE_MODEL so AutoGluon's
    Mitra loader (which only accepts a repo id) serves these exact bytes offline.
    Returns (synthetic_commit, sha256)."""
    msf, cfgf = model_dir / "model.safetensors", model_dir / "config.json"
    if not msf.exists() or not cfgf.exists():
        raise FileNotFoundError(
            f"DIMER_MODEL_DIR {model_dir} must contain model.safetensors and config.json"
        )
    sha = _sha256_file(str(msf))
    commit = sha[:40]  # deterministic synthetic revision from content
    repo = _hf_hub_dir() / ("models--" + BASE_MODEL.replace("/", "--"))
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json"):
        dst = snap / name
        if not dst.exists():
            shutil.copy(model_dir / name, dst)
    (repo / "refs" / "main").write_text(commit)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return commit, sha


def resolve_and_verify_weights(cfg: Config) -> dict[str, Any]:
    """Resolve the weights AutoGluon's Mitra loader will actually use, record provenance,
    and refuse to proceed on unexpected weights. AutoGluon 1.5.0's Mitra loads from a repo id
    via hf_hub_download with no revision arg, so the enforceable guarantee is a SHA-256 check
    of the resolved model.safetensors, not a revision pin."""
    prov: dict[str, Any] = {
        "baseModel": BASE_MODEL,
        "baseModelRevisionExpected": cfg.required_revision,
        "expectedSha256": EXPECTED_WEIGHTS_SHA256,
        "expectedConfigSha256": EXPECTED_CONFIG_SHA256,
    }
    if cfg.model_dir is not None:
        # _install sets HF_HUB_OFFLINE=1 before huggingface_hub is first imported (it reads the
        # flag at import time), so AutoGluon's loader serves the uploaded bytes from the cache.
        commit, sha = _install_uploaded_weights(cfg.model_dir)
        config_sha = _sha256_file(str(cfg.model_dir / "config.json"))
        prov.update({
            "source": "uploaded", "baseModelRevision": commit, "weightsSha256": sha,
            "configSha256": config_sha, "enforced": False,
            "note": "Uploaded weights used verbatim; not checked against the public pinned checksum.",
        })
        return prov

    from huggingface_hub import hf_hub_download

    # Resolve exactly as Mitra's from_pretrained does: hf_hub_download(repo, filename) on main.
    loaded = hf_hub_download(BASE_MODEL, "model.safetensors")
    config_path = hf_hub_download(BASE_MODEL, "config.json")
    commit = _snapshot_commit_from_path(loaded)
    sha = _sha256_file(loaded)
    config_sha = _sha256_file(config_path)
    prov.update({
        "source": "huggingface", "baseModelRevision": commit, "weightsSha256": sha,
        "configSha256": config_sha, "enforced": True,
    })
    if cfg.required_revision == PINNED_MITRA_REVISION and sha != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"Mitra weights to load have SHA-256 {sha}, expected {EXPECTED_WEIGHTS_SHA256} for "
            f"pinned revision {PINNED_MITRA_REVISION}. The hub 'main' may have drifted. Bake the "
            f"pinned revision into the image (Dockerfile Option A) or set DIMER_MITRA_REVISION."
        )
    if cfg.required_revision == PINNED_MITRA_REVISION and config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"Mitra config.json has SHA-256 {config_sha}, expected {EXPECTED_CONFIG_SHA256} for "
            f"pinned revision {PINNED_MITRA_REVISION}. The hub 'main' may have drifted; bake the "
            f"pinned revision (Dockerfile Option A) or set DIMER_MITRA_REVISION."
        )
    if commit and cfg.required_revision and commit != cfg.required_revision:
        raise RuntimeError(
            f"Mitra weights resolved to revision {commit}, required {cfg.required_revision}."
        )
    return prov


# --- Data preparation (class-preserving) ---

def _stratified_cap(train: pd.DataFrame, target_col: str, ceiling: int, seed: int) -> pd.DataFrame:
    """Sample train down to <= ceiling rows while keeping every class present. Guarantees at
    least one row per class (a plain stratified split can round a tiny class to zero), then
    fills the remaining budget at random."""
    if len(train) <= ceiling:
        return train
    classes_before = set(train[target_col].unique())
    if ceiling < len(classes_before):
        raise ValueError(
            f"max_train_rows ({ceiling}) is below the class count ({len(classes_before)}); "
            f"cannot keep every class."
        )
    rng = np.random.RandomState(seed)
    keep: list[Any] = []
    for cls in sorted(classes_before, key=str):
        idx = train.index[train[target_col] == cls].to_numpy()
        keep.append(int(rng.choice(idx, size=1)[0]))
    keep_set = set(keep)
    remaining = np.array([i for i in train.index.to_numpy() if i not in keep_set])
    need = ceiling - len(keep)
    if need > 0 and len(remaining) > 0:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        keep.extend(int(i) for i in extra)
    capped = train.loc[keep]
    if set(capped[target_col].unique()) != classes_before:
        raise RuntimeError("class-preserving cap dropped a class; refusing to train.")
    return capped


def _stratified_holdout(train: pd.DataFrame, target_col: str, val_frac: float,
                        seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a stratified holdout so every class stays represented in both train and val.

    The requested fraction is converted to an integer holdout size clamped so that both splits
    can hold every class: a scikit-learn stratified split requires the validation size and the
    training size to each be at least the class count. A fraction too small to satisfy that
    (e.g. 50 rows, 10 classes, split 0.05) is raised as a clear error rather than crashing deep
    in scikit-learn."""
    from sklearn.model_selection import train_test_split

    counts = train[target_col].value_counts()
    n_classes = int(counts.size)
    if (counts < 2).any():
        raise ValueError(
            f"class(es) with fewer than 2 rows cannot be split: "
            f"{counts[counts < 2].to_dict()}"
        )
    n = len(train)
    n_val = int(round(n * val_frac))
    n_val = min(max(n_val, n_classes), n - n_classes)  # val >= classes AND train >= classes
    if n_val < n_classes or n - n_val < n_classes:
        raise ValueError(
            f"cannot build a stratified holdout: {n} usable rows, {n_classes} classes, "
            f"validation_split={val_frac}. Provide more rows, fewer classes, or a larger split."
        )
    tr, va = train_test_split(
        train, test_size=n_val, random_state=seed, stratify=train[target_col]
    )
    if set(tr[target_col].unique()) != set(train[target_col].unique()):
        raise RuntimeError("stratified split left a class out of train; refusing to train.")
    return tr, va


def _prepare_frames(cfg: Config, source: DatasetSource) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, int]:
    """Return (train, val, test, num_classes). Non-null-target rows only; the cap and holdout
    split are class-preserving so no class is silently lost."""
    train_path = source.resolve_single("train")
    if train_path is None:
        raise FileNotFoundError("no train.csv in the dataset (validator should have caught this)")
    train = source.read_csv(train_path)
    if cfg.target_column not in train.columns:
        raise KeyError(f"target column '{cfg.target_column}' not in {list(train.columns)}")

    # Never drop the target, even if a malformed config lists it in drop_columns.
    drop = [c for c in cfg.drop_columns if c in train.columns and c != cfg.target_column]
    train = train.drop(columns=drop).dropna(subset=[cfg.target_column])

    classes_all = set(train[cfg.target_column].unique())
    num_classes = len(classes_all)
    if num_classes < 2:
        raise ValueError(
            f"target '{cfg.target_column}' has {num_classes} distinct class(es); "
            f"classification needs at least 2."
        )
    if num_classes > MITRA_CLASS_LIMIT:
        raise ValueError(
            f"target '{cfg.target_column}' has {num_classes} classes; Mitra supports at most "
            f"{MITRA_CLASS_LIMIT}. Reduce the number of classes (e.g. merge rare labels)."
        )

    def _prep_holdout(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in cfg.drop_columns if c in df.columns and c != cfg.target_column]
        return df.drop(columns=cols).dropna(subset=[cfg.target_column])

    val_path = source.resolve_single("val")
    if val_path is not None:
        val = _prep_holdout(source.read_csv(val_path))
    else:
        val_frac = min(max(cfg.validation_split, 0.0), 0.4)
        if val_frac > 0 and len(train) > MIN_ROWS_FOR_SPLIT:
            train, val = _stratified_holdout(train, cfg.target_column, val_frac, cfg.seed)
        else:
            val = pd.DataFrame(columns=train.columns)

    ceiling = min(cfg.max_train_rows, MITRA_ROW_LIMIT)
    if len(train) > ceiling:
        log(f"Class-preserving sample of train {len(train)} -> {ceiling} rows (seed={cfg.seed}).")
        train = _stratified_cap(train, cfg.target_column, ceiling, cfg.seed)

    # The problem type is fixed by the labels present across the whole cleaned train set.
    if set(train[cfg.target_column].unique()) != classes_all:
        raise RuntimeError("training class set changed after split/cap; refusing to train.")

    test_path = source.resolve_single("test")
    test = _prep_holdout(source.read_csv(test_path)) if test_path is not None else None

    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True) if test is not None else None, num_classes)


def _evaluate(cfg: Config, predictor, frame: pd.DataFrame) -> dict[str, Any]:
    """AutoGluon's own evaluation is authoritative — it knows the label mapping. Cap the
    evaluation set for memory."""
    if cfg.max_eval_rows and len(frame) > cfg.max_eval_rows:
        log(f"Capping evaluation set {len(frame)} -> {cfg.max_eval_rows} rows (seed={cfg.seed}).")
        frame = frame.sample(n=cfg.max_eval_rows, random_state=cfg.seed)
    out: dict[str, Any] = {"rows": int(len(frame))}
    try:
        raw = predictor.evaluate(frame, auxiliary_metrics=True, silent=True)
        # AutoGluon reports higher-is-better and sign-flips loss metrics; present log_loss in
        # its conventional positive (lower-is-better) form for downstream consumers.
        out["evaluation"] = {
            k: float(-v if "log_loss" in k else v) for k, v in raw.items()
        }
    except Exception as exc:  # noqa: BLE001
        out["evaluationError"] = str(exc)
    return out


# DIMER/AutoGluon eval-metric name -> Mitra's native early-stopping metric. Unmapped names
# fall back to Mitra's default and only drive AutoGluon's reported metric.
_MITRA_METRIC_MAP = {
    "log_loss": "log_loss",
    "accuracy": "accuracy", "acc": "accuracy",
    "roc_auc": "roc_auc", "auc": "roc_auc",
}


def _mitra_metric(name: str) -> str | None:
    return _MITRA_METRIC_MAP.get(name.strip().lower())


def _fit_and_evaluate(cfg: Config, train: pd.DataFrame, val: pd.DataFrame,
                      test: pd.DataFrame | None, num_classes: int) -> dict[str, Any]:
    requested_cpu = cfg.train_device.lower() == "cpu"
    if requested_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _seed_everything(cfg.seed)
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False
    use_gpu = gpu_available and not requested_cpu

    fine_tune = cfg.fine_tune
    if not use_gpu and fine_tune:
        why = "no GPU is available" if not gpu_available else "CPU was requested"
        log(f"Running zero-shot (fine_tune=False): {why}; Mitra fine-tuning requires a GPU.")
        fine_tune = False
    # Propagate the run seed and (when mappable) the eval metric into Mitra itself, not just
    # AutoGluon's reporting: "seed" seeds Mitra's val-split/augmentation RNG (ConfigRun.seed),
    # and "metric" drives its fine-tune early-stopping. NOTE: AutoGluon 1.5.0's Mitra disables
    # its global set_seed (an upstream FIXME), so a fixed seed makes the internal split
    # reproducible but not the full fit — a known upstream limit, not a bug here.
    mitra_hp: dict[str, Any] = {"fine_tune": fine_tune, "seed": cfg.seed}
    if fine_tune and cfg.fine_tune_steps:
        mitra_hp["fine_tune_steps"] = cfg.fine_tune_steps
    mitra_metric = _mitra_metric(cfg.eval_metric)
    if mitra_metric is not None:
        mitra_hp["metric"] = mitra_metric
    else:
        log(f"eval_metric '{cfg.eval_metric}' has no Mitra-native early-stopping equivalent; "
            f"Mitra keeps its default metric (AutoGluon still reports '{cfg.eval_metric}').")

    problem_type = "binary" if num_classes == 2 else "multiclass"

    from autogluon.tabular import TabularPredictor

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = cfg.output_dir / "mitra_predictor"

    predictor = TabularPredictor(
        label=cfg.target_column,
        problem_type=problem_type,
        eval_metric=cfg.eval_metric,
        path=str(predictor_path),
        verbosity=2,
    )
    predictor.fit(
        train,
        hyperparameters={MITRA_MODEL_KEY: mitra_hp},
        fit_weighted_ensemble=False,
        time_limit=cfg.time_limit,
    )

    trained = list(predictor.model_names())
    if not trained:
        raise RuntimeError(
            f"{MITRA_MODEL_KEY} did not train. A common cause is AutoGluon's memory guard "
            f"(it needs the projected footprint under the available-RAM threshold). Request "
            f"a larger GPU/memory profile for this pipeline, then re-run. Check the fit log "
            f"for 'Not enough memory to safely train model'."
        )
    if not any("mitra" in m.lower() for m in trained):
        raise RuntimeError(
            f"expected Mitra but AutoGluon trained {trained} — refusing to report a result "
            f"for a model that was not the one requested"
        )

    metrics: dict[str, Any] = {
        "trainedModels": trained,
        "trainRows": int(len(train)),
        "numClasses": num_classes,
        "problemType": problem_type,
        "mode": "fine-tune" if fine_tune else "zero-shot",
        "device": "cuda" if use_gpu else "cpu",
        "evalMetric": cfg.eval_metric,
        "mitraMetric": mitra_metric or "<mitra-default>",
        "mitraSeed": cfg.seed,
    }
    # Transparency: the holdout size can differ from the requested fraction — the stratified
    # split widens a too-small fraction to fit every class, and size caps can shift it too.
    metrics["requestedValidationSplit"] = cfg.validation_split
    if len(val) > 0:
        val_eval = _evaluate(cfg, predictor, val)
        metrics["valRows"] = val_eval["rows"]
        metrics["effectiveValidationRows"] = int(len(val))
        denom = metrics["trainRows"] + int(len(val))
        metrics["effectiveValidationSplit"] = round(int(len(val)) / denom, 4) if denom else 0.0
        evaluation = val_eval.get("evaluation", {})
        metrics["valEvaluation"] = evaluation
        if "evaluationError" in val_eval:
            metrics["valEvaluationError"] = val_eval["evaluationError"]
        headline = evaluation.get(cfg.eval_metric)
        if headline is not None:
            metrics["headlineMetric"] = cfg.eval_metric
            metrics["headlineScore"] = float(headline)
            log(f"Holdout {cfg.eval_metric}={headline:.4f} on {metrics['valRows']} rows "
                f"({num_classes} classes).")
    else:
        metrics["valRows"] = 0
        metrics["note"] = "No validation rows available; trained on all rows without holdout."

    if test is not None and len(test) > 0:
        test_eval = _evaluate(cfg, predictor, test)
        metrics["test"] = {"rows": test_eval["rows"], "evaluation": test_eval.get("evaluation", {})}
        log(f"Test evaluated on {test_eval['rows']} rows.")

    metrics["artifactPath"] = str(predictor_path)
    return metrics


def run(cfg: Config) -> int:
    provenance = resolve_and_verify_weights(cfg)
    source = DatasetSource(cfg.dataset_dir)
    try:
        train, val, test, num_classes = _prepare_frames(cfg, source)
    finally:
        source.close()
    metrics = _fit_and_evaluate(cfg, train, val, test, num_classes)
    provenance["dataset"] = _dataset_sha256(cfg)
    provenance["autogluonVersion"] = getattr(sys.modules.get("autogluon.tabular"), "__version__", None)
    headline = metrics.get("headlineScore")
    payload = {
        "successful": True,
        "message": (
            f"Mitra {metrics['mode']} succeeded on {metrics['trainRows']} rows"
            + (f"; holdout {cfg.eval_metric} {headline:.4f}." if headline is not None else ".")
        ),
        "metrics": metrics,
        "artifacts": {"modelDir": str(cfg.output_dir / "mitra_predictor")},
        "provenance": provenance,
        "metadata": {
            "template": TEMPLATE_NAME,
            "taskType": cfg.default_task_type,
            "baseModel": BASE_MODEL,
            "targetColumn": cfg.target_column,
            "dropColumns": cfg.drop_columns,
            "seed": cfg.seed,
            "timeLimitSeconds": cfg.time_limit,
            "evalMetric": cfg.eval_metric,
            "trainDevice": cfg.train_device,
        },
    }
    write_result(cfg, payload)
    log(f"Callback: {json.dumps(notify_done_callback(cfg), sort_keys=True)}")
    return 0


def _dataset_sha256(cfg: Config) -> dict[str, Any] | None:
    h = hashlib.sha256()
    zips = sorted(cfg.dataset_dir.glob("*.zip"))
    if zips:
        with open(zips[0], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"file": zips[0].name, "sha256": h.hexdigest()}
    csvs = sorted(cfg.dataset_dir.rglob("*.csv"))
    if not csvs:
        return None
    for p in csvs:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return {"files": [p.name for p in csvs], "sha256": h.hexdigest()}


def _failure_provenance(cfg: Config | None) -> dict[str, Any]:
    prov: dict[str, Any] = {"baseModel": BASE_MODEL, "baseModelRevisionExpected": PINNED_MITRA_REVISION}
    if cfg is not None:
        try:
            prov["dataset"] = _dataset_sha256(cfg)
        except Exception:  # noqa: BLE001
            pass
    prov["autogluonVersion"] = getattr(sys.modules.get("autogluon.tabular"), "__version__", None)
    return prov


def main() -> int:
    cfg: Config | None = None
    try:
        cfg = load_config()
        return run(cfg)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": f"Mitra fine-tuning failed: {exc}",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "provenance": _failure_provenance(cfg),
            "metadata": {
                "template": TEMPLATE_NAME,
                "taskType": (cfg.default_task_type if cfg else "tabular_classification"),
            },
        }
        try:
            if cfg is not None:
                write_result(cfg, payload)
                notify_done_callback(cfg)
            else:
                fallback = Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json"))
                fallback.parent.mkdir(parents=True, exist_ok=True)
                fallback.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
