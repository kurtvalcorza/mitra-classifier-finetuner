"""Unit tests for the Mitra classifier finetuner (no model fitting)."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as T  # noqa: E402


def _zip(tmp: Path, members: dict[str, pd.DataFrame]) -> Path:
    p = tmp / "d.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, df in members.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(arc, buf.getvalue())
    return p


def _cfg(tmp: Path, **over):
    base = dict(
        dataset_dir=tmp, output_dir=tmp / "out", result_path=tmp / "r.json", done_callback="",
        callback_timeout=1.0, train_device="cpu", default_task_type="tabular_classification",
        pipeline_metadata={}, target_column="target", drop_columns=[], max_train_rows=10000,
        validation_split=0.2, time_limit=60, seed=0, eval_metric="accuracy", fine_tune=True,
        fine_tune_steps=0, model_dir=None, required_revision=T.PINNED_MITRA_REVISION,
        max_eval_rows=50000,
    )
    base.update(over)
    return T.Config(**base)


def _frame(per_class: dict[str, int]):
    rows = []
    i = 0
    for cls, n in per_class.items():
        for _ in range(n):
            rows.append({"f1": i, "f2": i * 2, "target": cls})
            i += 1
    return pd.DataFrame(rows)


def test_ambiguous_train_rejected(tmp_path):
    _zip(tmp_path, {"train.csv": _frame({"a": 5}), "dataset/train.csv": _frame({"a": 5})})
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.resolve_single("train")
    finally:
        src.close()


def test_stratified_holdout_preserves_classes():
    train = _frame({"a": 50, "b": 50, "rare": 2})
    tr, va = T._stratified_holdout(train, "target", 0.2, 0)
    assert set(tr["target"]) == {"a", "b", "rare"}  # rare class stays in train


def test_stratified_cap_preserves_classes():
    train = _frame({"a": 5000, "b": 5000, "rare": 3})
    capped = T._stratified_cap(train, "target", 2000, 0)
    assert len(capped) <= 2000
    assert set(capped["target"]) == {"a", "b", "rare"}


def test_prepare_frames_infers_classes_and_preserves(tmp_path):
    _zip(tmp_path, {"train.csv": _frame({"a": 40, "b": 40, "c": 40})})
    src = T.DatasetSource(tmp_path)
    try:
        train, val, test, n = T._prepare_frames(_cfg(tmp_path), src)
    finally:
        src.close()
    assert n == 3
    assert set(train["target"]) == {"a", "b", "c"}
    assert len(val) > 0 and test is None


def test_too_many_classes_rejected(tmp_path):
    _zip(tmp_path, {"train.csv": _frame({f"c{i}": 5 for i in range(11)})})
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            T._prepare_frames(_cfg(tmp_path), src)
    finally:
        src.close()


def test_uploaded_weights_installed_and_resolvable(tmp_path, monkeypatch):
    # Isolate the HF cache so we never touch the real one.
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"FAKE_WEIGHTS_BYTES")
    (model_dir / "config.json").write_text('{"dim": 512}')

    prov = T.resolve_and_verify_weights(_cfg(tmp_path, model_dir=model_dir))
    assert prov["source"] == "uploaded"
    assert prov["weightsSha256"]
    assert prov.get("configSha256")  # finding 3: config.json is hashed too

    # The loader (hf_hub_download on the repo id, offline) now resolves to the uploaded bytes.
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(T.BASE_MODEL, "model.safetensors")
    assert Path(path).read_bytes() == b"FAKE_WEIGHTS_BYTES"


def test_stratified_holdout_small_fraction_many_classes():
    # 50 rows, 10 classes, split 0.05 -> would crash a naive stratified split; must not here.
    train = _frame({f"c{i}": 5 for i in range(10)})
    tr, va = T._stratified_holdout(train, "target", 0.05, 0)
    assert set(tr["target"]) == set(f"c{i}" for i in range(10))
    assert set(va["target"]) == set(f"c{i}" for i in range(10))


def test_stratified_holdout_class_too_small_raises():
    train = _frame({f"c{i}": 1 for i in range(10)})  # a class with only 1 row can't be split
    with pytest.raises(ValueError):
        T._stratified_holdout(train, "target", 0.2, 0)


def test_finetuner_never_drops_target(tmp_path):
    _zip(tmp_path, {"train.csv": _frame({"a": 40, "b": 40, "c": 40})})
    src = T.DatasetSource(tmp_path)
    try:
        # target listed in drop_columns must not cause a KeyError; the target survives.
        train, val, test, n = T._prepare_frames(_cfg(tmp_path, drop_columns=["target", "f1"]), src)
    finally:
        src.close()
    assert "target" in train.columns and n == 3


def test_log_loss_sign_normalized(tmp_path):
    class _FakePredictor:
        def evaluate(self, frame, auxiliary_metrics=True, silent=True):
            return {"log_loss": -0.5, "accuracy": 0.7}  # AutoGluon returns negated log_loss
    out = T._evaluate(_cfg(tmp_path), _FakePredictor(), pd.DataFrame({"target": ["a", "b"]}))
    assert out["evaluation"]["log_loss"] == 0.5   # normalized to conventional positive
    assert out["evaluation"]["accuracy"] == 0.7


def test_mitra_metric_map():
    assert T._mitra_metric("accuracy") == "accuracy"
    assert T._mitra_metric("ROC_AUC") == "roc_auc"
    assert T._mitra_metric("auc") == "roc_auc"
    assert T._mitra_metric("log_loss") == "log_loss"
    assert T._mitra_metric("f1") is None  # unmapped -> Mitra default


def test_member_byte_cap_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_MEMBER_UNCOMPRESSED_BYTES", 100)  # smaller than the CSV
    _zip(tmp_path, {"train.csv": _frame({"a": 40, "b": 40})})
    with pytest.raises(ValueError):
        T.DatasetSource(tmp_path)


def test_row_ceiling_rejected_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_CSV_ROWS", 10)
    monkeypatch.setattr(T, "CSV_READ_CHUNK_ROWS", 4)
    _zip(tmp_path, {"train.csv": _frame({"a": 40, "b": 40})})  # 80 rows > ceiling of 10
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_directory_mode_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_MEMBER_UNCOMPRESSED_BYTES", 50)
    _frame({"a": 40, "b": 40}).to_csv(tmp_path / "train.csv", index=False)  # no zip
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_normalize_device():
    # cpu passes through; the DIMER-documented bare-integer form becomes cuda:<n>.
    assert T._normalize_device("cpu") == "cpu"
    assert T._normalize_device("CPU") == "cpu"
    assert T._normalize_device("cuda:0") == "cuda:0"
    assert T._normalize_device("cuda:1") == "cuda:1"
    assert T._normalize_device("CUDA:2") == "cuda:2"
    assert T._normalize_device("0") == "cuda:0"      # bare integer -> cuda:0 (docs call-out)
    assert T._normalize_device("3") == "cuda:3"      # explicit index honored
    assert T._normalize_device("gpu") == "cuda:0"
    assert T._normalize_device("cuda") == "cuda:0"
    assert T._normalize_device("") == "cuda:0"
    assert T._normalize_device(None) == "cuda:0"
    assert T._normalize_device("mps") == "cpu"       # unknown accelerator -> safe CPU fallback
    assert T._normalize_device("cuda:x") == "cpu"    # malformed index -> CPU fallback
