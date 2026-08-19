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

    # The loader (hf_hub_download on the repo id, offline) now resolves to the uploaded bytes.
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(T.BASE_MODEL, "model.safetensors")
    assert Path(path).read_bytes() == b"FAKE_WEIGHTS_BYTES"
