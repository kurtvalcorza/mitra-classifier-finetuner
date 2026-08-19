#!/usr/bin/env python3
"""GPU integration smoke test for the Mitra classification finetuner.

Exercises the parts the fast unit suite deliberately cannot: the real AutoGluon/Mitra stack,
offline checkpoint loading, model-seed/metric propagation, and a saved -> reloaded -> predict
round-trip (the shape DIMER's serving layer needs). Run inside the built image with a GPU.

Flow: build a tiny 3-class dataset -> run train.py end-to-end against offline uploaded weights
-> assert result.json is successful and carries the new provenance/metric fields -> reload the
saved TabularPredictor and predict. Exits non-zero on any failure.

Usage (inside the container, repo mounted at CWD):
    python scripts/integration_smoke.py --weights /snap
where /snap holds the pinned model.safetensors + config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _make_dataset(dst: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    names = ["low", "mid", "high"]

    def frame(n: int) -> pd.DataFrame:
        cls = rng.randint(0, 3, size=n)
        centers = np.array([-2.0, 0.0, 2.0])
        x1 = centers[cls] + rng.normal(scale=0.4, size=n)
        x2 = rng.normal(size=n)
        x3 = rng.uniform(size=n)
        return pd.DataFrame({"f1": x1, "f2": x2, "f3": x3,
                             "target": [names[c] for c in cls]})

    zip_path = dst / "smoke.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, n in (("train.csv", 300), ("val.csv", 90), ("test.csv", 90)):
            p = dst / name
            frame(n).to_csv(p, index=False)
            zf.write(p, name)
            p.unlink()
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="dir with pinned model.safetensors + config.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="mitra-smoke-"))
    ds_dir = work / "dataset"; ds_dir.mkdir()
    out_dir = work / "output"; out_dir.mkdir()
    result_path = work / "result.json"
    _make_dataset(ds_dir)

    seed = 123
    env = dict(os.environ)
    env.update({
        "DIMER_DATASET_DIR": str(ds_dir),
        "DIMER_OUTPUT_DIR": str(out_dir),
        "DIMER_RESULT_PATH": str(result_path),
        "DIMER_TRAIN_DEVICE": args.device,
        "DIMER_MODEL_DIR": args.weights,
        "DIMER_HYPERPARAMETERS_JSON": json.dumps(
            {"seed": seed, "eval_metric": "accuracy",
             "fine_tune": True, "fine_tune_steps": 20, "time_limit_seconds": 600}
        ),
        "DIMER_PREPROCESSING_ARGS_JSON": json.dumps({"target_column": "target"}),
    })

    print("== running train.py ==", flush=True)
    proc = subprocess.run([sys.executable, "train.py"], cwd=str(repo), env=env)
    if proc.returncode != 0:
        print(f"train.py exited {proc.returncode}")
        return 1

    result = json.loads(result_path.read_text())
    assert result.get("successful") is True, f"run not successful: {result.get('message')}"
    prov = result["provenance"]
    metrics = result["metrics"]
    assert prov.get("configSha256"), "provenance missing configSha256 (finding 3)"
    assert metrics.get("mitraSeed") == seed, f"seed not propagated: {metrics.get('mitraSeed')}"
    assert metrics.get("mitraMetric") == "accuracy", f"metric not mapped: {metrics.get('mitraMetric')}"
    assert metrics.get("mode") == "fine-tune", f"expected fine-tune on GPU, got {metrics.get('mode')}"
    assert metrics.get("problemType") == "multiclass", f"expected multiclass, got {metrics.get('problemType')}"
    print(f"result OK: mode={metrics['mode']} classes={metrics.get('numClasses')} "
          f"mitraMetric={metrics['mitraMetric']} mitraSeed={metrics['mitraSeed']}")

    print("== reload saved predictor and predict ==", flush=True)
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(out_dir / "mitra_predictor"))
    probe = pd.DataFrame({"f1": [-2.0, 2.0], "f2": [0.0, 0.1], "f3": [0.5, 0.6]})
    preds = predictor.predict(probe)
    assert len(preds) == 2, f"expected 2 predictions, got {len(preds)}"
    print(f"reload+predict OK: {list(map(str, preds))}")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
