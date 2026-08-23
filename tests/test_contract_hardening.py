from __future__ import annotations
import json,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import contract_hardening as C

def test_tree_identity(tmp_path):
 (tmp_path/"b").mkdir();(tmp_path/"a").mkdir();(tmp_path/"b"/"train.csv").write_bytes(b"x,y\n1,2\n");(tmp_path/"a"/"val.csv").write_bytes(b"x,y\n3,4\n");first=C.dataset_identity(tmp_path);assert first["files"]==["a/val.csv","b/train.csv"];assert C.dataset_identity(tmp_path)==first;(tmp_path/"a"/"val.csv").rename(tmp_path/"a"/"other.csv");assert C.dataset_identity(tmp_path)["sha256"]!=first["sha256"]
def test_task_lock(tmp_path,monkeypatch):
 monkeypatch.setenv("DIMER_TASK_TYPE","object_detection");p={"successful":False,"metadata":{}};C.normalize_payload(p,task_type="tabular_classification",role="validator",cfg=SimpleNamespace(dataset_dir=tmp_path));assert p["schemaVersion"]==1 and p["code"]=="INVALID_DATASET" and p["metadata"]["taskType"]=="tabular_classification" and p["metadata"]["platformTaskType"]=="object_detection"
def test_artifact_metadata(tmp_path,monkeypatch):
 out=tmp_path/"out";(out/"artifacts").mkdir(parents=True);(out/"evaluation").mkdir();(out/"logs").mkdir();(out/"progress").mkdir();best=out/"artifacts"/"best.pt"
 with zipfile.ZipFile(best,"w") as zf:zf.writestr("mitra_predictor/predictor.pkl",b"fake")
 for p in [out/"evaluation"/"report.json",out/"logs"/"run-summary.json",out/"progress"/"epoch_0001.json"]:p.write_text("{}")
 cfg=SimpleNamespace(output_dir=out,dataset_dir=tmp_path,target_column="target",drop_columns=[],max_train_rows=10000,validation_split=.2,time_limit=60,seed=0,eval_metric="accuracy",fine_tune=False,fine_tune_steps=0,run_id="r1",session_id="s1",train_device="cpu");arts={"modelArtifact":{},"evaluationReport":{},"logArtifact":{}};monkeypatch.setattr(C,"verify_packaged_predictor",lambda *a:None);m=C._harden_artifacts(SimpleNamespace(),cfg,{"mode":"zero-shot"},{"datasetSha256":"abc"},arts,"tabular_classification")["modelArtifact"];assert m["sha256"]==C._sha256_file(best) and m["reloadVerified"] is True;assert json.loads((out/"evaluation"/"report.json").read_text())["schemaVersion"]==1
