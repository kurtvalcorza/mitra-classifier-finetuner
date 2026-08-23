#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
EXPECTED_HARDENING_SHA256="d694532cc23366c119e975cf67ba51f3d85f722ae1c0d1711b4071a1cca09393"
EXPECTED_CONTRACT_SHA256="d2dc62fe4f0437941a29140c35e083f39e0eba9db87561d3bda6a36e208d8bb9"
REQUIRED_CODES={"VALID","INVALID_DATASET","VALIDATION_FAILED","SUCCEEDED","INVALID_CONFIGURATION","RESOURCE_LIMIT","TRAINING_FAILED","ARTIFACT_FAILED"}
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->int:
 root=Path(__file__).resolve().parents[1];contract=root/"dimer-runtime-contract.json"
 if not contract.exists() or sha(contract)!=EXPECTED_CONTRACT_SHA256:return 1
 data=json.loads(contract.read_text());model_dir=next((x for x in data.get("runtimeInputs",[]) if x.get("name")=="DIMER_MODEL_DIR"),None)
 if data.get("schemaVersion")!=1 or data.get("sharedCodeStrategy")!="KEEP_PARITY_COPIES" or not REQUIRED_CODES.issubset(set(data.get("resultContract",{}).get("stableCodes",[]))) or not model_dir or model_dir.get("requirement")!="unsupported":return 1
 modules=[p for p in [root/"contract_hardening.py",root/"validator"/"contract_hardening.py",root/"finetuner"/"contract_hardening.py"] if p.exists()]
 if not modules or any(sha(p)!=EXPECTED_HARDENING_SHA256 for p in modules):return 1
 expected="tabular_regression" if "regressor" in root.name else "tabular_classification"
 for manifest in list(root.glob("dimer-pipeline.json"))+list(root.glob("finetuner/dimer-pipeline.json")):
  if json.loads(manifest.read_text()).get("taskType")!=expected:return 1
 print(f"Contract OK: {len(modules)} hardening module(s), task={expected}, result schema v1, pinned runtime contract.");return 0
if __name__=="__main__":sys.exit(main())
