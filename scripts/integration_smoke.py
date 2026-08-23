#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def make(dst):
 rng=np.random.RandomState(0);names=["low","mid","high"]
 def frame(n):
  cls=rng.randint(0,3,size=n);centers=np.array([-2.,0.,2.]);return pd.DataFrame({"f1":centers[cls]+rng.normal(scale=.4,size=n),"f2":rng.normal(size=n),"f3":rng.uniform(size=n),"target":[names[c] for c in cls]})
 z=dst/"smoke.zip"
 with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as f:
  for name,n in (("train.csv",300),("val.csv",90),("test.csv",90)):
   p=dst/name;frame(n).to_csv(p,index=False);f.write(p,name);p.unlink()
 return z
def main():
 a=argparse.ArgumentParser();a.add_argument("--weights",required=True);a.add_argument("--device",default="cuda:0");args=a.parse_args();repo=Path(__file__).resolve().parents[1];work=Path(tempfile.mkdtemp(prefix="mitra-smoke-"));ds=work/"dataset";ds.mkdir();out=work/"output";out.mkdir();res=work/"result.json";make(ds);seed=123;env=dict(os.environ);env.update({"DIMER_DATASET_DIR":str(ds),"DIMER_OUTPUT_DIR":str(out),"DIMER_RESULT_PATH":str(res),"DIMER_TRAIN_DEVICE":args.device,"DIMER_MODEL_DIR":args.weights,"DIMER_RUN_ID":"contract-smoke","DIMER_PIPELINE_ID":"mitra-classifier","DIMER_HYPERPARAMETERS_JSON":json.dumps({"seed":seed,"eval_metric":"accuracy","fine_tune":True,"fine_tune_steps":20,"time_limit_seconds":600}),"DIMER_PREPROCESSING_ARGS_JSON":json.dumps({"target_column":"target"})});proc=subprocess.run([sys.executable,"contract_entrypoint.py"],cwd=str(repo),env=env);assert proc.returncode==0;p=json.loads(res.read_text());assert p["successful"] is True and p["schemaVersion"]==1 and p["code"]=="SUCCEEDED" and p["metadata"]["taskType"]=="tabular_classification";assert p["provenance"]["resolvedConfiguration"]["finetuning"]["seed"]==seed and p["provenance"]["execution"]["pipelineId"]=="mitra-classifier";m=p["artifacts"]["modelArtifact"];best=out/"artifacts"/"best.pt";assert m["logicalFormat"]=="autogluon-tabular-predictor" and m["packageFormat"]=="zip" and m["reloadVerified"] is True and m["sha256"]==sha(best);unpack=work/"unpacked";unpack.mkdir();
 with zipfile.ZipFile(best) as z:z.extractall(unpack)
 from autogluon.tabular import TabularPredictor
 pred=TabularPredictor.load(str(unpack/"mitra_predictor"));assert len(pred.predict(pd.DataFrame({"f1":[-2.,2.],"f2":[0.,.1],"f3":[.5,.6]})))==2;print("SMOKE PASS");return 0
if __name__=="__main__":sys.exit(main())
