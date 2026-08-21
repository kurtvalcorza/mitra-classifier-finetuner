# DIMER portal build & manifest contract — classification finetuner (issue #3)

Traceability for *Align finetuner repository with DIMER portal build and manifest
contract*. DIMER builds this repository's default branch with the repository
**root** as the Docker build context, reads `dimer-pipeline.json` from the root on
every finetuner build, and launches the container by the portal naming
convention (`train.py`).

The repository already satisfied the portal file layout; this change adds a
`.dockerignore`, tightens `.gitignore`, and records the manifest→runtime
verification below. No fine-tuning semantics change.

## Build / layout

| Requirement | Status | Evidence |
|---|---|---|
| Root `Dockerfile`, `train.py`, `requirements.txt`, `README.md` | ✅ | present at root |
| Dockerfile invokes root `train.py` | ✅ | `CMD ["python", "train.py"]`; `COPY train.py ./` |
| `dimer-pipeline.json` at repository root | ✅ | present at root |
| `model_id` **not** in `dimer-pipeline.json` | ✅ | absent — Base Model comes from DIMER pipeline metadata/config |
| `DIMER_TASK_TYPE=tabular_classification` baked as fallback | ✅ | `ENV DIMER_TASK_TYPE=tabular_classification` (Custom/Other → object_detection cannot win) |
| HF Base Model id `autogluon/mitra-classifier` handled unchanged | ✅ | resolved via `resolve_and_verify_weights`, recorded in `provenance` |
| `.gitignore` / `.dockerignore` exclude dataset/result/model artifacts | ✅ | `.dockerignore` added; `.gitignore` extended (`weights/`,`*.safetensors`,`*.csv`,`*.zip`,`data/`,`datasets/`,`results/`,`result.json`) |
| Builds from repository root as DIMER CodeBuild does | ✅ | `docker build .` from root; context limited by `.dockerignore` |

## Manifest → runtime key mapping (1:1, verified)

`train.py` parses `DIMER_PREPROCESSING_ARGS_JSON` → `pre` and
`DIMER_HYPERPARAMETERS_JSON` → `hp` (train.py:275–276). Every declared key is
consumed; no undeclared runtime key is read (beyond intentional internal
`DIMER_*` operational defaults documented in the module header).

| Manifest group / key | Consumed at |
|---|---|
| `datasetPreprocessing.target_column` | `pre.get("target_column")` (train.py:287) |
| `datasetPreprocessing.drop_columns` | `pre.get("drop_columns")` (288) |
| `datasetPreprocessing.max_train_rows` | `pre.get("max_train_rows")` (289) |
| `datasetPreprocessing.validation_split` | `pre.get("validation_split")` (290) |
| `modelFinetuning.time_limit_seconds` | `hp.get("time_limit_seconds")` (291) |
| `modelFinetuning.seed` | `hp.get("seed")` (292) → Mitra `mitra_hp["seed"]` |
| `modelFinetuning.eval_metric` | `hp.get("eval_metric")` (293) — enum `accuracy`, `balanced_accuracy`, `log_loss`, `f1_macro`, `mcc` |
| `modelFinetuning.fine_tune` | `hp.get("fine_tune")` (294) → Mitra `mitra_hp["fine_tune"]` |
| `modelFinetuning.fine_tune_steps` | `hp.get("fine_tune_steps")` (295) → `mitra_hp["fine_tune_steps"]` |

## Tests

Unit + shared-code parity green in CI (`ci.yml`). Real-stack GPU smoke
(build → offline load → fit → save → reload → predict, `problemType=multiclass`)
is exercised by `integration.yml` (manual/nightly, self-hosted GPU) and was run
live on a 5070 Ti 2026-08-19; it is not run on hosted CI until a GPU runner is
attached.

`Closes #3`.
