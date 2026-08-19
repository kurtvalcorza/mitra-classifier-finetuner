# Issue #1 — acceptance record (classifier finetuner)

Traceability for *Harden DIMER classifier finetuning contract, model provenance, and
evaluation*. Each acceptance criterion maps to the code and test that satisfy it on `main`.
The runtime path was verified end-to-end on a GPU (5070 Ti) 2026-08-19: real fine-tune → save
→ reload → predict, `problemType=multiclass`, observed in `result.json`.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Runtime config parsing is exception-safe and covered by tests | ✅ | `load_config()` parses inside `main()`'s protected path; malformed input still writes `result.json` + attempts callback |
| Base Model selection determines the model actually loaded | ✅ | `resolve_and_verify_weights` resolves the exact checkpoint the loader uses; uploaded weights installed via `_install_uploaded_weights` |
| Provenance reports the exact loaded checkpoint/revision | ✅ | `provenance` records `baseModelRevision`, `weightsSha256`, **and `configSha256`**, both checksum-enforced; no lexicographic snapshot selection |

Additional required-change coverage from the issue body:

| Area | Status | Evidence |
|---|---|---|
| Deterministic shared split resolver; duplicate candidates rejected | ✅ | Shared `DatasetSource` block enforced byte-identical + cross-repo SHA by `scripts/check_shared.py`; `resolve_single` raises on ambiguity |
| Stratified auto-split preserves all classes; clear error when impossible | ✅ | `_stratified_holdout` / `_stratified_cap`; `test_stratified_holdout_preserves_classes`, `test_stratified_cap_preserves_classes`, `test_stratified_holdout_class_too_small_raises` |
| Optional `test.csv` scored after fit, reported separately | ✅ | `metrics.test` distinct from validation |
| Resource-bounded evaluation; structured failures not OOM | ✅ | `max_eval_rows` cap + per-file byte / chunked-read row ceiling in the shared block |
| Mode/result semantics: fine-tuned vs zero-shot; reproducibility metadata | ✅ | `metrics.mode`, `metrics.device`, `metrics.mitraSeed`, `metrics.mitraMetric`, `problemType`, `numClasses` |
| CI covers contract | ✅ | `ci.yml` (unit + `check_shared`) + `integration.yml` (real-stack GPU smoke) |

Also this round: `seed` + native `metric` propagated into Mitra; `config.json` checksum
enforced. `log_loss` sign already normalized (`test_log_loss_sign_normalized`).

No open items in-repo. `Closes #1`.
