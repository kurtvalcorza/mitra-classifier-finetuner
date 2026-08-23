# Mitra ↔ DIMER Workbench contract

This repository owns the Mitra worker side only. `dimer-pipeline.json` is authoritative for user-facing parameters; `dimer-runtime-contract.json` records the runtime matrix; result/artifact outputs are versioned.

Exact task identities are `tabular_classification` and `tabular_regression`. Validator and finetuner share a canonical dataset digest (exact selected ZIP bytes or deterministic path-framed tree). Results use schema v1 and stable machine codes; messages are human-facing. Validator `metadata.classNames` is mandatory, with `[]` for regression/unknown.

The logical model is AutoGluon `TabularPredictor`; `artifacts/best.pt` is a ZIP compatibility package, not native PyTorch. Successful finetuning requires SHA-256 of the exact package plus unpack→reload→predict verification. Resolved preprocessing, finetuning settings, device/mode, run/session/pipeline identity and source revision are recorded in provenance.

Decision: **KEEP_PARITY_COPIES**; the contract module and runtime matrix are byte-pinned in CI.

## DIMER-side requirements — documentation only

**No DIMER/backend code is changed here.** External requirements remain: first-class tabular task typing; neutral generic defaults; typed packaged/directory artifact support; backend validation/derivation from `dimer-pipeline.json`; result/artifact schema validation; preservation of validated dataset identity into training; non-vision AutoGluon serving; and confirmation of Base Model transport (#9). Nested-ZIP behavior remains separately tracked in validator #8.
