# Final V2 semantic verifier and offline ingestion

Final changes: strict V2 complete-output selection and source-bound elapsed timing, one token per request on every pure-decode forward, actual step batch bounded by actual run requests, selected pure-decode-only domain, optional explicit prefill prelude outside statistics, and an offline CLI V2 byte-ref ingestion branch. V1 history/test/raw-recorder/assembler and original cost arithmetic remain intact. No controller/policy/engine/kernel or production gate changed.

Independent eta review actually proved the former packed-token loophole: full128 IDs could be concentrated into one active1 step, builder and public loader accepted CPU semantics while production remained None. The old raw/plan/candidate/qualification chain and result are preserved at eta/review-cost/v2-independent-01/concentrated-128-output-single-active. Final verifier rejects it through finite physical output boundaries; root/eta are responsible for final independent replay.

Latest targeted test: run-15-v2-final,171 passed,0failed,0skipped in9.21s, importlib mode and backend-denied CPU process. Composition:75 historical measurement tests+43 V2 tests+14 new V2 CLI tests+39 unchanged V1 raw-preparation tests. Root final unified counts must count each collected test once; repeated runs are not summed.
Previous red/green evidence remains: run09 old-schema5fails; run10 fixture refresh1fail; run11/12 V2 positives pass; run13 old CLI2fails; run14 initial strict token+CLI167pass. Commands, outputs, XML, guard and relevant source copies are preserved.

Real offline CLI was separately run in five fresh CPU processes, with original persistent CPU fixture bytes copied into new append-only evidence directories:
- full128/selected1: exit0, builder analysis/candidate/final public loader receipt, production false.
- full128/selected2: exit0, same complete chain, production false.
- raw source-byte mutation: exit1, changed evidence size/SHA.
- complete trace-byte mutation: exit1, changed evidence size/SHA.
- V1 role in V2 bundle: exit1, exact role-version rejection.
All process guards report backend attempts[], GPU_runs0, cuda_initializedfalse. The two positives include actual complete-output IDs/step projections in synthetic CPU fixtures; no actual GPU output was collected.

Files: p4_paired_measurement_verifier.py, prepare_p4_raw_pair.py, test_semantic_v2.py and test_offline_v2_cli.py. Original loader p4_verified_cost_loader.py and p4_raw_pair_recorder.py source digests remain unchanged. Final source-lock-v2-final.json and SCHEMA_V2_FINAL.md freeze exact contracts. CLI source before change is run-13-v2-cli-red/cli-old-source.py.

Actual GPU operations0; no download/install/driver/system/permissions/cache/model mutation. New work is solely source/test/metadata evidence within the isolated P4-02 workspace. PreparedCostTable production lookup remains None and gpu_verified/production_qualified always false.

Next allowed stage is independent CPU replay and root unified regression/freeze. Real GPU cost qualification still needs an authorized card/run budget, actual original-sampling full output traces, valid CUDA-event references and actual final owner drain receipts. Current observer seam has only model_forward scope and cannot grant full_decode_step effect verification.
