# P4 V2 CPU semantic delivery

Actual changes: added bounded schema-version-2 preregistered selection, complete step/sample-token trace binding, full-vs-selected counts, explicit source-bound GPU elapsed timing and full-run extra-I/O drain consistency in p4_paired_measurement_verifier.py. The original estimator arithmetic is unchanged; a single accessor supplies V2 gpu_elapsed_ns rather than historical V1 host fixture intervals. Original CostTable/candidate parser/loader APIs, native engine and production gates are unchanged.

CPU-only test evidence:
- run-09-v2-red: old V1 verifier rejected all five V2 positive cases (5 failed, 29 passed). Negative passes at this stage only reflect old schema rejection and are not V2 validation proof.
- run-10-v2-green: 1 failed,108 passed. The missing-reference negative fixture regenerated its removed field during refresh; exact failure output/source retained.
- run-11-v2-final: corrected fixture,109 passed.
- run-12-v2-frozen: 111 passed in 5.90s,0failed,0skipped; original75+newV2 36. Importlib mode with project path, GPU backend import guard and CUDA_VISIBLE_DEVICES=''. Repeated runs are not summed.

Positive fixtures are explicit CPU fixtures, including continuously increasing contexts, complete128 outputs with selected1 or2 actual token IDs, excluded different-context warmup, nonselected and outside-window extra I/O included in final totals, both cost bases and append-only raw-builder/full-loader roundtrip. Negative cases cover omitted/duplicated/reordered ordinals, raw token identity/count drift, unregistered selection and geometry, missing source/GPU duration, host-clock/fallback/invalid-reference substitution, action attribution/drain totals, raw-byte changes after binding/recomputation, forged declared costs and split leakage. Existing plan-wide isolation, wrapper constructor and production lookup gates still pass.

Three persistent fixtures are in fixtures-v2-01. Its result.json records independent refs, exact context, recomputation and production None. Source-labelled native-looking data still cannot grant GPU qualification. The fixture's event-like durations and token trace are synthetic CPU data and must never be cited as actual GPU evidence.

Commands, stdout/stderr, exit codes, XML and guards are preserved for all stages. Latest frozen source/test hashes are source-lock-v2.json. V1 run-08 and earlier raw/source evidence are unchanged. Complete trace projection/drain/clock details are SCHEMA_V2.md.

Actual GPU operations:0. GPU workloads:0. CUDA/backend imports attempted by fixture generation:0. No download, installation, driver/system change, experiment/cache/model mutation or production activation was performed.

Next allowed stage: independent CPU review and root unified regression. Real GPU calibration remains blocked until an authorized card/run budget and actual original-sampling complete output trace, valid event-source timing and final owner drain receipts are available. The current new observer only scopes model_forward; it cannot establish full_decode_step qualification or fill missing output/drain evidence.
