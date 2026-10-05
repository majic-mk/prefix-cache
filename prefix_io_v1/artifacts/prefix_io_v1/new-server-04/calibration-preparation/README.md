# Native P1 cost-curve preparation: CPU candidates only

No GPU, model download, actual calibration, cache-engine modification or system change occurred in this work. P1 remains incomplete and P2–P7/P3 execution is not activated. The parent's new-server-04/io-uring-recheck.json still reports native io_uring setup EPERM.

New implementation files:
- experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py
- tests/prefix_io_v1_calibration/test_prepare_plan.py

The real local preparation command (working directory /root/autodl-tmp/prefix-io-v1-handoff/project) was:

~~~sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py --smoke-dir experiments/prefix_io_v1/runs/native-prefix-04/details --doc-sizes 64 128 192 --repeats 3 --output artifacts/prefix_io_v1/new-server-04/calibration-preparation/candidate-plan.json
~~~

It succeeded and produced three original-author cold-prefill candidate jobs with nine planned measured requests. These numbers do not include a resolved warmup design. The output is PLANNED_CPU_ONLY / execution UNEXECUTED / budget NOT_RESERVED / driver_ready false / configuration_frozen_for_execution false. All f/g_mem/g_ssd measurements are null; no top-level curves field is emitted.

## What was reused and verified

The entire fixed original scripts/pareto_measure.py SHA256 47f13dc13abbf042954f5a30f50aa7129b932f3f88fb02fd2b6871277dd4476c is checked before compiling its AST. Only the original Job dataclass (line 114) and build_job_plan (line 144) are executed, with COLD_ONLY=True and baseline-only selection. No copy of the scheduling algorithm was implemented. main, interactive model selection, start_server and run_benchmark are not executed or imported. Only Python stdlib imports occur.

The candidate identity cross-checks historical native-prefix-04 frozen-config/result, its same-run budget runner exit/child exit/timeout/drain/cleanup/GPU UUID, the current environment and dependency locks, the official manifest digest and model identity/local directory, and the small local model config. All referenced JSON records use a single read for parsing and hashing. It does not rehash the 15 GB weights or perform a fresh GPU probe. Historical GPU Prefix qualification is not promoted to current GPU availability or calibration.

The original native-prefix-04 evidence reports GPU Prefix cached tokens 0 then 112, equal output token IDs, actual BF16 model/KV tensors, 28 layers, 73 blocks and 66,977,792 unique storage bytes under a 64 MiB configured pool. The verified geometry gives 57,344 bytes/token. That number describes bytes; no time or cost is derived from it.

Optional --curve-file executes the original, separately SHA-pinned break_even.py load_curves stdlib module, only for a STRUCTURE_ONLY report. Scalar-only/v1 files are rejected even if labeled version 2. Exact local runtime model_name, auto KV string and byte geometry are required; finite/nonnegative times are checked. Even a structurally accepted v2 remains usable_for_planner=false and measurement_provenance_verified=false. Empty/absent golden is allowed by the native parser and does not demonstrate calibration. No interpolation, fitting, planner, GPU or I/O executes here.

## Original entry points, dependencies and remaining gaps

| Original entry | Input/output and actual use | Remaining constraint |
|---|---|---|
| scripts/pareto_measure.py build_job_plan | Config globals -> original Job objects; reused on CPU for baseline cold jobs only | doc_size is nominal, not validated tokenizer length |
| pareto_measure.py run_benchmark/parse_rows | Original streamed HTTP request CSV -> TTFT aggregates | Real requests/GPU required; failures, actual token counts and provenance need explicit evidence |
| common/model_info.py fetch_model_geometry | Existing local path -> small local config geometry; fallback can resolve HF | A fixed local path avoids network; new tool reads the verified small config directly |
| common/vllm_server.py start_server | Launches original vLLM server with start_new_session=True | This escapes the current budget runner session; must use a reviewed same-session adapter before any actual calibration |
| common/prefix_cache_common.py build_prompt (114–136) | request id plus repeated words, reuse suffix words | 64/128/192 candidates are word-derived arguments, not proven token counts; exact-token adapter and chat/completion choice remain unresolved |
| prefix_cache_benchmark.py pre-warmup defaults (187–197) | Five untimed 256-word prompts | Must freeze a bounded warmup explicitly; do not assume prior max_model_len=256 makes it fit |
| scripts/emit_break_even.py main (122–128) | Original measured CSV plus bandwidth -> scalar thresholds | Emits v1 only: no curves, golden or kv_bytes_per_token; no installed v2 exporter was found |
| emit_break_even.py _g_mem_curve (65–80) | g_ssd minus token bytes / measured disk bandwidth, floored at zero | An estimate dependent on real SSD inputs, not a measured RAM staging curve; cannot be filled while those inputs are absent |
| plots/pareto_plot.py build_interpolator (284–295) | Existing measured token/TTFT knots -> SciPy PCHIP | Reuse later only with real samples; no new curve algorithm or fabricated values here |
| py_kvcache/vllm.py _build_planner (663–739) | Complete v2 f/g_ssd/g_mem + positive KV bytes/maxlen + Plan API/preload/shared/lookahead | Full P1 staging/SSD and actual local curves remain blocked |

The current original pareto default configuration is unsuitable unchanged: max model length 92k, large token points, optional huge staging, interactive model choice, default text/chat workload and server session behavior differ from the qualified short GPU smoke. The CPU candidate cap merely keeps nominal arguments below the prior context limit; actual_token_fit_verified stays false.

Cold f(N) is technically independent of io_uring and can be measured later with a reviewed native GPU-only driver, exact input lengths, explicit warmup/repeats, fixed model/backend/engine/launch environment and authorized budget. This work did not run it. g_ssd remains blocked by native io_uring and unqualified production storage path. g_mem cannot be replaced by the native GPU Prefix hot request, a standalone H2D lower bound, a different CPU backend, dummy model output, or zero.

The native v2 reader does not verify schema_version, independent provenance, finite/nonnegative values or golden authenticity. The new optional check does not turn that parser into a scientific validator. A future actual artifact must include explicit original scalar gate fields (omitting them silently defaults thresholds to zero), all three curves, model/GPU/dtype/layout provenance, independent source CSV/trace/bandwidth evidence, supported domain and appropriately sourced golden reference calculations. Runtime linear tail extrapolation differs from SciPy cubic extrapolation; unsupported domains remain explicit, not silently qualified.

## CPU validation and evidence

~~~sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/prefix_io_v1_calibration -q -p no:cacheprovider --junitxml=artifacts/prefix_io_v1/new-server-04/calibration-preparation/cpu-tests.xml
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python artifacts/prefix_io_v1/new-server-04/calibration-preparation/verify_cpu.py
~~~

32 tests passed in 0.18 s. They vary candidate sizes/repeats, reject invalid bounds, failed historical results and runner state, mismatched identities/GPU/model paths/manifest revisions, and reject altered original source before any code executes. v1 relabeling is rejected; explicitly synthetic v2 fixtures are only structure tests. Tests import no Torch/vLLM/FlashInfer and produce no experiment timing samples. Compilation, stdlib-only import review and CLI help also passed.

Evidence: commands.json, cpu-tests.log/xml, candidate-plan.json, verification.json/log, cli-help.log, verify_cpu.py. The existing GPU ledger, locks, permissions, prior run results, source engine and smoke script were not modified.

Final script SHA256: bbea9525726a3e88a6cd2e5e7489cefe77b391606213999ad70737a36a1567f0
Tests SHA256: 5be05c0da892d9b30fdcd0e9c8c091ee6588379d49e9fbdbe7c50fe14125793e
Candidate plan SHA256: ffb164411e2980db5f8cb393c691ed639a35ba3646e99b99645487852ea839ab
