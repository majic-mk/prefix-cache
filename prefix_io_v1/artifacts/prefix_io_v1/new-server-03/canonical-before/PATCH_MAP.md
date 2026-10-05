# P0 patch map
Frozen sources remain clean. Modified py-kvcache: third_party/work/py-kvcache, based on 3abba7a502d553f6e7e2e58b92086487e3395d7e.

Common CPU repairs: transfer.py imports / ParsedKvLayout.allocate_staging_buffer retain CPU Torch without claiming GPU copy capability; staging.py budget helpers and reactor.py IoReactor.__init__ reject actual backing over-budget before allocation and record storage bytes. All experiment arms share these fixes. Project config/preflight rejects malformed/null active configs and fallback runtime capabilities.

Observer/progress preparation: a later handler.wait bridge must signal existing parents before futures_wait; reactor-owned immutable snapshots must distinguish failed Future from completed drain. No second I/O queue or ownership system. Do not activate quotas before mandatory progress passes.

Research policy remains gated by P1–P3. Reserve off/shadow/fixed/pressure/interference/dependency_only/joint config names, but refuse live activation without prerequisites. No speculative calibration values.

Unmodified: Prefix identity, LoadPlanner, allocator, attention, model/decode executor, liburing executor, D2H→SSD continuation, parent success protocol and swap_blocks_batch fusion. Common / observer / policy patches are separate. No Source/repair/Prefix-shadow code is imported.

## Continuation 01 — isolated CPU preparation

observer/0001-mandatory-progress-bridge.patch adds optional progress_run_id and native-queue Future markers in py-kvcache-progress. It changes handler.wait, constructor setup, intake and signal-reference cleanup only. Original common worktree stays separate; normal construction defaults to disabled. No new ordinary quota or policy is active. See CONTINUATION_01_REPORT.md and native-path-preservation.json for exact evidence.

## Continuation 02 — optional bounded observation

observer/0002-bounded-snapshot-hook.patch adds optional observation_sink to IoReactor and TransferCoordinator, then invokes it after the native pump. Default is None; ordinary dispatch/completion/fusion methods remain unchanged. Project publication.py supplies a rate-limited latest immutable snapshot with nonblocking publication. dependencies.py now requires native owner confirmation before counting immediate reusable bytes. No live GPU owner adapter or ordinary quota was enabled. See CONTINUATION_02_REPORT.md.

## New server 01 publisher review fix

Project-only correction moves owner/thread validation into the existing publisher invalidation guard. Three prior-failing tests now pass. The patch and originals are preserved in new-server-01/review-fix and review-fix-before. Author native sources and GPU lifecycle remain unchanged.

## New server 02 — source build and common UUID compatibility

Author vLLM is built from 817a7e3124f817cd6e549581d3e5483207a753a4 in third_party/work/vllm-author-build. Original author sources remain clean. No C++ kernel, model executor, Prefix identity, admission, preload, fusion or reactor implementation was replaced.

common/vllm-author/0001-full-gpu-uuid.patch modifies only NvmlCudaPlatform.device_id_to_physical_device_id in vllm/platforms/cuda.py: a complete GPU UUID is resolved through NVML before physical-index use. Numeric, empty and unset CUDA_VISIBLE_DEVICES follow the inherited path; invalid UUIDs fail instead of selecting another device. This is a common compatibility fix for every experiment arm, independent of research policy. CPU regression: 16 passed; real specified-GPU metadata/import checks passed. Forward/reverse apply and pinned Ruff checks passed. Evidence: artifacts/prefix_io_v1/new-server-02/platform-fix.

Project run_gpu_stage.py now reserves and accounts wall time before/after each launch and drains the complete child session, including nested process groups. Twenty-one CPU tests cover budget refusal, interruptions and descendant cleanup; uncertain cleanup/reservations fail closed. This runner does not enforce model download bytes or network isolation. Offline HF/Transformers settings apply to launched GPU jobs. Evidence: new-server-02/runner-review.

Build/qualification helpers under experiments/prefix_io_v1/scripts are experiment infrastructure. The small author copy test proves its explicitly recorded byte buffers only. Observer patches remain in separate inactive worktrees; policy_patches is empty.
