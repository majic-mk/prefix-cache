# Reproduce P3 stage accounting and model-start qualification

Server workspace: /root/autodl-tmp/prefix-io-v1-handoff/project.
Native py-kvcache base SHA: 3abba7a502d553f6e7e2e58b92086487e3395d7e.
Author vLLM base SHA: 817a7e3124f817cd6e549581d3e5483207a753a4.
Do not replace the installed binary environment or driver.

The isolated native worktree is third_party/work/py-kvcache-p3-quota-cpu.
Its complete tracked delta is artifacts/prefix_io_v1/server07-p3-09/complete-native-worktree.patch.
The untracked py_kvcache/linux_aio.py is supplied separately as source in this delivery.
Incremental native-stage-accounting and start-options patches are relative to the P308 worktree;
the P308 start-budget patch is not interchangeable with a clean author checkout.

All execution uses PYTHONPATH placing the isolated worktree before src and the experiment scripts.
Exact environment deltas and argv are in *-command.json. CUDA_VISIBLE_DEVICES is empty for CPU;
the budget wrapper binds the approved GPU UUID for GPU jobs.

Do not rerun recorded output labels. All evidence is append-only.
Use a fresh label and preflight both auxiliary capacity and primary log/cache space.
The full CPU runner currently fails its 640 MiB reservation until capacity is restored.
The large mixed model replay currently fails its 3 GiB reservation.
Do not reduce these reservations simply to bypass the guard.

The three model configs off-model-config.json, fixed-model-config.json and pressure-model-config.json
are frozen integration candidates. Their source hashes and the preceding GPU diagnostic receipt
are checked by start_budget_model_contract.py. A change in source requires requalification.
The timing table is a one-trial descriptive result, not a performance acceptance criterion.

The exact archival deduplication manifest and proposal are included.
The --apply command must not be run without a scoped authorization record bound to the manifest SHA.
No such approval record was created in this delivery.

Large KV data and CPU scratch remain on the server. The delivery includes code, patches, commands,
JUnit, GPU ledger records, model outputs/token timestamps, native traces and inventory hashes.
It excludes model weights, native binaries, large test scratch, and cached KV payloads.
See docs/prefix_io_v1/SERVER07_P3_STAGE_ACCOUNTING_MODEL_REPORT.md for interpretation and
docs/prefix_io_v1/SERVER07_P3_ARCHIVE_CONSOLIDATION_PROPOSAL.md for the pending archival action.
