# GPU stage: BLOCKED

No GPU operation, inference request, CUDA allocation, model download or performance experiment was run. Budget consumed: 0 GPU hours; 0 GiB model downloads. CPU packages and public source downloads are separate.

Blocking facts:
1. permissions.yaml has allow_gpu_runs=false and null budget; GPU IDs and approved experiment/dependency roots are unset.
2. The container exposes no /dev/nvidia* devices and has a 2 GiB cgroup memory limit.
3. io_uring_setup fails with EPERM. O_DIRECT's 4 KiB CPU round trip succeeds, which does not bypass this blocker.
4. The actual vLLM runtime, copy kernels, handler and full integration are not installed/qualified. Source-level Plan API presence is insufficient.
5. Model/tokenizer revisions, actual GPU/staging budgets, cost curves, traces, SLO and calibration remain unfrozen.

The following safe preflight command was executed and correctly returned exit 2 / BLOCKED:
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
PYTHONPATH=src .venv-prefix/bin/python -m prefix_io_control.preflight \
  --controller docs/prefix_io_v1/templates/controller_spec.yaml \
  --permissions experiments/prefix_io_v1/configs/permissions.yaml \
  --capabilities artifacts/prefix_io_v1/p0/capability-report.json
```

Next allowed stage is still P1: restore the locked author combination in a user-approved capable environment, keep all arms on the common fixes, and qualify real imports/handlers and io_uring before any GPU smoke. GPU execution additionally requires explicit permissions, finite time/download budgets, GPU IDs and isolated roots. No current instruction authorizes altering container/system policy.

After P1 passes, implement/verify the mandatory bridge and live snapshot adapter in P2 with policy off/shadow. Only then enter P3 simple baselines/calibration/pilot; P4 active policies depend on its results. P5–P7 have not started.

The author's actual later benchmark entry point is `bench.py --config <frozen JSON>` in kvcache-experiments; pareto calibration is `python -m scripts.pareto_measure --config <frozen JSON>`. No frozen runnable GPU config can honestly be supplied before the null permissions/environment/model/budget fields are decided. Do not use its wipe-shared-storage option. The shipped preflight is executable now and cannot launch the benchmark.

## Continuation 01 status

The optional mandatory signal bridge now exists in an isolated CPU preparation worktree and passes 23 CPU/mock tests (351 combined passes, 8 skips). The paragraph above describing future implementation is the first-delivery status; real-backend bridge validation and live observation still remain outstanding. Permissions and budgets are unchanged. No GPU work was performed; P1 remains the next acceptance gate.

## Continuation 02 status

Bounded optional owner-thread observation/publication now has CPU/mock coverage and a CPU-only microbenchmark. Combined tests: 385 passed, 8 skipped. The 0.5-CPU cgroup microbenchmark uses synthetic state and does not measure GPU/end-to-end overhead or satisfy the 2% target. Permissions remain unchanged, GPU usage remains zero and P1 remains blocked.

## New server 01 — device present, execution still gated

Latest host: connect.westc.seetacloud.com:24801. It exposes RTX 5090 device 6, with 90 GiB cgroup memory and 25 CPU quota; earlier no-device/2-GiB observations above are historical. Actual io_uring setup still fails EPERM; O_DIRECT passes. GPU/model budgets remain unapproved and no GPU operation ran. The independent .venv now contains Torch 2.11.0+cu130 and the 191 resolved runtime dependencies; author vLLM is not built/installed or qualified. See NEW_SERVER_01_REPORT.md and PLATFORM_IO_URING_REQUEST.md.
