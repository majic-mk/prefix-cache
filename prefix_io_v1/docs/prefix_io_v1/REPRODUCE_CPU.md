# CPU reproduction

Run in /root/autodl-tmp/prefix-io-v1-handoff/project:
```bash
bash experiments/prefix_io_v1/scripts/reproduce_cpu.sh
```

The script preserves this delivery's logs and creates a fresh artifacts/prefix_io_v1/cpu-replay.* output directory. It explicitly disables CUDA visibility and GPU E2E tests. Expected in this container: 328 passes, 8 skips (5 GPU, 3 io_uring EPERM); a capable different environment can change skip counts and must be reported honestly.

For a repeated P0 audit use a fresh output, e.g.:
```bash
CUDA_VISIBLE_DEVICES="" .venv-prefix/bin/python experiments/prefix_io_v1/scripts/audit.py \
  --output artifacts/prefix_io_v1/p0-repeat
```
Existing capability evidence is not overwritten. The repeated audit is source/runtime qualification only and generates its own source lock; it does not overwrite the final patched dependency lock.

The existing CPU venv uses preinstalled Torch 2.8.0+cu128, NumPy 2.3.2, PyYAML 6.0.2; local additions include pytest 9.1.1, SciPy 1.18.1 and editable simple-profiler at its pinned source. Full package versions are in artifacts/prefix_io_v1/p1/python-packages.json. It inherits system site packages; this is documented, not presented as a fully hermetic runtime.

To reconstruct source dependencies on another approved project workspace, fetch exactly the commits in dependency-lock.json into isolated directories. Clone/fetch author vLLM from https://github.com/t348575/vllm at 817a7e3124f817cd6e549581d3e5483207a753a4, not arbitrary current upstream. Preserve a clean py-kvcache checkout and create a worktree at its locked SHA. Apply common/0001 then common/0002 with git apply; both forward and reverse checks were verified. Set PYTHONPATH to src and that worktree for CPU tests.

The delivery ZIP contains only project additions, common patches and evidence, not the large original dependency sources or installed environments. No GPU launch script is enabled. GPU_STAGE_REPORT.md states the missing prerequisites.

## Continuation 01 isolated progress worktree

Apply observer/0001-mandatory-progress-bridge.patch after both common patches in a separate py-kvcache-progress worktree. Run bash experiments/prefix_io_v1/scripts/reproduce_progress_cpu.sh. It uses --import-mode=importlib to isolate project/upstream tests package names and treats unhandled thread exceptions as errors. Expected here: 351 passes, 8 skips. This is CPU/mock evidence; production default is disabled. The original reproduction command still targets only the common worktree.

## Continuation 02 isolated observer worktree

Create py-kvcache-observer at the pinned SHA and apply common/0001, common/0002, observer/0001, observer/0002 in order. Run bash experiments/prefix_io_v1/scripts/reproduce_observer_cpu.sh. Expected here: 385 passes and 8 skips, followed by an explicitly CPU-only observer microbenchmark in the same new evidence directory. New worktree defaults to no observation sink. Project-only delta against continuation 01 is artifacts/prefix_io_v1/continuation-02/project-changes.patch; the ZIP also contains final source files.

## New server 01 review correction

The same observer CPU reproduction script now expects 388 passes and 8 skips in the new container after three owner-guard regression cases were added. CPU testing continues to use .venv-prefix. A distinct .venv contains future author-runtime dependencies only; do not mistake its successful pip check for GPU qualification. Review evidence and dependency freeze are in artifacts/prefix_io_v1/new-server-01.
