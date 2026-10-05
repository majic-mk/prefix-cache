# GPU runner CPU review and bounded fix

Scope: only experiments/prefix_io_v1/scripts/run_gpu_stage.py and the new
tests/prefix_io_v1_runner/test_run_gpu_stage.py were changed. This review did
not run GPU/NVML/CUDA, download models, modify permissions, or change the real
GPU ledger / previously recorded results.

## Findings and changes

The earlier runner only cleaned up TimeoutExpired, recorded wall-clock elapsed
time, and stored its ledger after process exit. A terminated runner could leave
work unaccounted; a successful direct child could leave descendants running.

The fix serializes through the existing file lock and commits an active
reservation using write/fsync/replace/directory-fsync before creating a child.
An unfinished reservation blocks the next invocation. Runtime and cleanup
charges use time.monotonic; wall time is metadata only. All launched processes
have their original process group inspected and drained for normal completion,
timeout, launch/wait errors, and SIGINT/SIGTERM/SIGHUP/SIGQUIT. TERM then KILL have
bounded grace periods. Live descendants after an otherwise successful parent
produce exit 70 after cleanup rather than false success. Unverified cleanup
keeps an unresolved reservation. Signal exit codes are normalized to 128+signal.

A further launch-window race was identified: raising a signal exception after
the OS creates the process but before assignment to proc loses the handle.
Signals during that narrow window are now deferred until assignment completes.
The child signal mask is not changed. A deterministic CPU reproduction on an
isolated copy with this deferral reverted fails exactly at gpu_job_attempted:
the old code falsely reports False for an already-created CPU child. The proof
test explicitly kills its sleeper in finally if a regression leaves it alive.

## Results and evidence

- first.txt/xml: 15 CPU tests passed.
- final.txt/xml: 18 CPU tests passed after expanded signal and cleanup tests.
- launch-race-final.txt/xml: **19 passed**, 3.55 seconds, final implementation.
- launch-race-before.txt/xml: one expected failure, 18 deselected, against the
  isolated pre-deferral implementation. This is defect evidence, not an
  unexplained production test failure.
- verification.json: forward and reverse git apply --check both passed; original
  GPU ledger, permissions.yaml, and cuda_base_smoke.py have identical before/
  after SHA256 values.
- 0001-durable-gpu-budget-runner.patch contains the complete runner/test changes.
- run_gpu_stage.before.py preserves the initial runner.
- run_gpu_stage.pre-launch-race.py and launch_race_proof.py are isolated defect
  reproduction evidence; do not use them for GPU commands.

Final test command (all subprocesses run Python standard-library CPU code and
use temporary fake permissions/ledgers):
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_runner --junitxml=artifacts/prefix_io_v1/new-server-02/runner-review/launch-race-final.xml
```

The initial patch-verification invocation used a relative .venv-prefix path
without changing to the project and failed before Python ran (exit127). It was
rerun from the correct project directory; no code/test result was fabricated.

## Limits

The runner accepts trusted project commands. A command that deliberately starts
a new session/process group escapes group-based cleanup; no containment claim
is made for such commands. SIGKILL, host failure, filesystem failure, or kernel
uninterruptible states cannot be solved by Python finally. The persisted
reservation blocks further GPU launches until process ownership and elapsed
usage are reconciled. No automatic clearing/reset/recovery is implemented.
Termination grace is conservatively reserved (20 seconds), but OS delays cannot
be represented as an absolute hard real-time guarantee.

The existing cuda_base_smoke.py was read, not modified. It exercises 1024-element
CUDA arithmetic and D2H exact comparison. It does not qualify H2D, pinned staging,
author swap_blocks_batch, copy fences, io_uring, model execution, or Prefix
correctness. Its own vllm_verified/cache_verified false labels are appropriate.
For future portability it should derive its nvidia-smi UUID from the approved
visible GPU and use explicit exceptions instead of assert (assert can be
disabled by -O/PYTHONOPTIMIZE). No prior successful smoke result was rewritten.
