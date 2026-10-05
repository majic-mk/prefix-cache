# C5 server CPU invocation addendum

This addendum records the corrected server fixture provenance and the scope of the disabled path. It does not change the frozen reactor, collector, suite runner, manifest or any earlier evidence. No GPU operation was performed for this addendum.

## Distinct source roots

The old fixture variable `SERVER11_PREVIOUS_CANDIDATE_DIR` and the new runner argument `--previous-root` have different meanings. They must not both point at C4.

| Input | Required server value | Consumer |
| --- | --- | --- |
| `SERVER11_AUTHOR_SOURCE_ROOT` / `--author-root` | Project root below | Original policy/support modules and unchanged original tests |
| `SERVER11_PREVIOUS_CANDIDATE_DIR` | Single-file candidate **v1** | The inherited `test_empty_p4_window.py` reads the original empty-window baseline |
| `SERVER11_IDLE_PREVIOUS_CANDIDATE_DIR` | Single-file candidate **v3** | Frozen C4's retained-idle regression uses the V3-to-C4 comparison |
| `SERVER11_V3_CANDIDATE_ROOT` / `--v3-root` | Single-file candidate **v3** | The unchanged enriched-snapshot test compares 156 versus 2 replacements |
| `SERVER11_PREVIOUS_CANDIDATE_ROOT` / `--previous-root` | Single-file candidate **v4** | New C5 source-scope test compares C4 with C5 |

The frozen runner explicitly sets the author, V3 and new `PREVIOUS_CANDIDATE_ROOT` variables for its children. It preserves the calling environment, so the older `PREVIOUS_CANDIDATE_DIR=v1` must also be supplied by the caller. Its omission was not corrected by changing frozen code.

## Complete server command environment

These are the actual server directories and interpreter reported by the root executor. The reproduction command uses a new output name so that existing failed and successful evidence need not be overwritten.

```bash
PREFIX_PROJECT=/root/autodl-tmp/prefix-io-v1-handoff/project
PREFIX_PYTHON="$PREFIX_PROJECT/.venv/bin/python"
PREFIX_V1="$PREFIX_PROJECT/artifacts/prefix_io_v1/server11-p4-single-file-candidate-v1-20261003"
PREFIX_V3="$PREFIX_PROJECT/artifacts/prefix_io_v1/server11-p4-single-file-candidate-v3-20261003"
PREFIX_V4="$PREFIX_PROJECT/artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003"
PREFIX_C5="$PREFIX_PROJECT/artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003"

export CUDA_VISIBLE_DEVICES=''
export SERVER11_AUTHOR_SOURCE_ROOT="$PREFIX_PROJECT"
export SERVER11_PREVIOUS_CANDIDATE_DIR="$PREFIX_V1"
export SERVER11_IDLE_PREVIOUS_CANDIDATE_DIR="$PREFIX_V3"
export SERVER11_V3_CANDIDATE_ROOT="$PREFIX_V3"
export SERVER11_PREVIOUS_CANDIDATE_ROOT="$PREFIX_V4"

"$PREFIX_PYTHON" -B -I -S "$PREFIX_C5/run_cpu_candidate_qualification.py" \
  --candidate-root "$PREFIX_C5" \
  --author-root "$PREFIX_PROJECT" \
  --previous-root "$PREFIX_V4" \
  --v3-root "$PREFIX_V3" \
  --output "$PREFIX_C5/SERVER_CPU_QUALIFICATION_REPRO.json"

"$PREFIX_PYTHON" -B -I -S "$PREFIX_C5/run_original_cpu_regression.py" \
  --project-source "$PREFIX_PROJECT"

"$PREFIX_PYTHON" -B -I -S "$PREFIX_V4/test_retained_idle_wait.py" -q
```

`CUDA_VISIBLE_DEVICES=''` is an additional CPU-run setting, not itself proof of GPU absence. The selected tests use AST/source extraction and explicitly synthetic event/I/O fixtures, and do not run the model or GPU workload. `-I -S` remains in the actual interpreter command.

## Server results and the retained first failure

The root executor reported the following completed server checks: **106 C5 semantic/regression cases passed, 101 unchanged original policy/ABI cases passed, and 15 retained-idle cases passed against frozen C4**. These are separate checks; the 15 C4 cases are not C5 native integration tests. No older 197/547 totals or native-v6 GPU qualification are inherited.

The first 106-case attempt failed two of the seven inherited empty-window cases because its baseline came from the wrong inherited fixture path. The runner supplied the new C4 comparison variable, but that old fixture reads `SERVER11_PREVIOUS_CANDIDATE_DIR`; explicitly pointing this separate variable at **v1** resolved the two failures. All 106 then passed on the same frozen runtime source. This was a test-invocation provenance correction, not a model/runtime repair, and the initial failed result remains evidence.

The server candidate directory contains these root-reported evidence names:

- Initial attempt: `SERVER_CPU_QUALIFICATION.json` and `C5_SEMANTIC_CPU_*` command/result/output logs.
- Corrected attempt: `SERVER_CPU_QUALIFICATION_EXPLICIT_PROVENANCE.json` and `C5_SEMANTIC_CPU_EXPLICIT_PROVENANCE_*` command/result/output logs.
- The original policy/ABI invocation has a confirmed zero exit code; its captured stream is represented in the stdout JSON from the server executor. This addendum does not invent separate stderr files or additional statistics for it.

The local `LOCAL_CPU_QUALIFICATION.json` predates these server invocations and is not a replacement for them. The unchanged frozen manifest records local results and leaves server counters unset; this appended note explains the newer server execution without silently rewriting that manifest.

## What disabling the new path preserves

The baseline here is **C4 with its shared correctness/idle fixes**, not an earlier buggy revision. Disabling the notification candidate preserves the original native scheduling and cache behavior of that baseline.

- `native_full_step_collector.install()` does not call `attach_single_file_wait()`. Explicit installation requires an active single-file interference bridge; off and shadow installation attempts are rejected.
- Without installation, reactor `_prefix_wait_single_file_retry()` returns `False` at its first missing-reference check. It performs no clock read, policy import, live query, registration, lock acquisition or Queue wait. `_run()` then uses the same C4 `_drain_incoming(block=not _has_poll_work())` choice and the unchanged `_pump_once()`.
- Off's original `_prefix_stage_decide()` fast return remains before optional policy imports, clocks or snapshots. Scheduled-load publication also retains its bridge-disabled return before optional imports or metadata mutation.
- Original reserve/release, prefix-cache ownership, admission, native I/O/copy polling, mandatory/STOP intake tail, shutdown and policy preview methods remain unchanged by the waiting patch. The C4 idle correction and its original 0.5-second idle Queue timeout remain common behavior.
- The collector's notification hooks and observer/bridge failure wrappers are installed only through the explicit attachment. Detach wakes any current arm and restores only wrappers still owned by that attachment; future parking is rejected because the capture is detached. Reconfiguration by arbitrarily mutating private fields is not an added supported API.

This is a **behavioral fallback**, not a claim that C5 is byte-identical to C4 or has exactly zero off-path overhead. The new reactor still makes a small uninstalled-hook call per loop and tests the wake-message type in intake. The new collector also initializes notification metadata/lock and checks whether an end-event notification attachment exists. No off-path notification, new GPU event query or deadline parking is introduced, but any performance effect of these small Python differences requires measurement; CPU tests do not prove GPU equivalence.

## Frozen runtime references checked for this addendum

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py` | 185744 | `a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47` |
| `native_full_step_collector.py` | 24553 | `bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf` |

Only this Markdown addendum was added in this follow-up. No frozen source, test selector, GPU stage or authorization was changed.
