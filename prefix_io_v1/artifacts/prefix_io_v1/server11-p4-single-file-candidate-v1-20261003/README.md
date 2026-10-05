# Single-file conditional I candidate

This is a CPU-tested candidate overlay, not a general production interference
table, a GPU qualification receipt, a D/J implementation, or evidence of speedup.
Historical v1–v5 calibration files and original source trees are unchanged.

## Scope and binding

The original `P4Policy.issue_preview` and `NativeP4Bridge.preview_issue` remain
the only advice path. The candidate adds one private, source-bound receipt for
the actual v5 single-file calibration. The factory reserializes original raw
receipts and reruns the frozen native verifier; it does not accept a reported
PASS. Ordinary cost-table production lookup and batch qualification stay closed.

The supported action is one original preload file (917,504 bytes), with zero
existing I/O in all four native stages, batch 1, active decode 1, no prefill,
and actual pre-context 144. The actual model, GPU, KV geometry, eager Triton
execution and loaded source identities are checked at attachment. Calibration
common-source identities and the new runtime overlay are recorded separately.

The engineering budget uses only calibration A: baseline plus the largest
positive A residual. For v5 this is 12,999,104 ns; the qualified action upper
cost is 24,998,528 ns. These numbers are a finite development condition, not a
business SLO or a probability guarantee. Heldout is never used to choose budget.

`native_full_step_collector.current_single_file_step` borrows actual prepared
scalars and original event metadata. It creates no new CUDA events, queries,
waits or synchronization. A real start query must have completed. The value
becomes unavailable at scalar close, before original end-event recording,
on observer failure, or on detach. The bridge borrows the runner weakly.

The native ready descriptor supplies the existing `open_start_ns`, sequence,
hash and `_PreloadInfo`. The timestamp is unchanged across retries of that ready
descriptor. A re-opened descriptor has its original new native open timestamp;
this is not claimed as a global request arrival/deadline. Only a one-file
preload request is eligible. No parent, queue, allocation or completion owner is
invented. Mandatory, support, continuation, shutdown and original age overrides
remain. Unknown, stale and out-of-condition values use U.

The experimental shadow switch is `bridge.single_file_shadow=True` while the
existing policy mode is `interference`. The mode name is not a qualification.
Shadow records proposed deferrals but returns the original issue path. On
records actual native blocked attempts separately. At most eight distinct
step/work records retain first/last defer times and counts; all reason counts
are bounded metadata. Final verification must bracket actual decision times
against the original event witnesses. This is causal host containment, not an
exact absolute CUDA clock or a long-term allowance guarantee.

## Deployment and remaining checks

Copy the complete original control/native packages into a NEW private runtime
tree, excluding pycache, then apply only these source overlays. The partial
source directory alone is not an importable replacement package. Reuse the
original author vLLM. Freeze the actual loaded paths/SHA values; never report the
old reactor SHA as the running patched reactor. Include receipt binding,
collector, runtime identity helper and every overlay in the new runtime lock.

Construct the bridge with `single_file=load_verified_single_file(root,
binding_relative)` and an exactly matching `P4Config.internal_step_budget_ns`.
Keep parent admission and observation identical in off/shadow/on arms. Inject
the bridge only into the original TransferCoordinator. Attach after installing
the capture and before its preload callback:

```python
identity = verify_actual_single_file_runtime(
    worker, handler, receipt, root, refs, common=common, base=base)
bridge.attach_single_file_capture(
    capture, runtime_identity=identity, runtime_refs=refs, shadow=True)
```

Off installs no bridge. Run source checks and the original server CPU regression
before GPU use. Then validate the new overlay off/shadow before any on run,
including full output, whole 128-step timings, cost-envelope compatibility,
all native accepted/completed events and original shutdown/drain. The v5
serializer cannot validate delayed I/O as though it all occurred at offset 16.

On may only shift I/O into the next decode step. Compare all 128 ITLs/steps,
full request/cohort duration, I/O completion and final drain. A smaller selected
step alone is not system improvement, P4 completion, or P5 evidence.

## CPU validation

Local Python 3.12, `-B -I -S`, no backend imports or GPU operations:

* Original unchanged policy/bridge ABI tests against the overlay: 101 passed.
* Conditional policy, current-step expiry and shadow/actual decisions: 14 passed.
* Original native decision/queue AST, age, progress and off behavior: 10 passed.
* Source-bound receipt counterexamples: 13 passed.
* Actual-runtime metadata counterexamples: 5 passed.

Total: 143 passed, zero failures/skips. These are CPU protocol tests only.

For the new server, set `SERVER11_AUTHOR_SOURCE_ROOT` to the original cloned
project root for `test_single_file_policy.py` and `test_reactor_single_file.py`.
The receipt/runtime-helper tests need no environment variable. The original
pure regression runner supports `--project-source ORIGINAL_ROOT` and repeated
`--test relative/test_path.py`; the default two original test names are
`tests/prefix_io_v1_p4_policy/test_p4_policy.py` and
`tests/prefix_io_v1_p4_bridge/test_value_abi_independent.py`. Use the current
server's corresponding cloned test paths if a directory was renamed. The root
workflow also runs original full native P4 CPU regressions against the complete
private package copy; this local pure test result does not substitute for them.
