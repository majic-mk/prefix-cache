# CPU mandatory bridge coverage addendum

This supplements TEST_MATRIX.md; no T01–T36 GPU acceptance is upgraded.

| Evidence | Covered here | Still unverified |
|---|---|---|
| Wait before blocking / progress | Native handler→coordinator→reactor with test-only zero-credit seam, no new scheduler epochs, one/multiple parents | Real vLLM scheduler/worker wait and ordinary quota implementation |
| Physical safety | Native slot capacity and simulated compute/D2H event dependency | Actual CUDA event timing, tensors and streams |
| Store continuation | Native D2H-done→queue_write and completion path | Real io_uring and SSD persistence |
| Failure/drain | Native short-write failure settles Future early; another accepted write drains before marker cleanup | Real device failure/cancellation and scheduler source fence retirement |
| Completion identity | Duplicate simulated CQE, Future identity, reactor token and stale integer job ID | Runtime restart/integration protocol |
| Shutdown | Native STOP and shutdown without further scheduler epochs | Full vLLM teardown with live device I/O |
| Off | No signal state/queue marker, actual native CPU-backed event-loop run | Real output/token/latency equivalence |
| Fatal termination | Drop only added signal references; preserve native failure semantics | No recovery/release guarantee added |
| Observation | Existing immutable bounded snapshot runs in actual reactor thread with simulated CUDA/file backend | Live hot-path overhead and scheduling connection |
| Baseline preservation | 261 original author tests pass; common tests plus bridge tests total 351 passes | 5 GPU and 3 io_uring tests skipped |

The zero ordinary-credit hook exists only in tests/prefix_io_v1_progress/_cpu_backend.py. Do not report a production interference controller or mandatory bypass as implemented from this evidence. Bridge production default is None/off; no deployment enables it yet.
