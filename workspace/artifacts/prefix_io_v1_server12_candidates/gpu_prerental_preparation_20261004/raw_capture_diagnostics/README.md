V5 appends failure diagnostics to the immutable original raw driver functions.
Only the wrapper's source_patch function changes from V4. The original strict
128-frame assertion, scalar adapter/source identity, exact offsets, executor,
native cache ownership, heldout split, estimator and shutdown remain intact.

Before the strict assertion, a new append-only preassert diagnostic contains
the actual frontend output, actual capture export, initial/current adapter
identities and source binding, frame count, copied prepared frames, pending
prepared state, and actual observer/scalar/event reasons. The finally path
saves those diagnostics before the original detach. Already exported capture
data is reused without an extra event query. If export or diagnostic writing
fails, the bounded error is retained and original cleanup still runs once.

The CPU tests compile the actual patched original assertion/except/finally AST
and run clearly labelled CPU metadata stubs. They cover short capture, changed
adapter/source, export failure, frontend failure and diagnostic write failure.
They do not import a framework, run a model/subprocess, record a CUDA event or
issue a qualified table. These tests prove persistence and cleanup wiring only.

Server commands, from the project root (D is the actual preparation directory):

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/raw_capture_diagnostics/test_failure_diagnostics_cpu.py
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/runner/strong_native_cost_runner_v5.py --help

cal03's failed frame payload was not preserved by the previous driver. Its
specific bad frame cannot be reconstructed or claimed from these CPU tests.
The next real guarded run can preserve that evidence. V5 does not change or
qualify the currently sealed collector; a different collector needs its own
CPU proof and source pin before it can replace the plan's collector_ref.
