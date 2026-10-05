V6 changes only prepare_plan from the sealed V5 wrapper. The required
collector_source_ref selects runner/bounded_native_full_step_collector_v2.py.
The additional required original_collector_source_ref selects the immutable
runner/bounded_native_full_step_collector.py delegate and checks its exact SHA
9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d.

Both files must belong to the complete source lock and pass original byte
checks before writing a plan. V5 diagnostics and every other wrapper method,
strict 128-frame assertions, executor, cache, estimator and shutdown are kept.

The stdlib CPU test executes the actual prepare_plan AST with actual original
ref/check_ref/read/new_json functions. Its surrounding lock and pair validator
are explicitly CPU fixtures, so no resulting plan is a runtime qualification.
It covers both leaf selection, full=True, missing leaves, actual byte drift,
changed-and-relocked original delegate, and null GPU/nonissuable outputs.

The optional --structure-only mode uses a clearly marked pending new-source
fixture when V2 does not exist; it cannot declare final two-leaf verification.
The final default invocation requires the real V2 bytes and its delegate pin.

Server commands from the project root (D is the preparation directory):

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/raw_no_forward_binding/test_collector_binding_cpu.py
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/runner/strong_native_cost_runner_v6.py --help

No GPU, RPC, model, CUDA event, deletion or private qualification is performed
by this delivery. The real completed guard and before/after source closure,
128 actual continuous frames and independent heldout remain prerequisites.
