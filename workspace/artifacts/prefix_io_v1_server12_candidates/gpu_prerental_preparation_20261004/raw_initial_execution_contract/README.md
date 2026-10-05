V7 appends an exact initial-execution contract and immediate completed-window
validation. The separate frontend cached-token statistic remains 512. The
initial_execution_pre_context and descriptor cached_prompt_tokens are 496,
so the original validator requires 16 original prefill tokens on the first
model frame. The selected offset16 remains a decode frame with context527.

The real cal04 A and B captures each contain128 model frames and128 CUDA
witnesses. Replaying the immutable native validator rejects the old511
initial-execution contract and accepts496. Both original post-shutdown
drain and original validate_io pass, including action B's original causal
SSD acceptance and independent-payload checks. No elapsed values are exported
and no estimator, heldout fit or new GPU qualification is run.

cal04's interrupted outer guard is not a completed qualification receipt.
Its two historical windows are metadata evidence only and are never reused
as fit inputs for the next run. V7 requires fresh six-window acquisition,
the original successful outer guard and source closure before issuance.

The parent now invokes _strict_completed_window immediately after each real
closed child row is appended. The callback binds original external/native
request IDs and unchanged inputs/order, calls the unchanged validate_capture,
original_post_shutdown_drain and validate_io, then returns. Any capture,
I/O, native-tail or child failure prevents the next process from starting.
It records only completeness/contract fields and never computes costs.

Only the initial-execution constant, source_patch and configure_original
change from V6. The executor, native cache, collector, old wrapper files,
validator, numerical estimator, AB/BA/AB order, two fit families and independent
heldout family,128 outputs, frontend512 and source ownership stay unchanged.

CPU tests run the actual callback and original patched parent AST. The parent
subprocess API is replaced with explicit CPU refusal stubs; no model process
starts. Mutated failure cases are clearly CPU tests, never GPU receipts.

Server CLI, from the project root (D is preparation; A/B are the existing
actual cal04 windows00/01 native-cost-runtime-result.json, unchanged bytes):

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/raw_initial_execution_contract/test_initial_execution_contract_cpu.py --actual-child "$A" --actual-second "$B"
    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S $D/runner/strong_native_cost_runner_v7.py --help

Historical actual input SHA pins: A a8e1bba92e987335967760c0384be36dcf5d5e483eaf8c5562aea52512808bdd;
B c2b90512467b001b6ba3440c77f329745dabf8e26c391ce42e776f5327b6d05a.
