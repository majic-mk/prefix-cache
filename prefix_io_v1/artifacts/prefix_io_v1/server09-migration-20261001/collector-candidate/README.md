# Server09 full step collector CPU candidate

This append-new candidate completes a CPU scalar contract. It installs no runtime hook and is not a GPU collector, a model executor, an I/O engine, or production qualification.

The cloned author runner's original execute_model increments the native ordinal and returns None on the normal generation path. Original sample_tokens later performs sampling, bookkeeping and construction of ModelRunnerOutput. A full step must span those two original calls; forward-only timing cannot be relabelled as full step timing.

P4_FULL_STEP_COLLECTOR_CPU_PLAN.json binds the actual server09 source bytes read during audit. The historical engine configuration has no explicit async_scheduling value; author configuration may default it to True. The planned narrow collector must freeze False for all compared arms and verify the resolved original runner value. Original async I/O remains unchanged.

p4_full_step_frame_adapter.py accepts only compact copied CPU fields from original post-prepare inputs and original synchronous sampled output. It validates contiguous native steps, exact context progression, explicit prefill rows, token ID reconstruction and complete native closure evidence. It records no GPU duration or native per-window I/O; those fields stay None. All results remain gpu_verified=False and production_qualified=False.

The adapter supports the planned ordinary synchronous single TP/PP/DP generation path. Mixed/speculative/async work and unknown geometry are explicitly rejected. Genuine cold-prefill pre_context=0 is recorded. context_length remains prior computed tokens, matching the existing prepared-forward load signature. input_seq_lens_from_cpu_inputs separately records actual computed+scheduled values, which the original _prepare_inputs uses for optimistic CPU and GPU sequence lengths. For a cold 128-token prefill these distinct values are 0 and 128; neither is substituted for the other.

The frozen version 2 verifier requires context_length >= 1 for every complete-trace row. It therefore cannot accept this genuine cold first frame. Before GPU full-trace cost qualification, append a separately versioned context/verifier contract that permits zero only for explicit prefill, retains every original step, and preserves the prior-context cost feature. No version 3 verifier/loader was implemented in this candidate. Omitting the first frame, replacing zero with one, or renaming input sequence length into the prior-context cost dimension is prohibited.

The full cold-run fixture includes the real zero-context first frame and all 128 original output tokens: one sampled by completed prefill and 127 by pure decode. A separate negative fixture proves that completed prefill followed by 128 pure-decode forwards yields 129 actual outputs, so claiming only 128 is rejected. Partial cold prefill with a still-incomplete prompt must emit no token.

The existing original handler's actual shutdown remains the only resource lifetime boundary. Future.done, an empty copy queue, or nominal accepted/completed totals alone cannot establish complete release. Full shutdown evidence must show accounting validity, four-stage accepted/completed equality, exact transferred bytes, zero failed operations, worker join, closed/drained backend, zero outstanding/pending/active/inflight state, and no observation failures. The original completed_requested_bytes counter can equal accepted request bytes even after a short read/write; those counters alone do not prove successful payload completion. These scalar checks validate evidence shape, not its authenticity.

CPU tests run locally: 26 passed, 0 failed. Root must upload the three source/plan files and this note to an append-new server09 artifact directory and run:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S <collector directory>/test_p4_full_step_frame_adapter.py

The server result is pending in this local candidate. No GPU import, model load, GPU reservation, new download, native I/O or source modification occurred during this subtask.

The next implementation is one thin observer around original methods, owner scalar events and original frontend output reconciliation. Before any GPU collection, root must satisfy the current server's exact device/permission/budget gates. Before cost calibration, install and qualify real full step GPU timing plus complete native per-step I/O, reuse the existing version 2 verifier, and freeze AB/BA selections and independent splits. Until then, G2 and P4 remain incomplete.

