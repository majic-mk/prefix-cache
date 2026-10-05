# P4 V2 bounded paired measurement selection (CPU preparation only)

V1 remains the historical CPU fixture contract. V2 does not grant GPU origin, full-decode-step qualification, production admission or an interference allowance. The original candidate schema stays version 1 with the same five measurement roles. The public builder/loader APIs stay unchanged.

## New exact version 2 plan

All original plan fields remain. Add exactly:

- selections: exact cell-id map. Each cell contains load (active_decode, batch, prefill_tokens, context_length), warmup_step_offsets and measured_step_offsets.
- timing_contract: timing_scope (model_forward or full_decode_step), clock_domain=cuda_event_elapsed, reference_source_ref (actual byte-bound EvidenceRef), reference_valid=true, fallback_used=false, clock_domain_valid=true.

Offsets are nonempty, sorted, unique, bounded 0..4095. All warmup offsets precede all measured offsets. Selection is independently pinned before measurement. The measured load must equal the original exact9 candidate cell. Warmup and other full-trace steps may have their actual different contexts; they do not enter the estimator. No adaptive selection by observed duration or fabricated same-context warmup is supported.

The existing plan-wide calibration/margin-validation trace, prefix-family and workload hashes remain disjoint. Same-content per-cell pairs cannot be made independent by changing seed/trace/family labels. Validation participates in maximum positive residual; it is margin-validation, not an independent effect evaluation or confidence interval.

## Wrapper version 2 and complete trace binding

The wrapper top-level fields/roles remain. Each run keeps actual output_tokens and adds:

- complete_trace_ref (EvidenceRef to one complete step/output JSON).
- first_step_ordinal (actual native runner ordinal), full_steps, full_output_tokens.
- selected_output_tokens: actual sample output count only on preregistered measured steps.
- accepted_new_io; completed_new_io remains and must match the entire run accepted total.

output_tokens equals full_output_tokens and must remain the real complete output, for example 128. It is never rewritten to the selected 1 or 2 output tokens. warmup_windows/measured_windows separately equal the frozen selection counts. AB/BA order, actual host run boundaries, exit 0 and accepted_io_drained=true remain mandatory.

Complete trace exact schema:

schema_version=2; scope=p4_complete_step_output_trace; origin; context; arm; cell_id; pair_id; steps; outputs; outside_window_new_io; accepted_new_io; completed_new_io; accepted_io_drained.

Each step has exactly step_offset, native_step_ordinal, start_ns, end_ns, load, existing_io, new_io, outputs, timing. step_offset is its contiguous run-relative array index. native_step_ordinal equals first_step_ordinal+step_offset and preserves the actual runner _profile_step ordinal (0-based), rather than a guessed decode counter. Every step in the run must be present in order. Host start/end are non-overlapping monotonic call boundaries inside the actual run; they are not a GPU time axis.

Both complete outputs and per-step outputs are lists of {request_id, token_ids}. Every complete request is covered. Concatenating per-step sample token IDs for each request must equal its complete output IDs exactly; paired baseline/action complete output IDs must also agree. The real output IDs must come from the original sampling/driver output capture. A model-forward observer has output_tokens=None and cannot produce or infer this evidence.

All four additional-I/O stage counters are accounted as the sum of every complete step.new_io plus outside_window_new_io. That actual accepted vector must equal the complete trace and wrapper accepted/completed totals with full drain. Nonselected/warmup I/O cannot disappear from the total. Baseline experimental extra I/O is zero throughout. Each selected action step must separately equal the frozen one-stage operation/physical-byte geometry. Four-stage accounting does not introduce new scheduler actions or qualifications; scope stays unchanged. Raw totals remain declarations whose CPU consistency is checked, not trusted proof of real native execution.

Within each cell/arm, all complete steps across paired runs are bounded by 4096 and all output IDs by 262144. Existing bounded JSON/ref limits apply.

## Selected observation version 2

Keep the original selected row fields and add step_offset, native_step_ordinal, timing. window_id is step-{offset}; phase derives from the frozen selection. Each selected row must be an exact, strictly typed projection of its complete trace step for boundaries, actual ordinal/load, I/O and timing. output_tokens is recomputed from that step's actual sample token IDs. Every selected and warmup offset must be present exactly once; other offsets cannot be substituted.

## GPU duration field and source

Each timing object adds gpu_elapsed_ns to the six fields of the pinned timing_contract. Exact scope/domain, valid reference, no fallback and source EvidenceRef are mandatory. Only this explicit positive GPU-event elapsed duration enters V2 costs. Host end_ns-start_ns never substitutes for it and no host journal/GPU-domain overlap proof is inferred.

A source-bound model_forward observation remains model_forward. It cannot silently be renamed full_decode_step. Even an internally consistent raw claim of full_decode_step/native_gpu_recording remains CPU semantic preparation with production false; external trusted execution/qualification is still missing.

The current CPU observer seam provides frame.pre_context/exact9/native_step_ordinal, scope=model_forward, gpu_elapsed_ns, reference_valid/fallback_used/clock_domain_valid and host_start_ns/host_end_ns. Its output_tokens/post_context remain None. A caller may explicitly map host_start_ns/host_end_ns to V2 host boundaries; cross-clock mapped start/end fields are not required and cannot replace elapsed duration. Real complete original-sample outputs, final accepted-I/O drain receipts and source binding remain required outside that seam.

## Single estimator and append-only builder

The same original calculator computes upward calibration run means, upward mean paired delta and maximum positive action-step residual over calibration plus margin-validation. Only the duration accessor changes for V2. Builder emits recomputed analysis schema_version=2 and the original candidate-cell mapping. Loader rebinds all five roles, each nested complete-trace ref and timing source bytes before/after recomputation. PreparedCostTable production lookup always returns None and the original CostTable remains mock_only.

Public APIs remain:
load_verification_plan(root, plan_path, expected_plan_ref=...)
build_paired_cell_candidate(root, cell_geometry=..., raw_refs=..., expected_plan=..., analysis_path=...)
load_semantically_verified_table(root, candidate_path, expected_context=..., qualification_ref=..., expected_verifier_ref=..., plan_path=..., expected_plan_ref=...)

No V1 evidence is converted into V2 and no old aggregate is backfilled with unknown step/output/event data.


## Final non-speculative output boundary and offline CLI

The selected domain is strictly pure decode: active_decode=batch>0, prefill_tokens=0. Every actual pure-decode step in the complete trace, including nonselected steps, must contain exactly one sampled output token for every active request. Every request's per-step token_ids contains at most one token; step output count cannot exceed batch, and actual step batch cannot exceed the complete run request_count. Moving all 128 full output IDs into one selected step is rejected, even if the complete-output reconstruction and all raw refs are internally consistent.

Pure-decode full-trace rows keep the exact original V2 fields, which preserves the frozen fixtures' byte replay. A non-pure full-trace row is accepted only with the additional explicit step_kind="prefill", actual active_decode=0 and actual prefill_tokens>0; it is a complete-run prelude record, cannot be selected for this cost cell and never enters the estimator. Its output count remains at most one token per request and at most its actual batch. Event/source/complete-sample/drain requirements are unchanged. Mixed, speculative, dummy, async and unknown traces are unsupported and rejected. This prelude support provides no prefill cost cell or new qualification.

experiments/prefix_io_v1/scripts/prepare_p4_raw_pair.py now accepts either the unchanged strict V1 observation bundle or exactly this V2 bundle:

schema_version=2; scope=p4_explicit_pair_observations; context; cell_id; origin; cell_geometry; raw_refs.

cell_geometry contains exactly cell_id/load/existing_io/stage/physical_bytes. raw_refs is the four typed EvidenceRefs to baseline_wrapper/action_wrapper/baseline_observations/action_observations. The plan version/context/cell and every raw header version/context/cell/origin/arm must agree. Each wrapper's complete_trace_ref is byte-bound and checked. Source/reference bytes are rechecked after assembly.

The CLI does not collect model outputs, events or drain evidence. It ingests already complete raw bytes through the existing PreparedRawPair/write_raw_pair/assemble_raw_pair pipeline. That pipeline calls the original single builder/estimator and public final loader, producing exactly five candidate measurement roles and a CPU-only receipt. No duplicate arithmetic or new collector is introduced. V1 raw_recorder and assembler source are unchanged. V1/V2 bundle, plan, role or complete-trace version mixing is rejected.

Real offline CLI subprocess evidence is fixtures-v2-cli-01: two positive copies of the prior byte-frozen CPU fixture sets (128 complete outputs with selected1/2), and negative copies for source-byte change, full-trace-byte change and V1-role/V2-bundle mixing. Original fixtures are unchanged. The copied authorization/permissions are original CPU-only metadata; no permission was expanded. Each process denies backend imports, records 0 attempts/0 GPU and invokes the real script. These remain CPU fixture runs, never GPU experiments.
