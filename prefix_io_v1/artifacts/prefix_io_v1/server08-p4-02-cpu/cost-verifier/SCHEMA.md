# P4 paired measurement CPU semantic contract v1

This is an offline, bounded semantic verifier and a prepared loader. It never
proves that raw timestamps happened on a GPU. Both allowed origin labels
(cpu_fixture and native_gpu_recording), a matching SHA-256 chain and PASS_REPORTED
remain unqualified. Production lookup returns None. Existing CostTable, candidate
parser, P4 policy and all native owners/queues are unchanged.

API:
- load_semantically_verified_table(root, candidate_path, *, expected_context,
  qualification_ref, expected_verifier_ref, plan_path, expected_plan_ref)
  returns PreparedCostTable.
- PreparedCostTable.verification is an immutable semantic result.
- PreparedCostTable.cpu_mock_table is an existing CostTable(scope="mock_only").
  It is for explicit CPU fixtures, never native production activation.
- write_semantic_verification_manifest(root, verification, relative_path)
  writes a new receipt exclusively, rechecks source bytes and returns EvidenceRef.
- A candidate and its raw inputs can live below a project root. Every ref is an
  exact bounded relative path, byte count and lowercase SHA-256. Existing parser
  rejects symlinks, traversal, stale digests and duplicate JSON keys.

Frozen plan:
schema_version=1; scope=p4_paired_semantic_plan; exact TableContext;
workload_split_ref; action_operations[cell_id]; min_calibration_pairs>=2;
min_validation_pairs>=1; max_pairs=64; max_windows=4096;
formula=ceil_calibration_means_plus_max_positive_residual_v1.
Only uncertainty_method=paired_residual_margin is implemented.
paired_interval_margin is rejected because no interval/bootstrap contract is
being invented here.

Frozen split:
schema_version=1; scope=p4_frozen_measurement_split; entries with
cell_id, trace_sha256, prefix_family_sha256, seed, split, workload_sha256,
input_tokens, output_tokens, request_count. Raw wrapper workloads must exactly
cover the frozen entries. Validation trace and prefix family cannot occur in
calibration. Trace+seed pairs are unique independent run units.

The original five candidate measurement roles are preserved:
baseline_wrapper, action_wrapper, baseline_observations, action_observations,
pair_analysis. Semantic v1 requires JSON object metadata (bounded 2 MiB each),
not loosely interpreted JSONL or client chunk events.

Both wrappers have exact schema_version, scope=paired_measurement_wrapper,
origin, context, arm, cell_id, runs. Each run has pair_id, trace_sha256,
prefix_family_sha256, seed, split, arm_order, warmup_windows, measured_windows,
input_tokens, output_tokens, request_count, workload_sha256, start_ns, end_ns,
exit_code, accepted_io_drained, completed_new_io. Matched A/B work must agree;
normal exit=0 and drained=true are mandatory. Calibration AB and BA counts are
equal, timestamp order agrees, and independent run intervals cannot overlap.

Observation objects have exact schema_version, scope=paired_window_observations,
origin, context, arm, cell_id, windows. Each row has pair_id, window_id,
phase=warmup|measured, start_ns, end_ns, load, existing_io, new_io, output_tokens.
Load has exactly active_decode, batch, prefill_tokens, context_length.
This remains the prior exact9 model/GPU/layout/kernel/active-decode table.
It does NOT alias scheduled-work-v1 producer counts to active GPU decode.
The outer real GPU qualification must separately establish native step events
and actual active decode/context rather than relying on these declarations.

Four I/O stage counters are ordered SSD read/write/H2D/D2H, each {ops,bytes}.
The action adds exactly the frozen one-stage physical amount and operations;
baseline adds none. Existing-I/O basis baseline keeps the exact existing vector.
No-I/O-plus-joint basis baseline has zero existing I/O, and the action includes
the frozen existing vector. No extra existing-cost term is added.
Legal added batches are 1/2/4/8 quanta. All observed extra physical bytes must be
accounted as completed at run drain. Windows are positive, nonoverlapping,
inside their runs; warmup occurs first and is excluded from the estimators.

Estimator (integer nanoseconds, empirical margin, NOT a confidence interval):
1. Compute upward-rounded mean measured step duration for each independent run.
2. baseline = upward-rounded mean calibration baseline run means.
3. incremental_or_joint = max(0, upward-rounded mean calibration paired
   action-minus-baseline run means).
4. uncertainty = max(0, max measured action step duration across calibration and
   held-out validation minus baseline minus incremental_or_joint).
This retains single-window held-out tails; validation is not used to tune the
two point estimates. It is used only as an explicitly conservative residual
envelope. No SLO or generalization guarantee follows from this finite envelope.
Candidate numbers/counts and pair_analysis must exactly equal the recomputation.

Analysis has exact schema_version, scope=p4_paired_analysis_candidate, origin,
context, cell_id, plan_ref, evidence_refs (the four raw refs), formula,
baseline_ns, incremental_or_joint_ns, uncertainty_ns, calibration_pairs,
validation_pairs, paired_runs_reported, windows_reported.
All raw bytes and every reference are rechecked after arithmetic. Receipts are
append-only. A forged native-looking set can pass internal semantics, but all
results still have gpu_verified=false and production_qualified=false.

Remaining real qualification: an independently trusted source-locked native GPU
measurement runner/step-event capture; measured hardware/model/load identity;
actual native correctness, release/continuation/drain, interference quota and
end-to-end effect qualification. None was executed by these CPU tests.
