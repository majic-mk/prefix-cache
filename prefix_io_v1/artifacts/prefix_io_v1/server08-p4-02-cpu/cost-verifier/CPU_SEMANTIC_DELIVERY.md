# P4 CPU paired-cost semantic follow-through

Status: CPU_SEMANTIC_VERIFIER_AND_PREPARED_LOADER_COMPLETE_GPU_QUALIFICATION_BLOCKED.

Actual new source: p4_paired_measurement_verifier.py (bounded strict original
measurement recomputation), p4_verified_cost_loader.py (prepared exact CPU loader).
Actual new tests: tests/prefix_io_v1_p4_measurements/test_semantic_verifier.py and
run_cpu_guard.py. No old CostTable/candidate parser/policy/options/bridge/native
file was edited by this task. Root may independently connect diagnostic options.

The raw-byte refs, paired source/context/workload identity, warmup exclusion,
AB/BA balance, exact physical action geometry, completion/drain counters and
held-out trace/prefix-family isolation are now checked semantically rather than
only hashed. Candidate and pair-analysis times/counts must equal recomputation.
Plans and bound candidates are reparsed from their actual pinned bytes; manually
constructed divergent typed objects do not bypass those checks. A final full
byte recheck catches changes during arithmetic.

Only paired_residual_margin is implemented. Point estimates use calibration
independent run means, rounded upward. Validation never changes the point
estimates. The margin is the maximum positive measured action-window residual
over calibration and held-out validation. It preserves a long individual window
even if its run mean is unchanged. This is a finite empirical envelope, NOT a
confidence interval, guaranteed SLO, causal effect estimate or deployment promise.

The cost model retains the old exact9 active_decode signature. It does not claim
that the root scheduled-work-v1 producer's planned tasks equal active GPU decode.
Actual native active-decode/context/step evidence remains an outer GPU gate.

Real server test executions:
- run-01: 8 failed, 42 apparent passes. Test fixture accidentally assigned a
  self.load dictionary over its load method. None of those 42 negative results
  is accepted as verification evidence. Failure XML, CPU guard and failed test
  source are retained under run-01. Fixed the fixture name and narrowed all
  rejection assertions to TableContractError.
- run-02: 50 passed, 0 failed, 0 skipped.
- run-03: 52 passed after true AB/BA balance, pinned plan and held-out window-tail
  tests were added.
- run-04 final: 53 passed, 0 failed, 0 skipped, 1.01 seconds; guarded process exit0.
  It additionally checks actual bound-candidate bytes and readonly diagnostic
  source_ref/cell_count. This is the sole final test count, not a sum of repeats.

Guard denies torch/cuda/cupy/pycuda/vllm/py_kvcache imports and makes visible GPU
devices empty. No GPU imports, initialization, status probes, runs, reservations,
model downloads, package installs or driver/system operations were performed.
The test guard reports cuda_initialized=false and gpu_workloads_run=0.

Three real CPU-executed synthetic evidence packages were generated:
cpu_fixture, forged_native_origin, no_io_plus_joint. All internally consistent
fixtures pass semantic recomputation and explicitly remain unqualified; all
production lookups return None. Their nanosecond numbers are synthetic arithmetic
fixtures, not GPU measurements. The fake verifier.py deliberately raises if
executed; the contract hashes it and never executes referenced code.

API for root diagnostic startup:
load_semantically_verified_table(root, candidate_path, *, expected_context,
qualification_ref, expected_verifier_ref, plan_path, expected_plan_ref)
returns PreparedCostTable with status, source_ref (candidate EvidenceRef),
cell_count, gpu_verified=false, production_qualified=false, verification, and
cpu_mock_table. Native startup must not pass cpu_mock_table into production.
Verification cells expose origin, calibration_pairs, validation_pairs,
paired_runs, measured_windows, warmup_windows, max_positive_residual_ns and the
recomputed exact CostCell. write_semantic_verification_manifest writes exclusively
and rechecks actual evidence bytes. SCHEMA.md freezes all fields and the formula.

Evidence locations in this directory:
SCHEMA.md; source-lock.json; executed-commands.json; run-04/tests.xml;
run-04/cpu-guard.json; run-04/process-result.json; fixtures-index.json;
fixtures/*/semantic-verification.json; fixture-generation-result.json.
Earlier failures and successful repeat runs remain preserved.

Next allowed stage: finish root CPU integration and independently review these
contracts. When a GPU is available and authorized, obtain source-locked authentic
native GPU measurements and control/liveness qualification before any production
cost gate can open. Full P4 and positive effect remain unverified. No P5/P6/P7
execution is implied by this CPU follow-through.
