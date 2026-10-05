"""CPU-only P0 regression report; never grants model/GPU execution authority.

Writes a new evidence directory, leaves the supplied handoff/reference logs
unchanged, and binds the actual uncommitted code files rather than inventing a
new Git SHA. Run after code edits, not during concurrent editing.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def record(self, test, state, detail=None):
        self.records.append(dict(test_id=test.id(),
            taskbook_ids=sorted(set(re.findall(r'T\d{2}', test.id()))),
            input_case=test.id(), expected='test assertions hold; CPU evidence only',
            observed=state, passed=True if state == 'PASS' else False if state == 'FAIL' else None,
            detail=detail, evidence_scope='local CPU regression, fixtures and mocks; not native-model qualification'))

    def addSuccess(self, test):
        super().addSuccess(test); self.record(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err); self.record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err); self.record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.record(test, 'SKIP', reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.record(subtest, 'FAIL', self._exc_info_to_string(err, test))


# Exact unittest-discovery IDs, not counts or manually asserted passed=true.
# Keep the historical aggregate flags below conservative. These scoped claims
# describe existing local bridges, not the production factory or GPU status.
CPU_BRIDGE_TEST_IDS = {
    'target_capture_and_provenance_bridge': (
        'test_native_capture_hook_v2.NativeCaptureHookTests.test_actual_projection_once_pre_rope_owned_slices_and_no_cuda',
        'test_native_source_capture_v2.NativeSourceCaptureTests.test_same_layer_capture_no_extra_forward_and_pre_rope_slice',
        'test_native_source_capture_v2.NativeSourceCaptureTests.test_mixed_full_target_survives_upstream_selective_commit',
        'test_native_source_capture_v2.NativeSourceCaptureTests.test_G2_and_unknown_parent_do_not_publish_candidate',
    ),
    'issued_comparison_publication_bridge': (
        'test_native_comparison_publication_v2.NativeComparisonPublicationTests.test_real_miss_before_finish_publishes_after_finish_only_for_future_requests',
        'test_native_comparison_publication_v2.NativeComparisonPublicationTests.test_full_mismatch_computes_real_K_scores_then_publishes',
        'test_native_comparison_publication_v2.NativeComparisonPublicationTests.test_bare_scope_dict_and_forged_receipts_reject_without_store_commit',
        'test_native_comparison_publication_v2.NativeComparisonPublicationTests.test_partial_comparison_cannot_grow',
    ),
    'isolated_mixed_publication_and_storage': (
        'test_native_publication_v2.NativePublicationTests.test_mixed_parent_rank_preserved_and_G2_not_laundered',
        'test_source_store_v2.TargetSourceStoreTests.test_exact_mixed_share_capacity_at_k1_k2_k4',
        'test_source_store_v2.TargetSourceStoreTests.test_failed_staging_and_catalog_preserve_preselected_victim',
        'test_source_store_v2.TargetSourceStoreTests.test_child_remains_readable_after_parent_eviction_no_parent_lease',
    ),
    'bounded_p0_request_lifecycle': (
        'test_native_p0_request_v2.NativeP0RequestTests.test_cold_content_miss_needs_no_current_projection_and_stays_dense',
        'test_native_p0_request_v2.NativeP0RequestTests.test_T18_T19_actual_capture_publishes_post_answer_for_next_snapshot_only',
        'test_native_p0_request_v2.NativeP0RequestTests.test_prepared_missing_costs_dense_answer_and_cleanup_not_fake_commit',
        'test_native_p0_request_v2.NativeP0RequestTests.test_post_answer_uncertain_publication_preserves_output_and_stops_next_work',
    ),
}


def scoped_cpu_bridge_evidence(records):
    """Explain implemented CPU bridges without upgrading any native gate.

    Test IDs resolve into this report's test_results.json and digest-bound raw
    test log. Missing, skipped, failed or duplicate results cannot count as a
    passing bridge. This function does not authenticate external report files.
    """
    by_id = {}
    for record in records:
        by_id.setdefault(record.get('test_id'), []).append(record.get('observed'))
    bridges = {}
    for name, test_ids in CPU_BRIDGE_TEST_IDS.items():
        unresolved = {tid: by_id.get(tid, []) for tid in test_ids
                      if by_id.get(tid) != ['PASS']}
        bridges[name] = dict(local_code_implemented=True,
            cpu_evidence_passed=not unresolved, required_test_ids=list(test_ids),
            missing_or_nonpassing_test_results=unresolved,
            hardware_qualification='NOT_RUN')
    return dict(kind='scoped_p0_cpu_bridge_evidence_v1',
        scope='CPU fixtures/mocks and real local storage only; no native-model or GPU qualification',
        test_record_file='test_results.json', bridges=bridges,
        legacy_aggregate_flag_semantics={
            'source_proof_verifier_and_native_hooks_integrated':
                'Conservative end-to-end native qualification aggregate; see target_capture_and_provenance_bridge for local implementation.',
            'publication_scope_runtime_receipts_integrated':
                'Conservative production-runtime aggregate; issued receipts are locally connected in the isolated P0 driver.',
            'mixed_atomic_publication_integrated':
                'Conservative native mixed-publication qualification aggregate; isolated local transaction implementation exists.',
            'production_v2_pool_dispatch_integrated':
                'False: the general production factory still uses the historical schema10 pool; this is distinct from the bounded P0 driver.',
        },
        remaining_implementation_scope=[
            'general production factory v2 pool dispatch',
            'verified exact Prefix proof adapter for v2 capture with Prefix hits',
            'complete native P0-E/P0-M matrix qualification and P1 evidence consumer',
        ],
        native_runtime_qualified=False, gpu_execution_allowed=False,
        P1_E_execution_allowed=False, P1_M_execution_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory; never overwrite failed evidence')
    output.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONPATH'] = os.pathsep.join((str(root / 'src'), str(root)))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'src'))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')

    def command(name, argv):
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=str(root), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        path = output / (name + '.log')
        with path.open('x', encoding='utf-8') as stream:
            stream.write(proc.stdout)
        return dict(command=argv, returncode=proc.returncode, duration_seconds=time.monotonic()-start,
                    log=path.name, log_sha256=sha(path), passed=proc.returncode == 0)

    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=str(root), text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=str(root), text=True).strip()
    listed = subprocess.check_output(['git','ls-files','-z'], cwd=str(root)).decode('utf-8').split('\0')
    additions = subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z',
                                        '--','src','scripts','tests','docs'], cwd=str(root)).decode('utf-8').split('\0')
    files = {name:sha(root/name) for name in sorted(set(listed+additions))
             if name and (root/name).is_file()}
    files_digest = hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                                           separators=(',',':')).encode()).hexdigest()
    write(output/'worktree_files.json', dict(base_commit=head, branch=branch,
        digest=files_digest, scope='tracked files plus untracked src/scripts/tests/docs; excludes user bundles and supplied packet',
        files=files, uncommitted_code=True))
    packet = root/'ProbeKV_Codex_First_Handoff'
    checks = []
    for line in (packet/'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ',1)
        actual = sha(packet/name)
        checks.append(dict(file=name, expected=expected, actual=actual, passed=actual==expected))
    write(output/'package_checksums.json', checks)
    commands = [command('compileall',[sys.executable,'-m','compileall','-q','src','scripts','tests'])]
    log = output/'cpu_tests.log'
    with log.open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        suite = unittest.defaultTestLoader.discover(str(root/'tests'))
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    for row in result.records:
        row.update(runtime_commit=head, runtime_worktree_digest=files_digest,
                   evidence_path=str(log), evidence_hash=sha(log))
    write(output/'test_results.json', result.records)
    commands += [command('contract_validator',[sys.executable,'scripts/validate_contract.py']),
                 command('diff_check',['git','diff','--check'])]
    reference = command('packet_reference_only',[sys.executable,'-m','unittest','discover',
                       '-s',str(packet/'reference'),'-p','test_provenance_reference.py','-v'])
    reference['evidence_scope'] = 'packet reference rules only; not repository/model qualification'
    write(output/'commands.json', dict(repository_checks=commands, packet_reference=reference))
    all_passed = result.wasSuccessful() and all(c['passed'] for c in commands) and all(c['passed'] for c in checks)
    write(output/'correctness_report.json', dict(status='CPU_CHECKS_PASS_NATIVE_PENDING' if all_passed else 'FAILED',
        base_commit=head, worktree_digest=files_digest, python=platform.python_version(),
        platform=platform.platform(), cuda_visible_devices='', gpu_actions_executed=0,
        tests_run=result.testsRun, passed=sum(r['observed']=='PASS' for r in result.records),
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        new_p0_tests=sum(r['test_id'].startswith(('test_decoupled_v2',
            'test_native_capture_hook_v2', 'test_native_source_capture_v2',
            'test_native_v2_prefix_publication', 'test_source_manifest_v2',
            'test_source_store_v2', 'test_native_publication_v2',
            'test_source_comparison_v2', 'test_native_comparison_publication_v2',
            'test_native_consumption_v2', 'test_cuda_comparison_v2',
            'test_native_p0_operation_v2', 'test_native_p0_request_v2',
            'test_p0_evidence_v2', 'test_p0_batch_v2', 'test_p0_decode_evidence_v2',
            'test_p0_exact_control_v2', 'test_p0_exact_pair_v2', 'test_p0_exact_batch_v2',
            'test_p0_diagnostic_context_v2', 'test_p0_mixed_reference_v2',
            'test_p0_mixed_control_v2', 'test_p0_mixed_batch_v2', 'test_p0_target_kv_evidence_v2',
            'test_p0_mixed_sparse_v2', 'test_p0_mixed_pair_v2',
            'test_p0_stage_readiness_v2', 'test_p0_stage_readiness_cli_v2',
            'test_p0_partition_overlap_v2', 'test_p0_cpu_report_v2',
            'test_p0_launch_recipe_v2', 'test_p0_launch_cli_v2',
            'test_p0_cpu_handoff_v2', 'test_p0_cpu_handoff_cli_v2')) for r in result.records),
        native_correctness='NOT_RUN', p0_e_complete=False, p0_m_complete=False,
        P1_E_execution_allowed=False, P1_M_execution_allowed=False,
        scoped_cpu_bridge_evidence=scoped_cpu_bridge_evidence(result.records),
        source_proof_verifier_and_native_hooks_integrated=False,
        native_target_hook_callsite_integrated=True,
        shared_request_manifest_registry_integrated=True,
        registry_durable=True,
        isolated_target_atomic_store_implemented=True,
        exact_mixed_capacity_shared_in_isolated_store=True,
        completed_capture_publication_api_integrated=True,
        issued_k_comparison_receipts_cpu_integrated=True,
        comparison_bound_publication_cpu_integrated=True,
        native_v2_consumption_api_integrated=True,
        native_v2_consumption_qualified=False,
        comparison_gpu_workspace_qualified=False,
        comparison_cuda_operation_implemented=True,
        comparison_cuda_operation_hardware_tested=False,
        bounded_native_comparison_preparation_operation_implemented=True,
        bounded_native_p0_request_driver_implemented=True,
        native_p0_request_driver_hardware_tested=False,
        bounded_p0_batch_and_raw_evidence_entry_implemented=True,
        native_decode_input_trace_integrated=True,
        raw_logit_conditioning_alignment_checker_implemented=True,
        bounded_exact_capture_control_implemented=True,
        preregistered_exact_capture_pair_checker_implemented=True,
        exact_capture_control_hardware_tested=False,
        independent_mixed_reference_cpu_integrated=True,
        diagnostic_mixed_prefix_and_source_publication_blocked=True,
        raw_target_bf16_evidence_reader_writer_integrated=True,
        independent_mixed_reference_hardware_tested=False,
        mixed_reference_sparse_pair_verifier_integrated=True,
        independent_cacheblend_sparse_control_cpu_integrated=True,
        explicit_target_r1_sparse_control_cpu_integrated=True,
        mixed_reference_and_target_r1_hardware_tested=False,
        staged_controlled_P0_vs_natural_P1_readiness_implemented=True,
        controlled_p0_bounded_recipe_builder_implemented=True,
        controlled_p0_empty_pool_bootstrap_cli_implemented=True,
        controlled_p0_recipe_builder_grants_gpu_permission=False,
        cpu_validated_portable_snapshot_builder_implemented=True,
        portable_snapshot_is_gpu_qualification=False,
        stage_readiness_grants_gpu_authority=False,
        metadata_partition_overlap_auditor_implemented=True,
        metadata_overlap_establishes_data_qualification=False,
        p0_numerical_pair_recipe_qualification_integrated=False,
        full_native_p0_runner_integrated=False,
        production_v2_pool_dispatch_integrated=False,
        publication_scope_runtime_receipts_integrated=False,
        parent_binding_requires_verified_v2_publication_metadata=True,
        exact_prefix_proof_adapter_integrated=False,
        mixed_atomic_publication_integrated=False, gpu_execution_allowed=False,
        taskbook_matrix='CPU partial coverage only; see P0 report for unexecuted T01-T40 native obligations',
        paper_evidence=False, locked_test_accessed=False))
    print(json.dumps(dict(cpu_checks_passed=all_passed, tests_run=result.testsRun,
                         skipped=len(result.skipped), status='BLOCKED', output=str(output))))
    return 0 if all_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
