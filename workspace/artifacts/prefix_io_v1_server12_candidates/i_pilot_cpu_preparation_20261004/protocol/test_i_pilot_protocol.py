"""CPU rejection fixtures only; these are never measurements or GPU evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i_pilot_protocol as p


def frontend_fixture():
    # Fictional event clock only checks arithmetic/identity rejection.
    start = 1000
    steps, rows = [], []
    for index in range(128):
        at = start + (index + 1) * 100
        steps.append(dict(before_ns=at - 50, after_ns=at, cumulative_output_counts=[index + 1]))
        rows.append(dict(token_ordinal=index, at_ns=at))
    return dict(origin='cpu_fixture', request_started_ns=start,
                request_completed_ns=14000, steps=steps, token_times=rows)


def freeze_fixture():
    return dict(schema='i_pilot_development_freeze_v1', deadline_declared_ns=1,
                first_development_observation_ns=2, frozen_before_on_ns=3,
                first_on_observation_ns=4, declared_full_control_window_deadline_ns=100,
                development_non_GPU_reserve_ns=20, internal_step_budget_ns=80,
                deadline_origin='independent_predeclared_development_contract',
                source_partitions=['development'], used_old_heldout_or_evaluation=False,
                deadline_widened_after_results=False, service_TTFT_SLO_ns=None,
                service_ITL_SLO_ns=None, new_calibration_cost_upper_ns=90,
                origin='cpu_fixture')


class ProspectiveRejectionTests(unittest.TestCase):
    def test_protocol_valid_without_effect_promotion(self):
        doc = p.prospective_protocol()
        result = p.evidence_status(doc)
        self.assertEqual(result['effect_status'], 'BLOCKED_FOR_EFFECT')
        self.assertFalse(result['performance_benefit_proved'])
        self.assertNotEqual(result['status'], 'CPU_READY_FOR_GPU_CALIBRATION')

    def test_old_upper_or_budget_changes_rejected(self):
        for key in ('cost_upper_ns', 'step_budget_ns'):
            doc = p.prospective_protocol()
            doc['frozen_old_experiment'][key] += 1
            with self.assertRaises(ValueError):
                p.validate_protocol(doc)

    def test_calibration_heldout_reuse_rejected(self):
        doc = p.prospective_protocol()
        doc['calibration']['fit_family_ids'].append('cal2')
        with self.assertRaises(ValueError):
            p.validate_protocol(doc)

    def test_partition_family_leak_rejected(self):
        doc = p.prospective_protocol()
        doc['families'][-1]['prompt_token_ids'] = doc['families'][0]['prompt_token_ids']
        with self.assertRaises(ValueError):
            p.validate_protocol(doc)

    def test_selected_offset_or_mode_change_rejected(self):
        for change in ('offset', 'on_before_shadow'):
            doc = p.prospective_protocol()
            if change == 'offset':
                doc['diagnostic_geometry']['selected_offset'] = 17
            else:
                doc['jobs'][3], doc['jobs'][4] = doc['jobs'][4], doc['jobs'][3]
            with self.assertRaises(ValueError):
                p.validate_protocol(doc)

    def test_boolean_budget_or_unit_not_accepted_as_integer(self):
        for table, key in (('diagnostic_geometry', 'legal_storage_units'), ('budget', 'retry_slots')):
            doc = p.prospective_protocol()
            doc[table][key] = bool(doc[table][key])
            with self.assertRaises(ValueError):
                p.validate_protocol(doc)

    def test_total_budget_is_not_a_second_ledger(self):
        doc = p.prospective_protocol()
        self.assertEqual(sum(row['reserved_seconds'] for row in doc['jobs']), 3780)
        self.assertGreater(doc['budget']['remaining_snapshot_seconds'], 3780)
        doc['budget']['original_ledger'] += '.new'
        with self.assertRaises(ValueError):
            p.validate_protocol(doc)

    def test_guard_command_is_original_and_calibration_only_bound_path(self):
        command = p.guard_command(p.job_rows()[0],
                                  permission_relative=p.REMOTE + '/calibration_v2/EFFECTIVE_GPU_PERMISSION.json',
                                  runner_relative=p.REMOTE + '/calibration_v2/run_native_cost_experiment.py')
        self.assertIn('experiments/prefix_io_v1/scripts/run_gpu_stage.py', command)
        self.assertEqual(command[command.index('--seconds') + 1], '1200')
        self.assertEqual(command[-1], p.REMOTE + '/calibration_v2/NATIVE_COST_CONFIG.json')

    def test_foreign_guard_paths_rejected(self):
        for path in ('../runner.py', '/tmp/runner.py', p.REMOTE + '/../runner.py', p.REMOTE + '/a\\b'):
            with self.assertRaises(ValueError):
                p.guard_command(p.job_rows()[0], permission_relative=p.REMOTE + '/permission.json',
                                runner_relative=path)

    def test_missing_finite_record_not_assumed_complete(self):
        with self.assertRaises(ValueError):
            p.evidence_status(p.prospective_protocol(), runtime_ready=1)

    def test_boolean_declarations_cannot_replace_actual_byte_join(self):
        result = p.evidence_status(p.prospective_protocol(), runtime_ready=True,
                                   calibrated=True, development_frozen=True,
                                   ordinary_candidate=True, on_lifecycle=True, evaluation_complete=True)
        self.assertNotEqual(result['status'], 'CPU_READY_FOR_GPU_CALIBRATION')
        self.assertFalse(result['performance_benefit_proved'])

    def test_duplicate_and_nonfinite_JSON_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                p.parse(raw)

    def test_native_single_token_time_arithmetic(self):
        result = p.token_metrics(frontend_fixture())
        self.assertEqual(result['actual_ITL_observations'], 127)
        self.assertEqual(result['request_ITL_P95_ns'], 100)
        self.assertEqual(result['TTFT_ns'], 100)
        self.assertFalse(result['ITL_independence_assumed'])

    def test_batched_output_cannot_fake_per_token_ITL(self):
        frontend = frontend_fixture()
        frontend['steps'][10]['cumulative_output_counts'] = [12]
        frontend['token_times'][11]['at_ns'] = frontend['token_times'][10]['at_ns']
        with self.assertRaisesRegex(ValueError, 'chunk output'):
            p.token_metrics(frontend)

    def test_clock_mismatch_missing_or_reordered_tokens_rejected(self):
        for kind in ('time', 'ordinal', 'missing', 'bool'):
            frontend = frontend_fixture()
            if kind == 'time':
                frontend['token_times'][10]['at_ns'] += 1
            elif kind == 'ordinal':
                frontend['token_times'][10]['token_ordinal'] = 11
            elif kind == 'missing':
                frontend['token_times'].pop()
            else:
                frontend['token_times'][0]['token_ordinal'] = False
            with self.assertRaises(ValueError):
                p.token_metrics(frontend)

    def test_post_result_deadline_declaration_rejected(self):
        doc = freeze_fixture()
        doc['deadline_declared_ns'] = 2
        with self.assertRaises(ValueError):
            p.validate_development_freeze(doc)

    def test_missing_independent_development_deadline_rejected(self):
        doc = freeze_fixture()
        doc['declared_full_control_window_deadline_ns'] = None
        with self.assertRaises(ValueError):
            p.validate_development_freeze(doc)

    def test_A_max_or_fabricated_SLO_rejected(self):
        for key, value in (('deadline_origin', 'maximum_of_two_A_steps'), ('service_ITL_SLO_ns', 100)):
            doc = freeze_fixture()
            doc[key] = value
            with self.assertRaises(ValueError):
                p.validate_development_freeze(doc)

    def test_reserve_cannot_be_spent_twice_or_omitted(self):
        doc = freeze_fixture()
        doc['internal_step_budget_ns'] = 100
        with self.assertRaises(ValueError):
            p.validate_development_freeze(doc)

    def test_no_admission_stops_instead_of_widening(self):
        result = p.validate_development_freeze(freeze_fixture())
        self.assertEqual(result['status'], 'STOP_NO_CURRENT_INTERFERENCE_ADMISSION_OPPORTUNITY')
        self.assertFalse(result['ordinary_cost_admission_fits'])
        self.assertFalse(result['frozen_old_experiment_changed'])

    def test_heldout_and_evaluation_are_never_development(self):
        for key, value in (('used_old_heldout_or_evaluation', True), ('source_partitions', ['evaluation']),
                           ('deadline_widened_after_results', True)):
            doc = freeze_fixture()
            doc[key] = value
            with self.assertRaises(ValueError):
                p.validate_development_freeze(doc)

    def test_actual_CPU_join_rejects_false_origin_partial_proof_and_unsafe_ledger(self):
        # Pure mocked fixtures exercise only rejection paths. They never create
        # an actual-server report, GPU record or positive CPU-ready artifact.
        source_paths = [p.REMOTE + '/calibration_v2/' + name for name in
                        ('run_native_cost_experiment.py', 'control_native_cost_job.py', 'gpu_entry_binding.py')]
        source_paths += ['experiments/prefix_io_v1/scripts/run_gpu_stage.py',
                         'artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json']
        references = {path: dict(path=path, bytes=100, sha256='a' * 64) for path in source_paths}
        references[source_paths[-2]]['sha256'] = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'
        references[source_paths[-1]]['sha256'] = p.MODEL
        evidence_refs = {key: dict(path=key + '.json', bytes=100, sha256='b' * 64)
                         for key in ('lock', 'proof', 'tests', 'preflight')}
        ledger_ref = dict(path='experiments/prefix_io_v1/gpu-budget-ledger.json', bytes=100, sha256='c' * 64)
        out_ref = dict(path='stdout.log', bytes=100, sha256='d' * 64)
        err_ref = dict(path='stderr.log', bytes=0, sha256='e' * 64)
        references.update({out_ref['path']: out_ref, err_ref['path']: err_ref})
        rows = [references[path] for path in source_paths]
        proof = dict(failed=[], files_verified=len(rows),
                     source_rows_sha256=p.digest(sorted(rows, key=lambda row: row['path'])))
        tests = dict(exit_code=0, tests_passed=1, tests_failed=0, tests_skipped=0,
                     origin='actual_server_CPU', GPU_operations=0, stdout_ref=out_ref, stderr_ref=err_ref)
        preflight = dict(schema='i_pilot_calibration_CPU_preflight_v1', status='CPU_CALIBRATION_PREFLIGHT_PASS',
                         origin='actual_server_CPU', GPU_operations=0, full_source_verified=True,
                         entry_CPU_tests_passed=True, requires_live_original_guard_recheck=True,
                         source_lock_ref=evidence_refs['lock'], source_proof_ref=evidence_refs['proof'],
                         cpu_test_result_ref=evidence_refs['tests'], model_manifest_sha256=p.MODEL,
                         original_ledger_ref=ledger_ref, device_launch_ready=False,
                         guard_command=p.guard_command(p.job_rows()[0],
                             permission_relative=p.REMOTE + '/calibration_v2/EFFECTIVE_GPU_PERMISSION.json',
                             runner_relative=p.REMOTE + '/calibration_v2/run_native_cost_experiment.py'))
        base = dict(lock=dict(files=rows), proof=proof, tests=tests, preflight=preflight,
                    ledger=dict(gpu_wall_seconds=24678.89523890568, active_reservation=None))
        runner_bytes = b"PROMPT_FIRST=(40100,41100,42100)\nSEEDS=(4029,4030,4031)\nLABEL='server12-i-pilot-cal01'\nORDER=(('A','B'),('B','A'),('A','B'))\n"
        class FixturePath:
            def read_bytes(self):
                return runner_bytes
        for kind in ('fixture_origin', 'partial_proof', 'wrong_rows', 'active_ledger',
                     'budget_exhausted', 'wrong_guard', 'unobserved_device_as_true'):
            documents = copy.deepcopy(base)
            if kind == 'fixture_origin':
                documents['tests']['origin'] = 'cpu_fixture'
            elif kind == 'partial_proof':
                documents['proof']['files_verified'] -= 1
            elif kind == 'wrong_rows':
                documents['proof']['source_rows_sha256'] = 'f' * 64
            elif kind == 'active_ledger':
                documents['ledger']['active_reservation'] = {'label': 'foreign'}
            elif kind == 'budget_exhausted':
                documents['ledger']['gpu_wall_seconds'] = 28000
            elif kind == 'wrong_guard':
                documents['preflight']['guard_command'][2] = 'second_guard.py'
            else:
                documents['preflight']['device_launch_ready'] = 1
            def read_doc(root, row):
                if row == ledger_ref:
                    return documents['ledger']
                return documents[row['path'].removesuffix('.json')]
            with mock.patch.object(p, 'check_ref', side_effect=read_doc), \
                 mock.patch.object(p, 'file_ref', side_effect=lambda root, relative: references[relative]), \
                 mock.patch.object(p, 'safe', return_value=FixturePath()):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    p.assess_calibration_preparation(Path('.'), p.prospective_protocol(),
                        source_lock_ref=evidence_refs['lock'], source_proof_ref=evidence_refs['proof'],
                        cpu_test_result_ref=evidence_refs['tests'], preflight_ref=evidence_refs['preflight'])


if __name__ == '__main__':
    unittest.main()
