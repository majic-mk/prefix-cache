"""Meaningful synthetic CPU counterexamples, not native/GPU qualification."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DELIVERY = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
OVERLAY = DELIVERY + '/common_candidate/source'
REACTOR = OVERLAY + '/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
COLLECTOR = DELIVERY + '/common_candidate/native_full_step_collector.py'
CONTROL = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
CANDIDATE = Path(os.environ.get('C5_FROZEN_CANDIDATE', str(HERE.parent / 'notification_v5_cpu_delivery/server_replay/candidate')))
NATIVE = Path(os.environ.get('C5_NATIVE_PARENT', str(HERE.parent / 'native_cost_v6')))
ORIGINAL = Path(os.environ.get('C5_ORIGINAL_ESTIMATOR', str(CANDIDATE / CONTROL / 'p4_paired_measurement_verifier.py')))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load('_c5_factory_serializer', 'prepare_and_verify_native_cost.py')
F = load('_c5_factory_fixture', 'synthetic_raw_fixture.py')
T = load('_c5_factory_contract', 'native_preparation_contract.py')
R = load('_c5_factory_runner', 'run_native_cost_experiment.py')
J = load('_c5_factory_job', 'control_native_cost_job.py')


def post_shutdown(journal):
    """Construct synthetic original-shaped counters from this synthetic journal."""
    events = [row for row in journal['events'] if row['kind'] == 'accepted']
    stages = {}
    for stage in P.C.STAGES:
        rows = [row for row in events if row['stage'] == stage]
        size = sum(row['physical_bytes'] for row in rows)
        stages[stage] = dict(accepted_ops=len(rows), accepted_bytes=size,
            completed_ops=len(rows), completed_requested_bytes=size, transferred_bytes=size,
            inflight_ops=0, inflight_bytes=0, failed_ops=0)
    count = len(events)
    native = dict(active_parents=0, ring_ops=0, pending_copies=0, copy_ready=0,
        ready_load_fds=0, ready_preload_fds=0)
    aio = dict(closed=True, drained=True, fatal=None, outstanding=0, pending=0, ready=0,
        unreaped=0, accepted=count, completed=count, reaped=count)
    parents = dict(count_valid=True, native_drain_unknown=False, fatal_reason=None,
        source_gpu_reuse_inferred=False, staging_release_inferred=False, accepted_parents=0,
        accepted_count_lower_bound=0, waiting_submitters=0, failure_stream_syncs=0)
    snapshot = dict(native_shutdown_read=True, owner_capture=False, physical_drain_inferred=False,
        gpu_release_credit=False, native=native, aio=aio, parent_admission=parents, admission=parents,
        stage_accounting=dict(valid=True, bound=True, error=None, resource_release_inferred=False,
            outstanding_records=0, stages=stages))
    return dict(snapshot=snapshot, actual_native=native, actual_aio=aio,
        actual_handler_active=0, observation_failures=0, actual_stage_counters=stages)


class RawReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        plan, windows, journal, _ = F.fixture()
        self.plan = plan
        plan.update(cpu_preparation_only=True, evidence_origin='synthetic_cpu_contract', gpu_uuid=None,
            job_id=T.LABEL, journal_run_id=T.LABEL, common_overlay_relative=OVERLAY,
            common_owner_parameters=dict(max_accepted_parents=8, bridge_is_none=True),
            project_root=self.root.as_posix(), transfer_quantum_bytes=917504,
            model_plan_ref=dict(path='model-plan.json', bytes=1, sha256='e' * 64),
            external_warmup_output_tokens=128, external_flush_output_tokens=1)
        for relative, source in ((REACTOR, CANDIDATE / 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'),
                                  (COLLECTOR, CANDIDATE / 'native_full_step_collector.py')):
            target = self.root / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        native = P.ref(self.root, REACTOR)
        plan.update(native_source_ref=native, native_source_sha256=native['sha256'],
            collector_source_ref=P.ref(self.root, COLLECTOR))
        for event in journal['events']:
            event['physical_bytes'] *= 114688
            if event['result'] is not None: event['result'] *= 114688
        for owner in journal['frames']:
            owner.update(run_id=T.LABEL, source_sha256=native['sha256'])
            for kind in ('accepted', 'completed', 'inflight'):
                for amount in owner[kind]: amount['nbytes'] *= 114688
        post = post_shutdown(journal)
        tail = dict(worker_alive=False, aio_worker_alive=False, handler_shutdown=True, reactor_closed=True)
        self.report = dict(status='PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION',
            origin='native_gpu_recording', label=T.LABEL, gpu_uuid=None,
            original_model_subprocesses_started=6, actual_guarded_gpu_job_count=1,
            input_template_unchanged=True, original_engine_shutdown_returned=True, load_planner='off',
            guard=dict(session_id=123), windows=[], children=[])
        self.children = []
        for index, window in enumerate(windows):
            external_id = 'synthetic-external-' + str(index)
            capture = deepcopy(window['capture']); capture['run_id'] = external_id
            capture['event_source']['path'] = self.root.as_posix() + '/torch/cuda/streams.py'
            payload = deepcopy(window['independent_payload'])
            if payload is not None: payload['physical_bytes'] = 917504
            raw = dict(pair_index=index // 2, condition='A' if window['arm'] == 'baseline' else 'B',
                prompt_token_ids=window['prompt_token_ids'], seed=window['seed'], request_id=external_id,
                frontend=dict(output=dict(num_cached_tokens=128, output_token_ids=window['output_token_ids'],
                    request_id=external_id, native_request_id=window['request_id'])),
                capture=capture, independent_payload=payload)
            child = dict(status='PASS_NATIVE_FULL_STEP_RAW_CAPTURE_REQUIRES_VALIDATION', window_index=index,
                fresh_original_process=True, original_engine_shutdown_returned=True, gpu_uuid=None,
                source_lock=plan['source_lock_ref']['path'], model=dict(manifest_sha256='e' * 64, plan_sha256='e' * 64),
                subprocess_pid=1000 + index, private_storage='synthetic-private-' + str(index), subprocess_sid=123,
                windows=[raw], native_journal=deepcopy(journal), native_post_shutdown=deepcopy(post),
                native_tail_assertions=deepcopy(tail), warmups=[dict(output_token_ids=list(range(128)),
                    prompt_token_ids=window['prompt_token_ids'], seed=window['seed'], flush=dict(output_token_ids=[1]))],
                native_source_binding=dict(actual_source_ref=native, original_run_code_verified=True,
                    max_accepted_parents=8, bridge_is_none=True,
                    loaded_native_modules=[dict(module='py_kvcache.reactor', source_ref=native)]),
                collector_native_source_binding=dict(legacy_lookup_key=R.LEGACY_COLLECTOR_REACTOR_KEY,
                    actual_source_ref=native, adapter_native_source_sha256_before=native['sha256'],
                    adapter_native_source_sha256_after=native['sha256'], same_adapter_all_frames=True, full_frame_count=128))
            self.children.append(child)
            self.report['windows'].append(deepcopy(raw) | dict(native_journal=deepcopy(journal), native_post_shutdown=deepcopy(post)))
            self.report['children'].append(dict(window_index=index, exit=0))
        self.version = 0

    def write_raw(self):
        self.version += 1
        prefix = str(self.version)
        plan_ref = P.write(self.root, prefix + '/plan.json', self.plan)
        for index, child in enumerate(self.children):
            self.report['windows'][index]['child_receipt_ref'] = P.write(self.root, prefix + '/child-' + str(index) + '.json', child)
        runtime = prefix + '/runtime.json'; P.write(self.root, runtime, self.report)
        return dict(origin='synthetic_cpu_contract', runtime_relative=runtime, plan_ref=plan_ref, source_verification_refs={})

    def serialize(self):
        envelope = self.write_raw()
        return P.serialize_runtime_record(self.root, runtime_relative=envelope['runtime_relative'],
            plan_ref=envelope['plan_ref'], source_verification_refs={}, synthetic_cpu=True)

    def audit(self):
        return T.audit_synthetic_raw(self.root, envelope=self.write_raw(), original_estimator_path=ORIGINAL)

    def test_six_raw_files_reconstruct_original_full_formula_without_native_grant(self):
        result = self.audit()
        self.assertEqual(result['full_output_tokens'], 768)
        self.assertEqual(len(result['raw_child_refs']), 6)
        self.assertTrue(result['synthetic_holdout_covered'])
        for key in ('native_execution_verified', 'native_cost_qualified', 'gpu_launch_allowed',
                    'full_runtime_cost_qualified', 'on_observation_cost_measured', 'performance_claim'):
            self.assertIs(result[key], False)
        self.assertIsNone(result['effective_cost_upper_ns']); self.assertIsNone(result['valid_native_receipt'])

    def test_early_child_semantics_full_128_frames(self):
        env = self.write_raw()
        result = P.verify_raw_window(self.root, plan_ref=env['plan_ref'],
            child_relative=self.report['windows'][1]['child_receipt_ref']['path'], synthetic_cpu=True)
        self.assertEqual(result['full_frames'], 128)
        self.assertEqual(result['native_io']['accepted_physical_bytes'], 917504)
        self.assertFalse(result['native_cost_qualified'])

    def test_synthetic_flag_required_before_root_access(self):
        for flag in (False, None, 1, 'true'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'explicit synthetic'):
                P.serialize_runtime_record(None, runtime_relative=None, plan_ref=None,
                    source_verification_refs=None, synthetic_cpu=flag)

    def test_early_check_synthetic_flag_required(self):
        with self.assertRaisesRegex(ValueError, 'explicit synthetic'):
            P.verify_raw_window(None, plan_ref=None, child_relative=None)

    def test_no_native_envelope_promotion(self):
        env = self.write_raw(); env['origin'] = 'native_gpu_recording'
        with self.assertRaisesRegex(ValueError, 'explicit synthetic'):
            T.audit_synthetic_raw(self.root, envelope=env, original_estimator_path=ORIGINAL)

    def test_new_plan_marker_required(self):
        self.plan['cpu_preparation_only'] = False
        with self.assertRaisesRegex(ValueError, 'explicitly synthetic'): self.serialize()

    def test_old_job_identity_rejected(self):
        self.plan['job_id'] = 'server11-native-cost-six-window-06'
        self.report['label'] = self.plan['job_id']
        with self.assertRaisesRegex(ValueError, 'old C4/v6'): self.serialize()

    def test_unresolved_uuid_not_replaced_by_old_or_fixture_uuid(self):
        self.plan['gpu_uuid'] = self.report['gpu_uuid'] = 'GPU-CPU-TEST'
        with self.assertRaisesRegex(ValueError, 'no resolved GPU UUID'): self.serialize()

    def test_shared_child_pid_rejected(self):
        self.children[1]['subprocess_pid'] = self.children[0]['subprocess_pid']
        with self.assertRaisesRegex(ValueError, 'fresh original process'): self.serialize()

    def test_shared_private_storage_rejected(self):
        self.children[1]['private_storage'] = self.children[0]['private_storage']
        with self.assertRaisesRegex(ValueError, 'fresh original process'): self.serialize()

    def test_original_child_exit_rejected(self):
        self.report['children'][3]['exit'] = 1
        with self.assertRaisesRegex(ValueError, 'child exit'): self.serialize()

    def test_adapter_drift_rejected(self):
        self.children[2]['collector_native_source_binding']['adapter_native_source_sha256_after'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'all 128 frames'): self.serialize()

    def test_short_adapter_frame_count_rejected(self):
        self.children[2]['collector_native_source_binding']['full_frame_count'] = 127
        with self.assertRaisesRegex(ValueError, 'all 128 frames'): self.serialize()

    def test_enabled_strategy_during_common_calibration_rejected(self):
        self.children[1]['native_source_binding']['bridge_is_none'] = False
        with self.assertRaisesRegex(ValueError, 'actual common owner'): self.serialize()

    def test_wrong_parent_capacity_rejected(self):
        self.children[1]['native_source_binding']['max_accepted_parents'] = 64
        with self.assertRaisesRegex(ValueError, 'actual common owner'): self.serialize()

    def test_c5_source_byte_drift_rejected(self):
        (self.root / REACTOR).write_bytes(b'# changed synthetic source')
        with self.assertRaisesRegex(ValueError, 'source bytes'): self.serialize()

    def test_collector_source_byte_drift_rejected(self):
        (self.root / COLLECTOR).write_bytes(b'# changed synthetic source')
        with self.assertRaisesRegex(ValueError, 'C5 plan collector'): self.serialize()

    def test_parent_child_output_disagreement_rejected(self):
        self.report['windows'][1]['frontend']['output']['output_token_ids'][5] = 999
        with self.assertRaisesRegex(ValueError, 'parent/child raw'): self.serialize()

    def test_short_complete_output_rejected_by_original_formula(self):
        for index in (0,):
            self.children[index]['windows'][0]['capture']['frames'].pop()
            self.children[index]['windows'][0]['capture']['event_witnesses'].pop()
            self.children[index]['windows'][0]['frontend']['output']['output_token_ids'].pop()
            self.report['windows'][index] = deepcopy(self.children[index]['windows'][0]) | dict(
                native_journal=self.children[index]['native_journal'], native_post_shutdown=self.children[index]['native_post_shutdown'])
        with self.assertRaisesRegex(ValueError, '128 original steps'): self.audit()

    def test_short_external_warmup_rejected(self):
        self.children[0]['warmups'][0]['output_token_ids'] = [1]
        with self.assertRaisesRegex(ValueError, 'full-output warmup'): self.serialize()

    def test_real_release_not_inferred_from_pending_copy(self):
        self.children[1]['native_post_shutdown']['actual_native']['pending_copies'] = 1
        self.report['windows'][1]['native_post_shutdown'] = deepcopy(self.children[1]['native_post_shutdown'])
        with self.assertRaisesRegex(ValueError, 'native drain'): self.audit()

    def test_short_cqe_rejected_by_original_owner_formula(self):
        self.children[1]['native_journal']['events'][1]['result'] -= 1
        self.report['windows'][1]['native_journal'] = deepcopy(self.children[1]['native_journal'])
        with self.assertRaisesRegex(ValueError, 'CQE result'): self.audit()

    def test_holdout_never_refits_or_creates_cost(self):
        self.children[-1]['windows'][0]['capture']['event_witnesses'][16]['gpu_elapsed_ns'] = 130
        self.report['windows'][-1]['capture'] = deepcopy(self.children[-1]['windows'][0]['capture'])
        result = self.audit()
        self.assertFalse(result['synthetic_holdout_covered'])
        self.assertFalse(result['holdout_used_to_refit'])
        self.assertIsNone(result['effective_cost_upper_ns'])


class SourceAndBlockingTests(unittest.TestCase):
    def test_unchanged_c5_ancestry_except_new_canonical_receipt(self):
        inheritance = json.loads((HERE / 'SOURCE_INHERITANCE.json').read_bytes())
        self.assertFalse(inheritance['gpu_launch_allowed'])
        for row in inheritance['source_parent_files']:
            parent = CANDIDATE / row['path']
            self.assertEqual(sha256(parent.read_bytes()).hexdigest(), row['sha256'])
            if row['path'].endswith('/p4_single_file_receipt.py'): continue
            current = HERE / 'common_candidate' / row['path']
            self.assertEqual(current.read_bytes(), parent.read_bytes())

    def test_original_numerical_and_causal_ast_unchanged(self):
        result = T.assert_original_algorithms(NATIVE / 'native_conditional_cost.py')
        self.assertFalse(result['numerical_ast_changed'])

    def test_source_constants_remain_finite_uuid_unresolved(self):
        self.assertIs(R.GPU_UUID, None); self.assertIs(J.GPU, None)
        self.assertEqual(R.ORDER, (('A', 'B'), ('B', 'A'), ('A', 'B')))
        self.assertEqual(R.MAX_ACCEPTED_PARENTS, 8)
        self.assertEqual(R.FILE_BYTES, 917504)
        self.assertEqual([R.prompt(index)[0] for index in range(3)], [18100, 19100, 20100])
        draft = T.draft_common_calibration()
        self.assertIsNone(draft['gpu_uuid']); self.assertFalse(draft['gpu_job_created'])

    def test_native_launch_and_configuration_block_before_access(self):
        calls = ((R.load_configuration, (None,), {}), (R.verify_guard, (None,), {}),
            (R.execute_window, (None, None, None, None, None), {}),
            (R.execute_parent, (None, None, None, None), {}),
            (J.prepare, (), {}), (J.scope, (None,), {}), (J.launch, (), {}), (J.status, (), {}),
            (J.check_sources, (None,), {}), (P.create_plan, (None,), dict(source_lock_relative=None,
                config_relative=None, output_relative=None)),
            (P.C.verify_native_cell, (None,), dict(plan_ref=None, measurements_ref=None, guard_ref=None,
                expected_plan_ref=None, expected_guard_ref=None, original_source_path=None)))
        for function, args, kwargs in calls:
            with self.subTest(function=function.__name__), self.assertRaisesRegex(RuntimeError, T.BLOCKED):
                function(*args, **kwargs)

    def test_cli_blocks_invalid_arguments_before_plan_or_config(self):
        for filename in ('native_conditional_cost.py', 'prepare_and_verify_native_cost.py',
                         'run_native_cost_experiment.py', 'control_native_cost_job.py', 'native_preparation_contract.py'):
            result = subprocess.run([sys.executable, '-B', '-I', '-S', str(HERE / filename),
                '--execute', '--config', 'missing-never-read.json'], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, filename)
            self.assertEqual(result.stderr, '', filename)
            document = json.loads(result.stdout)
            self.assertEqual(document['status'], T.BLOCKED)
            self.assertFalse(document['gpu_launch_allowed'])

    def test_canonical_receipt_uses_same_exact_type_in_policy_and_bridge(self):
        src = HERE / 'common_candidate' / CONTROL
        package_paths = [str(HERE / 'common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src')]
        with patch.object(sys, 'path', package_paths + sys.path):
            import prefix_io_control.p4_policy as policy
            import prefix_io_control.p4_bridge as bridge
            import prefix_io_control.p4_single_file_receipt as receipt
            import prefix_io_control.p4_types as types
            self.assertEqual(policy.P4Policy.__module__, 'prefix_io_control.p4_policy')
            for pin, filename in ((receipt._VERIFIER, 'native_conditional_cost.py'),
                                  (receipt._SERIALIZER, 'prepare_and_verify_native_cost.py')):
                raw = (HERE / filename).read_bytes()
                self.assertEqual((pin.bytes, pin.sha256), (len(raw), sha256(raw).hexdigest()))
            with self.assertRaisesRegex(RuntimeError, T.BLOCKED): receipt.load_verified_single_file(None, None)
            with self.assertRaisesRegex(ValueError, 'load_verified'): receipt.ExactSingleFileReceipt()
            config = types.P4Config(mode='interference', sample_max_age_ns=1000,
                max_wait_ns=1000, internal_step_budget_ns=100)
            class ForeignReceipt: step_budget_ns = 100
            with self.assertRaisesRegex(ValueError, 'verified exact'): policy.P4Policy('cpu-test', config, single_file=ForeignReceipt())
            self.assertIn('type(single_file) is not ExactSingleFileReceipt', (src / 'p4_policy.py').read_text(encoding='utf-8'))
            self.assertIs(bridge.P4Policy, policy.P4Policy)
            self.assertIsInstance(bridge.NativeP4Bridge(policy.P4Policy('cpu-test', config)), bridge.NativeP4Bridge)

    def test_no_torch_vllm_or_gpu_backend_import(self):
        self.assertFalse(any(name == 'torch' or name.startswith(('torch.', 'vllm.', 'py_kvcache.')) for name in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
