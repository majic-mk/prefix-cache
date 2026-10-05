"""Read-only readiness fixtures. No hardware, model or payment is involved."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from probekv.p0_stage_readiness_v2 import (
    _BASE_REQUIRED, _STAGE_REQUIRED, _verify_cpu_artifacts, _validate_pair_scope,
    assess_stage_readiness, validate_p0_model_geometry,
)
from probekv.v8_schema10_execution import digest_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StageReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / 'workspace'; self.workspace.mkdir()
        self.output = self.root / 'evidence'; self.output.mkdir()
        self.refs = {}
        files = {}
        for name in _BASE_REQUIRED + _STAGE_REQUIRED['CONTROLLED_P0_E'] + ('docs/中文.md',):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('CPU fixture, not a native implementation\n', encoding='utf-8')
            files[name] = sha(path)
        self.worktree = dict(base_commit='c' * 40, files=files,
            digest=hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode()).hexdigest())
        self.raw_log = self.output / 'cpu.log'; self.raw_log.write_text('fixture test PASS\n')
        self.tests = [dict(test_id='cpu_fixture.test_readiness', observed='PASS', passed=True,
            runtime_commit=self.worktree['base_commit'], runtime_worktree_digest=self.worktree['digest'],
            evidence_path=str(self.raw_log), evidence_hash=sha(self.raw_log))]
        self.report = dict(status='CPU_CHECKS_PASS_NATIVE_PENDING',
            base_commit=self.worktree['base_commit'], worktree_digest=self.worktree['digest'],
            tests_run=1, passed=1, skipped=0, errors=0, failures=0, locked_test_accessed=False)
        checks = []
        for name in ('compileall.log', 'contract_validator.log', 'diff_check.log'):
            log = self.output / name; log.write_text('fixture command completed\n')
            checks.append(dict(log=name, log_sha256=sha(log), returncode=0, passed=True))
        self.commands = dict(repository_checks=checks)
        self.flush_cpu()

    def put(self, name, data):
        path = self.output / (name + '.json')
        path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False), encoding='utf-8')
        self.refs[name] = dict(path=str(path), sha256=sha(path))
        return path

    def flush_cpu(self):
        for name, value in (('cpu_report', self.report), ('worktree_files', self.worktree),
                            ('test_results', self.tests), ('commands', self.commands)):
            self.put(name, value)

    def assess(self, stage='CONTROLLED_P0_E'):
        return assess_stage_readiness(stage, workspace=self.workspace, evidence=self.refs, now_unix=1000.)

    def test_missing_environment_preserves_verified_cpu_without_requiring_p1_cohort(self):
        result = self.assess()
        self.assertTrue(result['local_cpu_inputs_verified'])
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertFalse(result['requires_natural_p1_cohort'])
        self.assertFalse(result['controlled_p0_inputs_ready'])
        self.assertFalse(result['gpu_execution_allowed'])
        self.assertFalse(result['automatic_rental_allowed'])
        self.assertNotIn('qualified_cohort', str(result['blockers']))

    def test_real_files_are_read_without_writes(self):
        before = {str(p): sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.assess()
        self.assertEqual(before, {str(p): sha(p) for p in self.root.rglob('*') if p.is_file()})

    def test_changed_code_invalidates_old_cpu_summary_even_with_same_commit(self):
        (self.workspace / _BASE_REQUIRED[0]).write_text('changed after tests\n')
        result = self.assess()
        self.assertFalse(result['local_cpu_inputs_verified'])
        self.assertIn('current worktree differs', result['blockers'][0]['detail'])

    def test_hand_written_pass_without_raw_test_files_is_not_evidence(self):
        self.tests = []; self.report.update(tests_run=0, passed=0)
        self.flush_cpu()
        self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_raw_log_corruption_blocks(self):
        self.raw_log.write_text('corrupt')
        self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_duplicate_tests_and_counter_mismatch_block(self):
        original = copy.deepcopy(self.tests)
        for tests, count in ((original * 2, 2), (original, 5)):
            self.tests = tests; self.report['tests_run'] = count; self.flush_cpu()
            self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_failed_command_or_forged_hash_block(self):
        self.commands['repository_checks'][0]['returncode'] = 1
        self.flush_cpu(); self.assertFalse(self.assess()['local_cpu_inputs_verified'])
        self.commands['repository_checks'][0]['returncode'] = 0
        self.commands['repository_checks'][0]['log_sha256'] = '0' * 64
        self.flush_cpu(); self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_new_skip_cannot_be_hidden_as_old_cpu_skip(self):
        self.tests[0].update(observed='SKIP', detail='new skipped check', passed=None)
        self.report.update(passed=0, skipped=1); self.flush_cpu()
        self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_existing_fail_closed_skip_is_explicitly_counted(self):
        self.tests[0].update(test_id='test_schema10_native_stage_progress.PreflightResourceTests.test_cpu_cannot_produce_real_preflight_rows',
            observed='SKIP', detail='CPU fail-closed test', passed=None)
        self.report.update(passed=0, skipped=1); self.flush_cpu()
        result = _verify_cpu_artifacts(self.workspace, self.refs, 'CONTROLLED_P0_E')
        self.assertEqual(result['skipped'], 1)

    def test_required_code_cannot_be_omitted_from_frozen_worktree(self):
        del self.worktree['files'][_BASE_REQUIRED[0]]
        self.worktree['digest'] = hashlib.sha256(json.dumps(self.worktree['files'], sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        self.report['worktree_digest'] = self.worktree['digest']
        self.tests[0]['runtime_worktree_digest'] = self.worktree['digest']
        self.flush_cpu()
        self.assertIn('omitted required implementation', self.assess()['blockers'][0]['detail'])

    def test_p1_cannot_inherit_controlled_p0_or_metadata_pass(self):
        for name in ('p0_real_model_evidence', 'qualified_cohort', 'source_construction_manifest', 's0_recipe'):
            self.put(name, {'passed': True, 'evidence_origin': 'cpu_fixture'})
        result = self.assess('P1_E')
        self.assertFalse(result['P1_E_execution_allowed'])
        self.assertTrue(result['requires_natural_p1_cohort'])
        self.assertIn('P1_QUALIFICATION_CONSUMER_NOT_IMPLEMENTED', [b['code'] for b in result['blockers']])

    def test_mixed_cannot_inherit_exact_implementation(self):
        result = self.assess('CONTROLLED_P0_M')
        self.assertFalse(result['local_cpu_inputs_verified'])
        self.assertIn('p0_mixed_', result['blockers'][0]['detail'])

    def test_bad_ref_sha_is_fail_closed(self):
        self.refs['cpu_report']['sha256'] = 'f' * 64
        self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_malformed_hashed_metadata_is_blocked_not_a_pass_flag(self):
        self.put('cpu_report', ['passed', True])
        self.put('native_manifest', ['passed', True])
        result = self.assess()
        self.assertFalse(result['local_cpu_inputs_verified'])
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertIn('NATIVE_ASSETS', [b['code'] for b in result['blockers']])

    def test_unsafe_worktree_path_is_fail_closed(self):
        self.worktree['files']['../outside.py'] = 'a' * 64
        self.worktree['digest'] = hashlib.sha256(json.dumps(self.worktree['files'], sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        self.report['worktree_digest'] = self.worktree['digest']
        self.flush_cpu()
        self.assertFalse(self.assess()['local_cpu_inputs_verified'])

    def test_unknown_stage_and_nonfinite_clock_rejected(self):
        with self.assertRaises(ValueError): self.assess('P1')
        for value in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                assess_stage_readiness('CONTROLLED_P0_E', workspace=self.workspace,
                                      evidence=self.refs, now_unix=value)

    def test_controlled_exact_requires_nonempty_matched_numerical_recipe(self):
        from tests.test_p0_exact_pair_v2 import ExactCapturePairTests
        fixture = ExactCapturePairTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        manifest = fixture.manifest
        for job in manifest['jobs']:
            job['input_origin'] = 'controlled_provenance_diagnostic'
        self.assertEqual(_validate_pair_scope(manifest, 'CONTROLLED_P0_E'), 1)
        manifest['jobs'][0]['input_origin'] = 'frozen_development'
        with self.assertRaises(ValueError): _validate_pair_scope(manifest, 'CONTROLLED_P0_E')
        manifest['exact_capture_pairs'] = []
        with self.assertRaises(ValueError): _validate_pair_scope(manifest, 'CONTROLLED_P0_E')

    def test_checks_do_not_query_hardware_or_execute_model(self):
        with patch('subprocess.check_output', side_effect=AssertionError('no process')):
            result = self.assess()
        self.assertFalse(result['actual_gpu_observed'])
        self.assertTrue(result['server_recheck_required'])

    def bound_fixture(self):
        from tests.test_p0_exact_pair_v2 import ExactCapturePairTests
        from probekv.source_comparison_v2 import runtime_binding_digest
        fixture = ExactCapturePairTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        jobs = fixture.manifest['jobs']
        for job in jobs:
            job.update(input_origin='controlled_provenance_diagnostic', upper_seconds=5.)
        source = dict(code_commit=self.report['base_commit'], model_signature='CPU-fixture-model',
                      tokenizer_hash='CPU-fixture-tokenizer')
        runtime = dict(source_provenance=source, cost_provenance={'gpu': 'CPU-fixture-no-GPU'})
        model = self.root / 'model'; model.mkdir()
        config = model / 'config.json'
        config.write_text(json.dumps(dict(num_hidden_layers=32, vocab_size=128, max_position_embeddings=64)))
        audit = self.put('fixture_model_audit', {'files': {'config.json': sha(config)}})
        runtime.update(model_key='mistralai/Mistral-7B-Instruct-v0.3', model_path=str(model),
            model_audit_path=str(audit), model_audit_sha256=sha(audit), max_model_len=64)
        pool = dict(kind='target_source_catalog_v2', epoch=0, use_epoch=0,
            content_opportunities={}, rows={}, config=dict(
                model_signature=source['model_signature'], tokenizer_hash=source['tokenizer_hash'],
                authorization_domain='CPU-fixture-domain', policy='ALLOW_MIXED_G1',
                max_bytes=100000, staging_bytes=10000, max_variants=4,
                probation_opportunities=2, purpose='isolated_P0_diagnostic'))
        self.put('initial_pool', dict(payload=pool, digest=digest_json(pool)))
        self.put('initial_registry', {'kind': 'CPU-fixture'})
        self.put('patch_audit', {'cacheblend_patch_sha256': 'a' * 64})
        self.put('patch_manifest', {'kind': 'CPU-fixture'})
        native = dict(binding=dict(code_commit=source['code_commit'], patch_sha256='a' * 64),
                      native_runtime=runtime)
        self.put('native_manifest', native)
        self.refs['instance_id'] = 'CPU-fixture-not-a-server'
        self.refs['installed_runtime_root'] = str(self.root / 'fixture-code-root')
        binding = dict(code_commit=source['code_commit'], runtime_digest=runtime_binding_digest(),
            patch_sha256='a' * 64, model_signature=source['model_signature'],
            tokenizer_hash=source['tokenizer_hash'], instance_id=self.refs['instance_id'],
            gpu_uuid='CPU-fixture-no-GPU', initial_pool_sha256=self.refs['initial_pool']['sha256'],
            initial_registry_sha256=self.refs['initial_registry']['sha256'],
            input_manifest_sha256=digest_json([dict(action_id=j['action_id'],
                request_sha256=j['request_sha256'], input_origin=j['input_origin']) for j in jobs]))
        manifest = dict(kind='bounded_native_p0_batch_v2', phase='P0', binding=binding,
            native_manifest_sha256=self.refs['native_manifest']['sha256'], jobs=jobs,
            exact_capture_pairs=fixture.manifest['exact_capture_pairs'],
            locked_test_accessed=False, automatic_rental_allowed=False,
            authority=dict(approval_reference='CPU fixture only, no authority',
                instance_id=binding['instance_id'], gpu_uuid=binding['gpu_uuid'],
                starts_at_unix=900., expires_at_unix=5000., unit_price_per_hour=2.,
                maximum_gpu_hours=1., maximum_cost=2.),
            limits=dict(initialization_upper_seconds=10., cleanup_seconds=5., maximum_actions=2,
                host_capture_bytes=10000, host_comparison_bytes=10000, cuda_comparison_bytes=10000),
            registry_budget=dict(max_bytes=100000, max_manifest_bytes=10000))
        self.sign_batch(manifest)
        return runtime, manifest

    def sign_batch(self, manifest):
        manifest['manifest_sha256'] = digest_json({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        self.put('batch_manifest', manifest)

    def fixture_assess(self, runtime):
        # Only external environment consumers are substituted. Real local
        # test/file/pair/batch/authority validators still run in these tests.
        with ExitStack() as stack:
            stack.enter_context(patch('probekv.v8_schema10_native_factory.validate_native_attachment', return_value=runtime))
            stack.enter_context(patch('probekv.v8_schema10_native_factory.verify_installed_runtime_sources'))
            stack.enter_context(patch('probekv.cacheblend_patch.validate_native_patch_audit'))
            return self.assess()

    def test_complete_input_fixture_still_does_not_authorize_gpu_or_qualify_P0(self):
        runtime, _ = self.bound_fixture()
        result = self.fixture_assess(runtime)
        self.assertEqual(result['blockers'], [])
        self.assertTrue(result['controlled_p0_inputs_ready'])
        self.assertEqual(result['status'], 'INPUTS_VERIFIED_RUNTIME_RECHECK_REQUIRED')
        self.assertFalse(result['gpu_execution_allowed'])
        self.assertFalse(result['P0_E_qualified'])
        self.assertFalse(result['P1_E_execution_allowed'])

    def test_expired_authority_zero_price_and_insufficient_cleanup_budget_block(self):
        runtime, manifest = self.bound_fixture()
        original = copy.deepcopy(manifest)
        for updates in ({'expires_at_unix': 999.}, {'unit_price_per_hour': 0.},
                        {'maximum_cost': .00001}, {'approval_reference': ''}):
            manifest = copy.deepcopy(original); manifest['authority'].update(updates)
            self.sign_batch(manifest)
            with self.subTest(updates=updates):
                result = self.fixture_assess(runtime)
                self.assertFalse(result['controlled_p0_inputs_ready'])
                self.assertIn('SCOPED_AUTHORITY_AND_BATCH_BINDINGS', [b['code'] for b in result['blockers']])

    def test_stale_runtime_or_pool_digest_cannot_use_valid_authority(self):
        runtime, manifest = self.bound_fixture()
        for field in ('runtime_digest', 'initial_pool_sha256'):
            previous = manifest['binding'][field]
            manifest['binding'][field] = 'f' * 64; self.sign_batch(manifest)
            self.assertFalse(self.fixture_assess(runtime)['controlled_p0_inputs_ready'])
            manifest['binding'][field] = previous

    def test_no_operator_instance_cannot_inherit_historical_instance(self):
        runtime, _ = self.bound_fixture()
        del self.refs['instance_id']
        self.assertFalse(self.fixture_assess(runtime)['controlled_p0_inputs_ready'])

    def test_flat_or_tampered_catalog_blocks_even_when_outer_file_hash_is_rebound(self):
        runtime, manifest = self.bound_fixture()
        original = json.loads(Path(self.refs['initial_pool']['path']).read_text(encoding='utf-8'))
        tampered = copy.deepcopy(original)
        tampered['payload']['config']['max_variants'] = 2  # Deliberately stale inner digest.
        for bad in (original['payload'], tampered):
            self.put('initial_pool', bad)
            manifest['binding']['initial_pool_sha256'] = self.refs['initial_pool']['sha256']
            self.sign_batch(manifest)
            result = self.fixture_assess(runtime)
            self.assertFalse(result['controlled_p0_inputs_ready'])
            self.assertIn('SCOPED_AUTHORITY_AND_BATCH_BINDINGS', [b['code'] for b in result['blockers']])

    def test_server_dry_run_reads_actual_empty_store_envelope_without_gpu_or_writes(self):
        from scripts.server import run_decoupled_v2_p0 as entry
        from probekv.source_manifest_v2 import RequestManifestRegistry
        from probekv.source_store_v2 import TargetSourceStoreV2
        runtime, manifest = self.bound_fixture()
        registry_root = self.root / 'actual-registry'
        store_root = self.root / 'actual-store'
        registry = RequestManifestRegistry(persistent_root=registry_root,
            max_bytes=100000, max_manifest_bytes=10000)
        store = TargetSourceStoreV2(store_root, registry=registry,
            model_signature=runtime['source_provenance']['model_signature'],
            tokenizer_hash=runtime['source_provenance']['tokenizer_hash'],
            authorization_domain='CPU-fixture-domain', policy='ALLOW_MIXED_G1',
            max_bytes=100000, staging_bytes=10000)
        store.close()
        catalog_path = store_root / 'catalog.json'; registry_path = registry_root / 'registry.json'
        before = catalog_path.read_bytes(), registry_path.read_bytes()
        manifest['binding'].update(initial_pool_sha256=sha(catalog_path),
                                   initial_registry_sha256=sha(registry_path))
        self.sign_batch(manifest)
        output = self.root / 'server-dry-run'
        argv = ['run_decoupled_v2_p0.py', '--manifest', self.refs['batch_manifest']['path'],
            '--native-manifest', self.refs['native_manifest']['path'], '--store-root', str(store_root),
            '--registry-root', str(registry_root), '--instance-id', self.refs['instance_id'],
            '--output', str(output)]

        def git_only(command, **kwargs):
            self.assertEqual(command, ['git', 'rev-parse', 'HEAD'])
            return self.report['base_commit'] + '\n'

        with patch.object(sys, 'argv', argv), \
                patch.object(entry, 'validate_native_attachment', return_value=runtime), \
                patch.object(entry.subprocess, 'check_output', side_effect=git_only), \
                patch.object(entry.time, 'time', return_value=1000.), \
                patch.object(entry, 'validate_p0_batch', wraps=entry.validate_p0_batch) as validator, \
                patch.object(entry, 'run_p0_native_batch', side_effect=AssertionError('no model execution')), \
                patch('probekv.v8_schema10_native_factory.create_native_backend',
                      side_effect=AssertionError('no model load')), \
                patch('probekv.v8_schema10_native_factory.create_native_measurement_backend',
                      side_effect=AssertionError('no model load')):
            self.assertEqual(entry.main(), 0)
            validator.assert_called_once()
        report = json.loads((output / 'preflight.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'INPUTS_VALIDATED')
        self.assertEqual(report['model_geometry']['num_layers'], 32)
        self.assertFalse(report['execute_requested'])
        self.assertFalse(report['gpu_identity_observed'])
        self.assertEqual((catalog_path.read_bytes(), registry_path.read_bytes()), before)

        # Rebinding the outer file hash must not hide a stale inner catalog
        # digest; rejection occurs before git/GPU/model calls or output writes.
        damaged = json.loads(before[0]); damaged['payload']['epoch'] = 1
        catalog_path.write_text(json.dumps(damaged), encoding='utf-8')
        manifest['binding']['initial_pool_sha256'] = sha(catalog_path)
        self.sign_batch(manifest)
        argv[-1] = str(self.root / 'tampered-dry-run')
        with patch.object(sys, 'argv', argv), \
                patch.object(entry, 'validate_native_attachment', return_value=runtime), \
                patch.object(entry.subprocess, 'check_output', side_effect=AssertionError('no process')):
            with self.assertRaisesRegex(ValueError, 'catalog digest'): entry.main()
        self.assertFalse(Path(argv[-1]).exists())


class ModelGeometryPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / 'config.json'
        self.audit_path = self.root / 'model_audit.json'
        self.config = dict(num_hidden_layers=32, vocab_size=128, max_position_embeddings=64)
        self.runtime = dict(model_key='mistralai/Mistral-7B-Instruct-v0.3', model_path=str(self.root),
            model_audit_path=str(self.audit_path), max_model_len=64)
        self.bind_config()
        from tests.test_p0_mixed_pair_v2 import MixedPairTests
        fixture = MixedPairTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.manifest = copy.deepcopy(fixture.manifest)
        for job in self.manifest['jobs']:
            job['repair_positions_by_layer'] = {str(layer): [1] for layer in range(2, 33)}

    def bind_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding='utf-8')
        self.audit_path.write_text(json.dumps({'files': {'config.json': sha(self.config_path)}}), encoding='utf-8')
        self.runtime['model_audit_sha256'] = sha(self.audit_path)

    def validate(self):
        return validate_p0_model_geometry(self.manifest, self.runtime)

    def test_real_config_geometry_is_checked_without_loading_model(self):
        with patch('subprocess.check_output', side_effect=AssertionError('no GPU/process')):
            report = self.validate()
        self.assertEqual(report['num_layers'], 32)
        self.assertEqual(report['vocab_size'], 128)
        self.assertFalse(report['model_loaded'])
        self.assertFalse(report['gpu_observed'])

    def test_three_layer_mask_cannot_self_declare_mistral_model_end(self):
        self.manifest['jobs'][0]['repair_positions_by_layer'] = {'2': [1], '3': [1]}
        with self.assertRaisesRegex(ValueError, 'model end'): self.validate()

    def test_wrong_model_layer_count_is_rejected_even_with_new_config_digest(self):
        self.config['num_hidden_layers'] = 3; self.bind_config()
        with self.assertRaisesRegex(ValueError, 'frozen adapter'): self.validate()

    def test_qwen_uses_its_own_28_layer_geometry(self):
        self.runtime['model_key'] = 'Qwen/Qwen2.5-7B-Instruct'
        self.config['num_hidden_layers'] = 28; self.bind_config()
        with self.assertRaises(ValueError): self.validate()
        for job in self.manifest['jobs']:
            job['repair_positions_by_layer'] = {str(layer): [1] for layer in range(2, 29)}
        self.assertEqual(self.validate()['num_layers'], 28)

    def test_stale_config_and_missing_config_audit_are_rejected(self):
        self.config_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'verified audit'): self.validate()
        self.bind_config()
        self.audit_path.write_text(json.dumps({'files': {}}))
        self.runtime['model_audit_sha256'] = sha(self.audit_path)
        with self.assertRaisesRegex(ValueError, 'verified audit'): self.validate()

    def test_out_of_vocabulary_prompt_or_teacher_is_rejected(self):
        original = copy.deepcopy(self.manifest)
        for name, value in (('token_ids', 128), ('teacher_token_ids', 128),
                            ('token_ids', -1), ('teacher_token_ids', True)):
            self.manifest = copy.deepcopy(original)
            self.manifest['jobs'][0]['request'][name][0] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError): self.validate()

    def test_generation_and_runtime_context_are_bounded_by_audited_model(self):
        self.runtime['max_model_len'] = 8
        with self.assertRaisesRegex(ValueError, 'context bound'): self.validate()
        self.runtime['max_model_len'] = 65
        with self.assertRaisesRegex(ValueError, 'context geometry'): self.validate()
        self.runtime['max_model_len'] = True
        with self.assertRaises(ValueError): self.validate()

    def test_completed_depth_cannot_observe_beyond_model(self):
        self.manifest['jobs'] = [dict(action_id='source', operation='source_request',
            request=dict(token_ids=[1, 2, 3], max_new_tokens=1),
            comparison_profile={'completed_depth': 32})]
        with self.assertRaisesRegex(ValueError, 'd\+1'): self.validate()
        self.manifest['jobs'][0]['comparison_profile']['completed_depth'] = 2
        self.assertEqual(len(self.validate()['jobs']), 1)

    def test_server_entry_rejects_toy_masks_before_GPU_probe_or_factory_with_or_without_execute(self):
        from scripts.server import run_decoupled_v2_p0 as entry
        self.manifest['jobs'][0]['repair_positions_by_layer'] = {'2': [1], '3': [1]}
        native_path = self.root / 'native.json'; native_path.write_text('{}')
        self.manifest['native_manifest_sha256'] = sha(native_path)
        manifest_path = self.root / 'manifest.json'
        manifest_path.write_text(json.dumps(self.manifest))
        argv = ['run_decoupled_v2_p0.py', '--manifest', str(manifest_path),
            '--native-manifest', str(native_path), '--store-root', str(self.root / 'missing-store'),
            '--registry-root', str(self.root / 'missing-registry'), '--instance-id', 'CPU-fixture',
            '--output', str(self.root / 'out')]
        for extra in ([], ['--execute']):
            with patch.object(sys, 'argv', argv + extra), \
                    patch.object(entry, 'validate_native_attachment', return_value=self.runtime), \
                    patch.object(entry.subprocess, 'check_output', side_effect=AssertionError('no process')), \
                    patch('probekv.v8_schema10_native_factory.create_native_backend',
                          side_effect=AssertionError('no model')), \
                    patch('probekv.v8_schema10_native_factory.create_native_measurement_backend',
                          side_effect=AssertionError('no model')):
                with self.assertRaisesRegex(ValueError, 'model end'): entry.main()
            self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
