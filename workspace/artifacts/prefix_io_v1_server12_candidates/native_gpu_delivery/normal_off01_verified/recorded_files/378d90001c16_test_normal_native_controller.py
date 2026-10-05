"""CPU rejection examples: fixture bytes never issue a positive native receipt."""
from copy import deepcopy
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'prefix_io_control')
ATTEMPTS = []


class NoGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            ATTEMPTS.append(fullname)
            raise RuntimeError('CPU controller test attempted GPU/model import: ' + fullname)
        return None


sys.meta_path.insert(0, NoGpuImports())
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_cpu_normal_controller', HERE / 'control_p4_single_file.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
GPU = 'GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac'


class ControllerRejections(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / C.D).mkdir(parents=True)

    def write(self, relative, value, *, raw=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if raw else json.dumps(value).encode())
        return C.ref(relative, self.root)

    def config(self, mode='off'):
        previous = None if mode == 'off' else dict(path=C.result_path('off' if mode == 'shadow' else 'shadow'),
            bytes=1, sha256='0' * 64)
        return C.expected_config(self.root, mode, GPU, previous)

    def partial_authority(self, config):
        # These fixture refs cannot establish GPU/native qualification. Tests
        # always mutate or omit a mandatory authoritative input and expect reject.
        self.write(C.config_path(config['mode']), config)
        self.write(C.LOCK, {'fixture_only': True})
        self.write(C.BINDING, {'fixture_only': True})
        mode = config['mode']
        return dict(schema='c5_normal_mode_site_authority_v1', status='LIVE_HUMAN_BOUND',
            origin='live_site_binding', synthetic=False, root=str(self.root), mode=mode,
            label=C.label(mode), gpu_uuid=GPU, config_ref=C.ref(C.config_path(mode), self.root),
            source_lock_ref=C.ref(C.LOCK, self.root), binding_ref=C.ref(C.BINDING, self.root),
            previous_qualification_ref=config['previous_qualification_ref'], seconds_limit=300,
            reserved_seconds=320, max_attempts=1, max_processes=1,
            context_ref={'path': C.D + '/LIVE_CONTEXT_' + mode + '.json', 'bytes': 1, 'sha256': '0' * 64},
            human_grant_ref={'path': C.D + '/HUMAN_GPU_GRANT_' + mode + '.json', 'bytes': 1, 'sha256': '0' * 64},
            effective_permission_ref={'path': C.D + '/EFFECTIVE_GPU_PERMISSION_' + mode + '.json', 'bytes': 1, 'sha256': '0' * 64},
            base_permission_ref={'path': C.BASE_PERMISSION, 'bytes': 1, 'sha256': '0' * 64},
            guard_source_ref={'path': C.GUARD, 'bytes': 1, 'sha256': C.GUARD_SHA})

    def test_path_traversal_rejected(self):
        for relative in ('../outside', C.D + '/../outside', '/tmp/outside', 'C:/outside', 'a\\b', 'a//b', 'a/./b'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                C.safe(relative, self.root)

    def test_relative_argparse_path_not_resolved_against_cwd(self):
        path = Path(C.config_path('off'))
        self.assertEqual(C.config_relative(self.root, path), C.config_path('off'))
        self.assertEqual(C.config_relative(self.root, self.root / path), C.config_path('off'))

    def test_other_revision_config_rejected(self):
        with self.assertRaises(ValueError):
            C.config_relative(self.root, C.V6 + '/NATIVE_COST_CONFIG.json')

    def test_source_drift_rejected(self):
        row = self.write('source.py', b'original fixture\n', raw=True)
        self.write('source.py', b'changed fixture\n', raw=True)
        with self.assertRaises(ValueError):
            C.verify_reference(row, self.root)

    def test_typed_reference_rejects_bool_size(self):
        row = self.write('source.py', b'x', raw=True); row['bytes'] = True
        with self.assertRaises(ValueError):
            C.verify_reference(row, self.root)

    def test_empty_initializer_is_valid_source_metadata(self):
        row = self.write('fixture_package/__init__.py', b'', raw=True)
        self.assertEqual(C.verify_reference(row, self.root), row)
        self.assertEqual(C.rows_map([row], self.root, verify_bytes=True), {row['path']: row})
        # A bounded dynamic controller/receipt module must still contain code.
        with self.assertRaisesRegex(ValueError, 'nonempty bounded source module'):
            C.load_module(self.root, row, '_empty_source_fixture')

    def test_reference_size_rejects_negative_and_noninteger(self):
        original = self.write('source.py', b'x', raw=True)
        for size in (-1, 1.0, '1', None):
            row = dict(original, bytes=size)
            with self.subTest(size=size), self.assertRaises(ValueError):
                C.verify_reference(row, self.root)

    def test_duplicate_json_key_rejected(self):
        self.write('bad.json', b'{"mode":"off","mode":"on"}', raw=True)
        with self.assertRaises(ValueError):
            C.read('bad.json', self.root)

    def test_nonfinite_json_rejected(self):
        self.write('bad.json', b'{"cost":NaN}', raw=True)
        with self.assertRaises(ValueError):
            C.read('bad.json', self.root)

    def test_duplicate_common_source_rejected(self):
        row = self.write('source.py', b'x', raw=True)
        with self.assertRaises(ValueError):
            C.rows_map([row, row], self.root)

    def test_per_mode_reference_cannot_enter_common_lock(self):
        row = self.write(C.config_path('off'), {'fixture_only': True})
        self.write(C.LOCK, dict(schema=C.LOCK_SCHEMA, scope=C.SCOPE, GPU_operations=0,
            normal_runtime_qualified=False, gpu_authority_issued=False, files=[row]))
        with self.assertRaisesRegex(ValueError, 'outside shared common lock'):
            C.common_rows(self.root)

    def test_per_mode_ledger_snapshot_cannot_enter_common_lock(self):
        row = self.write(C.D + '/LEDGER_BEFORE_off.json', {'fixture_only': True})
        self.write(C.LOCK, dict(schema=C.LOCK_SCHEMA, scope=C.SCOPE, GPU_operations=0,
            normal_runtime_qualified=False, gpu_authority_issued=False, files=[row]))
        with self.assertRaisesRegex(ValueError, 'outside shared common lock'):
            C.common_rows(self.root)

    def test_common_lock_cannot_reference_itself(self):
        self.write(C.LOCK, {'fixture_only': True}); row = C.ref(C.LOCK, self.root)
        self.write(C.LOCK, dict(schema=C.LOCK_SCHEMA, scope=C.SCOPE, GPU_operations=0,
            normal_runtime_qualified=False, gpu_authority_issued=False, files=[row]))
        with self.assertRaisesRegex(ValueError, 'outside shared common lock'):
            C.common_rows(self.root)

    def test_configuration_fixed_limits_reject_bool(self):
        config = self.config(); config['seconds_limit'] = True
        with self.assertRaises(ValueError):
            C.validate_configuration_document(self.root, config, C.config_path('off'))

    def test_configuration_unknown_uuid_rejected(self):
        config = self.config(); config['gpu_uuid'] = None
        with self.assertRaises(ValueError):
            C.validate_configuration_document(self.root, config, C.config_path('off'))

    def test_configuration_extraneous_authorization_flag_rejected(self):
        config = self.config(); config['gpu_launch_allowed'] = True
        with self.assertRaises(ValueError):
            C.validate_configuration_document(self.root, config, C.config_path('off'))

    def test_shadow_requires_off_prerequisite_path(self):
        config = self.config('shadow'); config['previous_qualification_ref']['path'] = C.result_path('shadow')
        with self.assertRaises(ValueError):
            C.validate_configuration_document(self.root, config, C.config_path('shadow'))

    def test_off_cannot_have_prerequisite(self):
        config = self.config(); config['previous_qualification_ref'] = {'fixture_only': True}
        with self.assertRaises(ValueError):
            C.validate_configuration_document(self.root, config, C.config_path('off'))

    def test_consumed_common_grant_rejected(self):
        config = self.config(); authority = self.partial_authority(config)
        authority['human_grant_ref']['path'] = C.V6 + '/HUMAN_GPU_GRANT.json'
        self.write(C.authority_path('off'), authority)
        with self.assertRaisesRegex(ValueError, 'fresh normal reference path: human_grant_ref'):
            C.load_authority(self.root, config)

    def test_synthetic_authority_rejected(self):
        config = self.config(); authority = self.partial_authority(config); authority['synthetic'] = True
        self.write(C.authority_path('off'), authority)
        with self.assertRaisesRegex(ValueError, 'separate normal mode'):
            C.load_authority(self.root, config)

    def test_old_job_label_authority_rejected(self):
        config = self.config(); authority = self.partial_authority(config)
        authority['label'] = 'server11-c5-native-common-cost-gpu03'
        self.write(C.authority_path('off'), authority)
        with self.assertRaises(ValueError):
            C.load_authority(self.root, config)

    def test_source_lock_authority_drift_rejected(self):
        config = self.config(); authority = self.partial_authority(config)
        self.write(C.authority_path('off'), authority); self.write(C.LOCK, {'fixture_only': 'changed'})
        with self.assertRaisesRegex(ValueError, 'separate normal mode'):
            C.load_authority(self.root, config)

    def test_missing_authority_blocks_configuration_before_receipt(self):
        config = self.config(); self.write(C.config_path('off'), config)
        paths = [C.D + '/' + name for name in ('control_p4_single_file.py', 'run_p4_single_file_experiment.py',
            'verify_p4_single_file.py', 'combined_runtime_contract.py')]
        paths += [C.SITE_SDK, C.RECEIPT, C.BINDING, C.GUARD, C.BASE_PERMISSION]
        refs = {path: self.write(path, b'fixture rejection source\n', raw=True) for path in paths}
        with patch.object(C, 'common_rows', return_value=({}, refs)), patch.object(C, 'load_receipt') as receipt:
            with self.assertRaises((ValueError, FileNotFoundError)):
                C.verify_configuration(self.root, Path(C.config_path('off')))
            receipt.assert_not_called()

    def test_unauthorized_launch_does_not_probe_gpu(self):
        self.write(C.config_path('off'), self.config())
        with patch.object(C.subprocess, 'run') as probe, patch.object(C.subprocess, 'Popen') as launch:
            with self.assertRaises((ValueError, FileNotFoundError)):
                C.launch('off', self.root)
            probe.assert_not_called(); launch.assert_not_called()

    def test_receipt_rejects_changed_canonical_before_import(self):
        self.write(C.RECEIPT, b'raise RuntimeError("must not load")\n', raw=True)
        with self.assertRaisesRegex(ValueError, 'pinned public historical-compatible canonical'):
            C.load_receipt(self.root)

    def test_freeze_rejects_wrong_calibration_before_receipt(self):
        self.write(C.CALIBRATION_BINDING, {'origin': 'synthetic_fixture_only'})
        with patch.object(C, 'load_receipt') as receipt:
            with self.assertRaisesRegex(ValueError, 'calibration binding provenance'):
                C.freeze_common('0' * 64, self.root)
            receipt.assert_not_called()

    def test_no_mode_authority_created_by_invalid_freeze(self):
        with self.assertRaises(ValueError):
            C.freeze_common(None, self.root)
        self.assertFalse(list((self.root / C.D).glob('AUTHORITY_*.json')))
        self.assertFalse(list((self.root / C.D).glob('EFFECTIVE_GPU_PERMISSION_*.json')))

    def test_prerequisite_flags_do_not_bypass_real_replay(self):
        config = self.config('shadow')
        binding_ref = self.write(C.BINDING, {'fixture_only': True})
        self.write(C.LOCK, {'fixture_only': True})
        self.write(C.result_path('off'), dict(scope=C.SCOPE, mode='off', native_execution_verified=True,
            runtime_condition_qualified=True, permits_next_mode='shadow', source_lock_ref=C.ref(C.LOCK, self.root),
            binding_ref=binding_ref, evidence_refs={}))
        config['previous_qualification_ref'] = C.ref(C.result_path('off'), self.root)
        with self.assertRaisesRegex(ValueError, 'complete prior raw evidence'):
            C.verify_previous_qualification(self.root, config, {C.BINDING: binding_ref})

    def test_guard_budget_rejects_bool(self):
        with self.assertRaises(ValueError):
            C.number(True, 'used GPU budget')

    def test_expired_context_cannot_start_scope(self):
        with self.assertRaisesRegex(ValueError, 'expired'):
            C.validate_live_interval(dict(observed_unix=time.time() - 120,
                valid_until_unix=time.time() - 60), require_current=True)

    def test_launch_timestamp_outside_grant_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, 'launch outside'):
            C.validate_live_interval(dict(observed_unix=time.time() - 30,
                valid_until_unix=time.time() + 60), launch_unix=time.time() + 120)

    def test_unbounded_context_validity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bounded genuine'):
            C.validate_live_interval(dict(observed_unix=time.time(), valid_until_unix=time.time() + 7200))

    def test_live_ledger_cannot_be_in_common_lock(self):
        row = self.write(C.LEDGER, {'fixture_only': True})
        self.write(C.LOCK, dict(schema=C.LOCK_SCHEMA, scope=C.SCOPE, GPU_operations=0,
            normal_runtime_qualified=False, gpu_authority_issued=False, files=[row]))
        with self.assertRaisesRegex(ValueError, 'outside shared common lock'):
            C.common_rows(self.root)

    def test_launch_intent_requires_immutable_before_snapshot(self):
        config = self.config()
        paths = [C.D + '/SCOPE_off.json', C.config_path('off'), C.authority_path('off'), C.LOCK,
            C.BINDING, C.D + '/SOURCE_off_BEFORE.json', C.D + '/SOURCE_off_LAUNCH.json',
            C.D + '/LEDGER_BEFORE_off.json', C.LEDGER]
        refs = {path: self.write(path, {'fixture_only': True}) for path in paths}
        authority = dict(human_grant_ref=refs[C.authority_path('off')], effective_permission_ref=refs[C.config_path('off')])
        intent = dict(command=C.guard_command('off'), scope_ref=refs[C.D + '/SCOPE_off.json'],
            config_ref=refs[C.config_path('off')], authority_ref=refs[C.authority_path('off')],
            source_lock_ref=refs[C.LOCK], binding_ref=refs[C.BINDING],
            human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
            source_before_ref=refs[C.D + '/SOURCE_off_BEFORE.json'],
            launch_source_ref=refs[C.D + '/SOURCE_off_LAUNCH.json'], ledger_before_ref=refs[C.LEDGER])
        self.write(C.D + '/LAUNCH_INTENT_off.json', intent)
        with (patch.object(C, 'verify_scope'), patch.object(C, 'verify_source_proof')):
            with self.assertRaisesRegex(ValueError, 'frozen prelaunch normal intent'):
                C.verify_launch_intent(self.root, config, authority, {})

    def test_old_guard_reservation_shape_rejected(self):
        config = self.config()
        authority = {'effective_permission_ref': dict(path=C.D + '/EFFECTIVE_GPU_PERMISSION_off.json',
            bytes=1, sha256='0' * 64)}
        self.write(C.LEDGER, dict(gpu_wall_seconds=0, active_reservation=dict(reservation_id='wrong',
            label=config['label'], gpu_uuid=GPU), events=[]))
        with (patch.object(C, 'verify_configuration', return_value=(config, {}, authority)),
                patch.object(C, 'verify_launch_intent', return_value={})):
            with self.assertRaises(ValueError):
                C.verify_active_guard(self.root, config, authority)

    def test_exact_current_host_metadata_is_only_a_marker_not_a_receipt(self):
        # Actual host bytes must separately be audited on the real Linux site.
        self.assertIsNone(C.metadata_reference(dict(C.HOST_DRIVER_REF), self.root))

    def test_unknown_external_metadata_rejected(self):
        value = dict(C.HOST_DRIVER_REF, path='/usr/lib/libcuda.so.unapproved')
        with self.assertRaisesRegex(ValueError, 'external'):
            C.metadata_reference(value, self.root)

    def test_current_host_driver_digest_drift_rejected(self):
        value = dict(C.HOST_DRIVER_REF, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'pinned host driver'):
            C.metadata_reference(value, self.root)

    def test_current_host_boolean_size_rejected(self):
        value = dict(C.HOST_DRIVER_REF, bytes=True)
        with self.assertRaisesRegex(ValueError, 'typed immutable metadata'):
            C.metadata_reference(value, self.root)

    def test_absolute_project_metadata_retains_real_byte_hash_and_raw_value(self):
        original = self.write('sdk/cpu_fixture_only.json', {'fixture_only': True})
        raw_value = dict(original, path=str(self.root / original['path']))
        expected = deepcopy(raw_value)
        self.assertEqual(C.metadata_reference(raw_value, self.root), original)
        self.assertEqual(raw_value, expected)

    def test_absolute_project_metadata_byte_drift_rejected(self):
        original = self.write('sdk/cpu_fixture_only.json', {'fixture_only': True})
        raw_value = dict(original, path=str(self.root / original['path']), sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            C.metadata_reference(raw_value, self.root)

    def test_absolute_project_metadata_traversal_rejected(self):
        original = self.write('sdk/cpu_fixture_only.json', {'fixture_only': True})
        raw_value = dict(original, path=str(self.root / 'sdk' / '..' / 'sdk' / 'cpu_fixture_only.json'))
        with self.assertRaises(ValueError):
            C.metadata_reference(raw_value, self.root)


if __name__ == '__main__':
    program = unittest.main(verbosity=2, exit=False)
    loaded = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    print('CPU_NORMAL_CONTROLLER_IMPORT_AUDIT ' + json.dumps(dict(
        forbidden_imports=loaded, forbidden_import_attempts=ATTEMPTS,
        tests_run=program.result.testsRun, failures=len(program.result.failures),
        errors=len(program.result.errors), skipped=len(program.result.skipped), actual_gpu_runs=0), sort_keys=True))
    raise SystemExit(0 if program.result.wasSuccessful() and not loaded and not ATTEMPTS else 1)
