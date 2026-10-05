"""CPU-only path/authority/resource rejects; no native-positive receipt fixture."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_entry_binding as B
import control_native_cost_job as C
FORBIDDEN_IMPORTS = []


class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in ('torch', 'vllm', 'py_kvcache', 'prefix_io_control', 'pynvml'):
            FORBIDDEN_IMPORTS.append(fullname)
            raise ImportError('CPU binding tests prohibit GPU/model imports: ' + fullname)
        return None


sys.meta_path.insert(0, NoGpuImports())


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='c5-invalid-binding-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        return path

    def resource(self):
        # Pure validation of numeric/path prerequisites; this is never persisted
        # as a live context, native run, grant or performance qualification.
        return dict(origin='live_linux_readonly_resource_probe', status='LIVE_RESOURCE_READY',
            nvidia_nodes_visible=True, gpu_uuid='GPU-12345678-1234-1234-1234-123456789abc',
            gpu_free_mib=28000, compute_processes=[], cpu_cores=1,
            driver_version=B.SITE_DRIVER_VERSION,
            cpu_affinity_count=1, cpu_counters_readable=True,
            available_memory_bytes=32 * 1024**3, primary_free_bytes=10 * 1024**3)

    def device_metadata_fixture(self, modes):
        # Names enumerate real temporary CPU files; only lstat metadata is
        # substituted. No device node is created or any GPU telemetry called.
        directory = self.root / 'cpu-device-metadata'
        directory.mkdir()
        for name in modes:
            (directory / name).write_text('CPU metadata fixture only', encoding='utf-8')
        original_lstat = Path.lstat
        def fixture_lstat(path):
            if path.parent == directory and path.name in modes:
                return SimpleNamespace(st_mode=modes[path.name] | 0o600)
            return original_lstat(path)
        with patch.object(Path, 'lstat', fixture_lstat):
            return B.nvidia_device_nodes(directory), directory

    def test_character_gpu_zero_with_character_control_is_visible(self):
        nodes, directory = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia0': stat.S_IFCHR})
        self.assertEqual(nodes, [str(directory / 'nvidia0')])

    def test_character_gpu_five_without_gpu_zero_is_visible(self):
        nodes, directory = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia5': stat.S_IFCHR})
        self.assertEqual(nodes, [str(directory / 'nvidia5')])
        self.assertFalse((directory / 'nvidia0').exists())

    def test_numeric_gpu_without_control_node_is_not_visible(self):
        nodes, _ = self.device_metadata_fixture({'nvidia5': stat.S_IFCHR})
        self.assertEqual(nodes, [])

    def test_uvm_and_capability_nodes_are_not_numeric_gpu_nodes(self):
        nodes, _ = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia-uvm': stat.S_IFCHR,
            'nvidia-uvm-tools': stat.S_IFCHR, 'nvidia-caps': stat.S_IFDIR})
        self.assertEqual(nodes, [])

    def test_regular_numeric_gpu_file_is_not_a_character_device(self):
        nodes, _ = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia5': stat.S_IFREG})
        self.assertEqual(nodes, [])

    def test_regular_control_file_cannot_enable_character_gpu(self):
        nodes, _ = self.device_metadata_fixture({'nvidiactl': stat.S_IFREG, 'nvidia5': stat.S_IFCHR})
        self.assertEqual(nodes, [])

    def test_symlink_numeric_gpu_does_not_satisfy_device_metadata_gate(self):
        nodes, _ = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia5': stat.S_IFLNK})
        self.assertEqual(nodes, [])

    def test_numeric_directory_is_not_a_character_gpu(self):
        nodes, _ = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR, 'nvidia5': stat.S_IFDIR})
        self.assertEqual(nodes, [])

    def test_only_strict_numeric_character_gpu_names_are_returned(self):
        nodes, directory = self.device_metadata_fixture({'nvidiactl': stat.S_IFCHR,
            'nvidia5': stat.S_IFCHR, 'nvidia0': stat.S_IFCHR, 'nvidia5-extra': stat.S_IFCHR,
            'nvidia': stat.S_IFCHR, 'nvidia-caps': stat.S_IFDIR})
        self.assertEqual(nodes, [str(directory / 'nvidia0'), str(directory / 'nvidia5')])

    def test_duplicate_top_level_json_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'duplicate JSON'):
            B.parse('{"status":"UNBOUND","status":"LIVE_HUMAN_BOUND"}')

    def test_duplicate_nested_json_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'duplicate JSON'):
            B.parse('{"permission":{"approved_gpu_ids":[],"approved_gpu_ids":["other"]}}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            B.parse('{"seconds":NaN}')

    def test_valid_json_remains_unbound(self):
        value = B.parse('{"status":"UNBOUND","gpu_uuid":null}')
        self.assertEqual(value, dict(status='UNBOUND', gpu_uuid=None))

    def test_relative_traversal_rejected(self):
        for path in ('../ledger.json', 'a/../b', './b', 'a//b', '/etc/passwd', 'C:/file', 'a\\b'):
            with self.subTest(path=path), self.assertRaises(RuntimeError): B.safe(self.root, path)

    def test_directory_symlink_rejected(self):
        target = self.root / 'real'; target.mkdir()
        link = self.root / 'link'
        try: link.symlink_to(target, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege still exercises the same branch.
            with patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'symlink'): B.safe(self.root, 'link/file')
            return
        with self.assertRaisesRegex(RuntimeError, 'symlink'): B.safe(self.root, 'link/file')

    def test_reference_drift_rejected(self):
        path = self.put('cpu.json', {'origin': 'synthetic_cpu'})
        old = B.ref(self.root, 'cpu.json')
        path.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'drift'): B.checked_ref(self.root, old)

    def test_reference_boolean_length_rejected(self):
        self.put('cpu.json', {})
        row = B.ref(self.root, 'cpu.json'); row['bytes'] = True
        with self.assertRaisesRegex(RuntimeError, 'invalid immutable'): B.checked_ref(self.root, row)

    def test_reference_unknown_fields_rejected(self):
        self.put('cpu.json', {})
        row = B.ref(self.root, 'cpu.json'); row['origin'] = 'native_gpu_recording'
        with self.assertRaisesRegex(RuntimeError, 'invalid immutable'): B.checked_ref(self.root, row)

    def test_duplicate_source_path_rejected(self):
        self.put('cpu.json', {})
        row = B.ref(self.root, 'cpu.json')
        with self.assertRaisesRegex(RuntimeError, 'duplicate source'): B.checked_rows(self.root, [row, row])

    def test_empty_source_closure_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'nonempty'): B.checked_rows(self.root, [])

    def test_shape_only_source_check_still_rejects_traversal(self):
        row = dict(path='../escape', bytes=0, sha256='0' * 64)
        with self.assertRaises(RuntimeError): B.checked_rows(self.root, [row], verify_bytes=False)

    def test_absolute_configuration_path_normalizes(self):
        path = self.put(B.CONFIG, {'origin': 'invalid_cpu_fixture'})
        self.assertEqual(B.config_relative(self.root, path), B.CONFIG)
        self.assertEqual(B.config_relative(self.root, str(path)), B.CONFIG)

    def test_relative_configuration_path_normalizes(self):
        self.put(B.CONFIG, {})
        self.assertEqual(B.config_relative(self.root, B.CONFIG), B.CONFIG)

    def test_relative_configuration_path_object_normalizes(self):
        self.put(B.CONFIG, {'origin': 'invalid_cpu_fixture'})
        self.assertEqual(B.config_relative(self.root, Path(B.CONFIG)), B.CONFIG)

    def parsed_config(self, value):
        # This is the actual shape emitted by the native launcher's parser.
        # Only path normalization is exercised; no valid GPU config is issued.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--execute', action='store_true')
        parser.add_argument('--config', required=True, type=Path)
        return parser.parse_args(['--execute', '--config', value]).config

    def test_actual_argparse_relative_config_Path_normalizes(self):
        self.put(B.CONFIG, {'origin': 'invalid_cpu_fixture'})
        value = self.parsed_config(B.CONFIG)
        self.assertIsInstance(value, Path)
        self.assertFalse(value.is_absolute())
        self.assertEqual(B.config_relative(self.root, value), B.CONFIG)

    def test_actual_argparse_absolute_contained_config_Path_normalizes(self):
        actual = self.put(B.CONFIG, {'origin': 'invalid_cpu_fixture'})
        self.assertEqual(B.config_relative(self.root, self.parsed_config(str(actual))), B.CONFIG)

    def test_actual_argparse_outside_absolute_Path_rejected(self):
        value = self.parsed_config(str(self.root.parent / 'NATIVE_COST_CONFIG.json'))
        with self.assertRaisesRegex(RuntimeError, 'outside'): B.config_relative(self.root, value)

    def test_actual_argparse_relative_Path_traversal_rejected(self):
        value = self.parsed_config('../' + B.CONFIG)
        with self.assertRaises(RuntimeError): B.config_relative(self.root, value)

    def test_actual_argparse_relative_Path_symlink_rejected(self):
        target = self.put('cpu-invalid-config.json', {'origin': 'invalid_cpu_fixture'})
        link = self.root / B.CONFIG
        link.parent.mkdir(parents=True, exist_ok=True)
        try: link.symlink_to(target)
        except OSError:
            original = Path.is_symlink
            def is_config_symlink(path):
                return path == link or original(path)
            with patch.object(Path, 'is_symlink', is_config_symlink):
                with self.assertRaisesRegex(RuntimeError, 'symlink'):
                    B.config_relative(self.root, self.parsed_config(B.CONFIG))
            return
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            B.config_relative(self.root, self.parsed_config(B.CONFIG))

    def test_actual_argparse_absolute_Path_symlink_rejected(self):
        target = self.put('cpu-invalid-config.json', {'origin': 'invalid_cpu_fixture'})
        link = self.root / B.CONFIG
        link.parent.mkdir(parents=True, exist_ok=True)
        try: link.symlink_to(target)
        except OSError:
            original = Path.is_symlink
            def is_config_symlink(path):
                return path == link or original(path)
            with patch.object(Path, 'is_symlink', is_config_symlink):
                with self.assertRaisesRegex(RuntimeError, 'symlink'):
                    B.config_relative(self.root, self.parsed_config(str(link)))
            return
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            B.config_relative(self.root, self.parsed_config(str(link)))

    def test_old_configuration_path_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'only the new'): B.config_relative(self.root, 'old/NATIVE_COST_CONFIG.json')

    def test_configuration_outside_root_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'outside'): B.config_relative(self.root, self.root.parent / 'config.json')

    def test_unbound_binding_rejects_before_any_probe(self):
        self.put(B.BINDING, dict(schema='c5_gpu_entry_site_binding_v1', status='UNBOUND'))
        with patch.object(B, 'probe_resources', side_effect=AssertionError('unexpected resource/GPU probe')):
            with self.assertRaisesRegex(RuntimeError, 'UNBOUND'): B.load_binding(self.root, B.BINDING)

    def test_synthetic_origin_cannot_be_live_binding(self):
        self.put(B.BINDING, dict(schema='c5_gpu_entry_site_binding_v1', status='LIVE_HUMAN_BOUND',
                               origin='synthetic_cpu', synthetic=True))
        with self.assertRaisesRegex(RuntimeError, 'synthetic'): B.load_binding(self.root, B.BINDING)

    def test_wrong_binding_path_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'wrong revision'): B.load_binding(self.root, 'other/GPU_ENTRY_BINDING.json')

    def test_old_root_or_label_cannot_be_live_binding(self):
        self.put(B.BINDING, dict(schema='c5_gpu_entry_site_binding_v1', status='LIVE_HUMAN_BOUND',
            origin='live_site_binding', synthetic=False, root=str(self.root), label='old-gpu-job'))
        with self.assertRaisesRegex(RuntimeError, 'root/label'): B.load_binding(self.root, B.BINDING)

    def test_half_core_resource_rejected(self):
        value = self.resource(); value['cpu_cores'] = 0.5
        with self.assertRaisesRegex(RuntimeError, 'CPU cores'): B._resource_shape(value)

    def test_unknown_quota_resource_rejected(self):
        value = self.resource(); value['cpu_counters_readable'] = False
        with self.assertRaisesRegex(RuntimeError, 'unknown'): B._resource_shape(value)

    def test_cpu_fixture_origin_resource_rejected(self):
        value = self.resource(); value['origin'] = 'synthetic_cpu'
        with self.assertRaisesRegex(RuntimeError, 'observation unavailable'): B._resource_shape(value)

    def test_no_gpu_resource_rejected(self):
        value = self.resource(); value['nvidia_nodes_visible'] = False
        with self.assertRaisesRegex(RuntimeError, 'observation unavailable'): B._resource_shape(value)

    def test_wrong_uuid_resource_rejected(self):
        value = self.resource(); value['gpu_uuid'] = 'GPU-OLD'
        with self.assertRaisesRegex(RuntimeError, 'UUID'): B._resource_shape(value)

    def test_old_driver_site_resource_rejected(self):
        value = self.resource(); value['driver_version'] = '595.58.03'
        with self.assertRaisesRegex(RuntimeError, 'driver'): B._resource_shape(value)

    def test_missing_or_untyped_driver_resource_rejected(self):
        for driver in (None, 580.95, True, ''):
            value = self.resource(); value['driver_version'] = driver
            with self.subTest(driver=driver), self.assertRaisesRegex(RuntimeError, 'driver'):
                B._resource_shape(value)

    def test_new_job_paths_keep_original_shared_sources(self):
        self.assertEqual(B.DIR, 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2')
        self.assertEqual(B.LABEL, 'server12-i-pilot-cal01')
        self.assertEqual(B.COMMON, 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate')
        self.assertEqual(len(B.SDK_ASSETS), 6)
        self.assertNotEqual(B.DIR, B.COMMON.removesuffix('/common_candidate'))

    def test_previous_gpu_configuration_path_rejected(self):
        old = B.COMMON.removesuffix('/common_candidate') + '/NATIVE_COST_CONFIG.json'
        with self.assertRaisesRegex(RuntimeError, 'only the new'):
            B.config_relative(self.root, old)

    def test_old_context_uuid_rejected_before_new_grant_or_permission(self):
        context = dict(origin='live_site_readonly_context', root=str(self.root), label=B.LABEL,
            resources=self.resource())
        context['resources']['gpu_uuid'] = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'
        self.put(B.CONTEXT, context)
        with patch.object(B, 'probe_resources', return_value=self.resource()), \
                patch.object(C, 'put', side_effect=AssertionError('authorization must not be emitted')):
            with self.assertRaisesRegex(RuntimeError, 'live context target mismatch'):
                C.bind(self.root)
        self.assertFalse(B.safe(self.root, B.GRANT).exists())
        self.assertFalse(B.safe(self.root, B.PERMISSION).exists())

    def test_missing_standing_grant_blocks_bind_before_permission(self):
        context = dict(origin='live_site_readonly_context', root=str(self.root), label=B.LABEL,
            resources=self.resource(), revision_lock_ref={'origin': 'invalid_cpu_fixture'})
        self.put(B.CONTEXT, context)
        with patch.object(B, 'probe_resources', return_value=self.resource()), \
                patch.object(B, '_revision', return_value={}), \
                patch.object(C, 'put', side_effect=AssertionError('authorization must not be emitted')):
            with self.assertRaisesRegex(ValueError, 'STANDING_AUTHORIZATION_REJECTED'):
                C.bind(self.root)
        self.assertFalse(B.safe(self.root, B.GRANT).exists())
        self.assertFalse(B.safe(self.root, B.PERMISSION).exists())

    def test_existing_compute_resource_rejected(self):
        value = self.resource(); value['compute_processes'] = [123]
        with self.assertRaisesRegex(RuntimeError, 'observation unavailable'): B._resource_shape(value)

    def test_vram_memory_storage_resource_floors(self):
        for key, value in [('gpu_free_mib', 27999), ('available_memory_bytes', 32 * 1024**3 - 1),
                           ('primary_free_bytes', 10 * 1024**3 - 1)]:
            data = self.resource(); data[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'invalid number'): B._resource_shape(data)

    def test_boolean_and_nonfinite_resource_numbers_rejected(self):
        for key, value in [('cpu_cores', True), ('cpu_affinity_count', False), ('gpu_free_mib', math.inf)]:
            data = self.resource(); data[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'invalid number'): B._resource_shape(data)

    def test_resource_numeric_boundary_returns_only_uuid(self):
        self.assertEqual(B._resource_shape(self.resource()), self.resource()['gpu_uuid'])

    def test_prepare_writes_only_unbound_review_template(self):
        output = 'cpu-output/UNBOUND_REVIEW_TEMPLATE.json'
        with patch.object(subprocess_target(), 'Popen', side_effect=AssertionError('GPU launch forbidden')):
            result = C.prepare(self.root, output)
        self.assertFalse(result['gpu_launch_allowed']); self.assertEqual(result['actual_gpu_runs'], 0)
        value = B.read(self.root, output)
        for key in ('gpu_uuid', 'effective_permission', 'valid_config', 'site_binding', 'human_grant', 'native_receipt'):
            self.assertIsNone(value[key])
        for path in (B.CONFIG, B.BINDING, B.GRANT, B.PERMISSION, B.SITE_LOCK, B.REVISION_LOCK, B.RUN):
            self.assertFalse(B.safe(self.root, path).exists())

    def test_prepare_cannot_write_effective_configuration(self):
        with self.assertRaisesRegex(RuntimeError, 'UNBOUND review'): C.prepare(self.root, B.CONFIG)
        self.assertFalse(B.safe(self.root, B.CONFIG).exists())

    def test_no_card_context_rejects_before_freeze(self):
        limited = dict(origin='live_linux_readonly_resource_probe', status='RESOURCE_LIMITED')
        with patch.object(B, 'probe_resources', return_value=limited), patch.object(C, 'freeze_revision', side_effect=AssertionError('unexpected freeze')):
            with self.assertRaisesRegex(RuntimeError, 'observation unavailable'): C.context(self.root)
        self.assertFalse(B.safe(self.root, B.CONTEXT).exists())

    def test_missing_config_launch_rejects_before_popen(self):
        with patch.object(C.subprocess, 'Popen', side_effect=AssertionError('GPU launch forbidden')):
            with self.assertRaises(FileNotFoundError): C.launch(self.root)

    def test_active_and_completed_require_real_configuration_first(self):
        for call in (B.verify_active_guard, B.verify_completed_guard):
            with self.subTest(call=call.__name__), self.assertRaises(FileNotFoundError): call(self.root, {}, {})

    def test_exact_bool_is_not_integer_success(self):
        self.assertFalse(B.exact(False, 0)); self.assertFalse(B.exact(1, True))
        self.assertTrue(B.exact(0, 0)); self.assertTrue(B.exact(True, True))


def subprocess_target():
    return C.subprocess


if __name__ == '__main__':
    program = unittest.main(exit=False)
    loaded = sorted(name for name in sys.modules if name.split('.', 1)[0] in
        ('torch', 'vllm', 'py_kvcache', 'prefix_io_control', 'pynvml'))
    success = program.result.wasSuccessful() and not loaded and not FORBIDDEN_IMPORTS
    print(json.dumps(dict(scope='CPU_SERVER12_BINDING_IMPORT_AUDIT', status='PASS' if success else 'FAILED',
        tests=program.result.testsRun, failures=len(program.result.failures), errors=len(program.result.errors),
        skipped=len(program.result.skipped), forbidden_import_attempts=FORBIDDEN_IMPORTS,
        loaded_forbidden_modules=loaded, actual_gpu_runs=0)))
    raise SystemExit(0 if success else 1)
