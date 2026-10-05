"""CPU-only path/authority/resource rejects; no native-positive receipt fixture."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_entry_binding as B
import control_native_cost_job as C


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
            cpu_affinity_count=1, cpu_counters_readable=True,
            available_memory_bytes=32 * 1024**3, primary_free_bytes=10 * 1024**3)

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
    unittest.main()
