"""CPU-only boundaries for the explicit existing 580-driver SDK selection.

Synthetic receipt metadata is used only to exercise rejecting boundaries.
The actual unchanged helper is imported for its pure environment function;
no asset qualification, overlay creation, compilation, model or GPU is run.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_cpu_site_sdk_binding', HERE / 'site_sdk_binding.py')
S = importlib.util.module_from_spec(spec); sys.modules[spec.name] = S; spec.loader.exec_module(S)
spec = importlib.util.spec_from_file_location('_cpu_server12_native_runner', HERE / 'run_native_cost_experiment.py')
R = importlib.util.module_from_spec(spec); sys.modules[spec.name] = R; spec.loader.exec_module(R)


class MetadataRejections(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.rows = {relative: dict(path=relative, bytes=size, sha256=digest)
                     for relative, (size, digest) in S.PINS.items()}
        self.inventory = dict(driver=dict(S.DRIVER), toolkit_root=str(self.root / '.venv/lib/python3.12/site-packages/nvidia/cu13'))
        self.proof = dict(driver_ref=dict(S.DRIVER), manifest_ref=S.absolute_ref(self.root, self.rows[S.INVENTORY]),
            output_refs=[dict(path=str(self.root/S.SITE/name), bytes=1, sha256='a'*64)
                         for name in ('probe.cu', 'probe.o', 'probe.so')])
        self.result = dict(status='PASS_SERVER10_CPU_SDK_REBIND_EXISTING_HELPER_VALIDATED',
            inventory_ref=S.absolute_ref(self.root, self.rows[S.INVENTORY]),
            compiler_proof_ref=S.absolute_ref(self.root, self.rows[S.PROOF]), driver_ref=dict(S.DRIVER),
            source_files=1934, source_bytes=217111529, compiler_executions=3, binutils_executions=1,
            GPU_operations=0, shared_object_loaded=False, old_assets_modified=False,
            SDK_trees_copied=False, GPU_qualification=False)

    def reject(self, pattern=None):
        context = self.assertRaisesRegex(ValueError, pattern) if pattern else self.assertRaises(ValueError)
        with context:
            S.validate_receipts(self.root, self.rows, self.inventory, self.proof, self.result)

    def write(self, relative, raw):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        return S.source_ref(self.root, relative)

    def test_old_595_driver_receipt_rejected(self):
        self.inventory['driver']['path'] = '/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05'
        self.reject('580')

    def test_other_host_driver_sha_rejected(self):
        self.proof['driver_ref']['sha256'] = 'a'*64; self.reject('580')

    def test_proof_not_bound_to_exact_580_inventory_rejected(self):
        self.proof['manifest_ref']['sha256'] = 'a'*64; self.reject('580')

    def test_boolean_gpu_operation_count_rejected(self):
        self.result['GPU_operations'] = False; self.reject('semantics')

    def test_shared_library_loading_claim_rejected(self):
        self.result['shared_object_loaded'] = True; self.reject('semantics')

    def test_gpu_qualification_claim_rejected(self):
        self.result['GPU_qualification'] = True; self.reject('semantics')

    def test_sdk_copying_or_old_asset_mutation_rejected(self):
        for field in ('SDK_trees_copied', 'old_assets_modified'):
            with self.subTest(field=field):
                self.result[field] = True; self.reject('semantics'); self.result[field] = False

    def test_wrong_original_compiler_command_count_rejected(self):
        self.result['compiler_executions'] = 0; self.reject('semantics')

    def test_foreign_cuda_toolkit_root_rejected(self):
        self.inventory['toolkit_root'] = '/usr/local/cuda'; self.reject('toolkit')

    def test_probe_outside_original_project_rejected(self):
        self.proof['output_refs'][0]['path'] = str(self.root.parent/'probe.cu'); self.reject('outside')

    def test_probe_duplicate_cannot_replace_shared_object(self):
        self.proof['output_refs'][2] = deepcopy(self.proof['output_refs'][0]); self.reject('distinct')

    def test_probe_other_directory_rejected(self):
        self.proof['output_refs'][0]['path'] = str(self.root/'cpu_fixture-other/probe.cu'); self.reject('bounded CPU probe')

    def test_receipt_bytes_changed_even_when_lock_row_changed_rejected(self):
        row = self.write(S.PROOF, b'{"origin":"cpu_fixture"}\n')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            S.checked(self.root, {S.PROOF:row}, S.PROOF)

    def test_same_size_source_drift_rejected(self):
        row = self.write('cpu_fixture/source.py', b'abcd')
        (self.root/row['path']).write_bytes(b'abce')
        with self.assertRaisesRegex(ValueError, 'drift'):
            S.checked(self.root, {row['path']:row}, row['path'])

    def test_reference_boolean_size_rejected(self):
        row = self.write('cpu_fixture/source.py', b'a'); row['bytes'] = True
        with self.assertRaisesRegex(ValueError, 'immutable'):
            S.checked(self.root, {row['path']:row}, row['path'])

    def test_source_reference_absent_from_lock_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            S.checked(self.root, {}, S.HELPER)

    def test_duplicate_receipt_json_rejected(self):
        row = self.write('cpu_fixture/duplicate.json', b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            S.read(self.root, row['path'])

    def test_nonfinite_receipt_json_rejected(self):
        row = self.write('cpu_fixture/nonfinite.json', b'{"x":NaN}')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            S.read(self.root, row['path'])

    def test_source_path_traversal_rejected(self):
        for relative in ('../receipt.json', '/etc/passwd', 'a\\b', 'a/../b', 'a//b'):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError): S.safe(self.root, relative)

    def test_no_asset_gate_pass_does_not_mutate_environment_or_create_overlay(self):
        out = self.root/'experiments/prefix_io_v1/runs/cpu-fixture/details'; out.mkdir(parents=True)
        self.write(S.SOURCE, b'# explicit cpu_fixture source\n')
        rows = {S.SOURCE:S.source_ref(self.root, S.SOURCE)}
        before = dict(os.environ)
        with self.assertRaises(ValueError): S.prepare_site_sdk(self.root, rows, out)
        self.assertEqual(dict(os.environ), before)
        self.assertFalse((out/'runtime-cache/cuda13-sdk').exists())

    def test_direct_runner_call_rejects_before_site_or_gpu_import(self):
        with patch.object(R, 'verify_execution_inputs', side_effect=ValueError('cpu_fixture missing site authority')):
            with self.assertRaisesRegex(ValueError, 'authority'):
                R.execute_window(self.root, {}, {}, {}, 0)

    def test_argparse_relative_configuration_reaches_site_gate_as_path(self):
        observed = []
        def reject(path):
            observed.append(path)
            raise ValueError('cpu_fixture missing site authority')
        with patch.object(R, 'load_configuration', side_effect=reject):
            with self.assertRaisesRegex(ValueError, 'authority'):
                R.main(['--execute', '--config', S.ENTRY+'/NATIVE_COST_CONFIG.json'])
        self.assertEqual(observed, [Path(S.ENTRY+'/NATIVE_COST_CONFIG.json')])


class OriginalSourceAndEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = HERE.parents[2]
        cls.helper_path = cls.root/S.HELPER
        if not cls.helper_path.is_file():
            cls.helper_path = cls.root/'artifacts/prefix_io_v1_server09_candidates/g2_cuda13_toolchain/cuda13_sdk_overlay.py'
        raw = cls.helper_path.read_bytes()
        if len(raw) != S.PINS[S.HELPER][0] or hashlib.sha256(raw).hexdigest() != S.PINS[S.HELPER][1]:
            raise AssertionError('unchanged actual SDK helper source not present')
        spec = importlib.util.spec_from_file_location('_cpu_actual_unchanged_sdk_helper', cls.helper_path)
        cls.helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = cls.helper; spec.loader.exec_module(cls.helper)

    def test_actual_original_environment_function_keeps_stubs_off_runtime_path(self):
        overlay = Path('/cpu-fixture/guarded/details/runtime-cache/cuda13-sdk')
        source = dict(PATH='/usr/bin', LD_LIBRARY_PATH='/usr/lib', CUDA_VISIBLE_DEVICES='')
        result = self.helper.planned_environment(overlay, source)
        self.assertEqual(result['LD_LIBRARY_PATH'], str(overlay/'lib64') + os.pathsep + '/usr/lib')
        self.assertNotIn('stubs', result['LD_LIBRARY_PATH'])
        self.assertEqual(source, dict(PATH='/usr/bin', LD_LIBRARY_PATH='/usr/lib', CUDA_VISIBLE_DEVICES=''))

    def test_actual_helper_refuses_inherited_arch_and_jit_bypass(self):
        for key in ('FLASHINFER_CUDA_ARCH_LIST', 'TORCH_CUDA_ARCH_LIST', 'FLASHINFER_DISABLE_JIT'):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.helper.planned_environment(Path('/cpu-fixture/sdk'), {key:'1'})

    def test_actual_helper_refuses_stubs_in_runtime_path(self):
        with self.assertRaises(ValueError):
            self.helper.planned_environment(Path('/cpu-fixture/sdk'), {'LD_LIBRARY_PATH':'/cpu-fixture/lib64/stubs'})

    def test_actual_helper_refuses_foreign_compiler_root(self):
        with self.assertRaises(ValueError):
            self.helper.planned_environment(Path('/cpu-fixture/sdk'), {'CUDA_HOME':'/usr/local/cuda'})

    def test_engine_lifecycle_helpers_remain_exact_g_ast(self):
        old = self.root/'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/run_native_cost_experiment.py'
        if not old.is_file():
            old = HERE.parents[1]/'prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/run_native_cost_experiment.py'
        old_funcs = {n.name:n for n in ast.parse(old.read_bytes()).body if isinstance(n, ast.FunctionDef)}
        new_funcs = {n.name:n for n in ast.parse((HERE/'run_native_cost_experiment.py').read_bytes()).body if isinstance(n, ast.FunctionDef)}
        # The wrapper alone changes identity/SDK metadata selection. Its
        # admission, original execution, native I/O and drains do not change.
        for name in sorted(set(old_funcs)-{'execute_window','load_configuration'}):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old_funcs[name], include_attributes=False), ast.dump(new_funcs[name], include_attributes=False))
        modified = deepcopy(new_funcs['execute_window'])
        for node in ast.walk(modified):
            if not isinstance(node, ast.Try): continue
            body = node.body
            start = next((i for i,n in enumerate(body) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'site_sdk' for t in n.targets)), None)
            if start is not None:
                body.pop(start)
                sdk_assign = body[start]
                sdk_assign.value = ast.parse('common.prepare_process_sdk(root,refs,out)', mode='eval').body
        self.assertEqual(ast.dump(old_funcs['execute_window'], include_attributes=False), ast.dump(modified, include_attributes=False))

    def test_all_native_g_common_paths_stay_original(self):
        self.assertEqual(R.CANDIDATE, 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate')
        self.assertEqual(R.OVERLAY, R.CANDIDATE+'/source')
        self.assertEqual(R.REACTOR, R.OVERLAY+'/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
        self.assertEqual(R.ORDER, (('A','B'), ('B','A'), ('A','B')))
        self.assertEqual(R.LABEL, 'server12-c5-native-common-cost-gpu01')

    def test_no_backend_imports_by_adapter_runner_or_test(self):
        self.assertFalse(any(name.split('.')[0] in ('torch','vllm','py_kvcache') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
