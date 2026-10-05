"""Targeted CPU-only checks for the new physical-GPU calibration domain.

These checks issue no receipt, consume no GPU authorization, and import no
model/backend. Old raw costs remain history, never new-domain measurements.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
D = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
G = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
JOB = 'server12-c5-native-common-cost-gpu01'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
ATTEMPTS = []


def import_guard(event, args):
    if event == 'import' and args and type(args[0]) is str and any(
        args[0] == prefix or args[0].startswith(prefix + '.') for prefix in FORBIDDEN):
        ATTEMPTS.append(args[0])
        raise ImportError('CPU domain checks prohibit model/backend imports')


sys.addaudithook(import_guard)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


N = module('_server12_cpu_domain_native', 'native_conditional_cost.py')
S = module('_server12_cpu_domain_serializer', 'prepare_and_verify_native_cost.py')
Q = module('_server12_cpu_domain_public', 'p4_single_file_receipt.py')


def ancestor():
    physical = Path('/root/autodl-tmp/prefix-io-v1-handoff/project') / G
    if physical.is_dir():
        return physical
    return HERE.parents[1] / 'prefix_io_v1_server11_candidates' / 'notification_v5_gpu_entry_path_revision'


def functions(path):
    return {node.name: node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


class DomainTests(unittest.TestCase):
    def test_new_job_and_site_with_original_common(self):
        self.assertEqual(N.ENTRY, D)
        self.assertEqual(N.ENTRY_LABEL, JOB)
        self.assertEqual(N.ENTRY_COMMON, G + '/common_candidate')
        self.assertEqual(N.ENTRY_OVERLAY, G + '/common_candidate/source')
        self.assertEqual(S.C.ENTRY, D)
        self.assertEqual(S.C.ENTRY_LABEL, JOB)

    def test_native_13_non_entry_functions_unchanged(self):
        old = functions(ancestor() / 'native_conditional_cost.py')
        new = functions(HERE / 'native_conditional_cost.py')
        names = set(old) - {'binding_api', 'validate_native_plan', 'validate_prelaunch_plan',
            'verify_native_cell', 'main'}
        self.assertEqual(len(names), 13)
        self.assertEqual(set(old), set(new))
        for name in names:
            self.assertEqual(dump(old[name]), dump(new[name]), name)
        self.assertEqual([name for name in old if dump(old[name]) != dump(new[name])], ['validate_native_plan'])

    def test_native_metadata_delta_is_only_common_path_resolution(self):
        old = functions(ancestor() / 'native_conditional_cost.py')['validate_native_plan']
        new = deepcopy(functions(HERE / 'native_conditional_cost.py')['validate_native_plan'])
        class OriginalCommonPath(ast.NodeTransformer):
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                if isinstance(node.left, ast.Name) and node.left.id == 'ENTRY_COMMON' and isinstance(node.right, ast.Constant):
                    node.left.id = 'ENTRY'
                    node.right.value = '/common_candidate' + node.right.value
                return node
        self.assertEqual(dump(old), dump(OriginalCommonPath().visit(new)))

    def test_serializer_byte_exact_with_original_frame_and_measurement_rules(self):
        before = (ancestor() / 'prepare_and_verify_native_cost.py').read_bytes()
        after = (HERE / 'prepare_and_verify_native_cost.py').read_bytes()
        self.assertEqual(after, before)
        self.assertEqual((len(after), sha256(after).hexdigest()),
            (24942, 'cc777713612ebebf361f4c476440317e97610a0f8a0553cb3d7d33e26ea3ea64'))

    def test_public_native_serializer_and_original_math_pins(self):
        for reference, filename in ((Q._VERIFIER, 'native_conditional_cost.py'),
            (Q._SERIALIZER, 'prepare_and_verify_native_cost.py')):
            raw = (HERE / filename).read_bytes()
            self.assertEqual(reference.path, D + '/' + filename)
            self.assertEqual((reference.bytes, reference.sha256), (len(raw), sha256(raw).hexdigest()))
        self.assertEqual(Q._MATH.path,
            'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py')
        self.assertEqual(Q._MATH.sha256, '3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac')

    def test_public_original_model_reactor_collector_helper_and_math_ast(self):
        self.assertEqual(Q._COMMON_OVERLAY, G + '/common_candidate/source')
        self.assertEqual(Q._NATIVE, G + '/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
        old = functions(ancestor() / 'common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py')
        new = functions(HERE / 'p4_single_file_receipt.py')
        self.assertEqual(set(old), set(new))
        self.assertEqual([name for name in old if dump(old[name]) != dump(new[name])], ['load_verified_single_file'])
        class OriginalJob(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == JOB:
                    node.value = 'server11-c5-native-common-cost-gpu03'
                return node
        self.assertEqual(dump(old['load_verified_single_file']),
            dump(OriginalJob().visit(deepcopy(new['load_verified_single_file']))))

    def test_old_gpu03_plan_rejected_before_new_site_or_model(self):
        # Explicit rejection input, never a positive native fixture or receipt.
        with self.assertRaisesRegex(ValueError, 'old native plans'):
            N.validate_native_plan(None, dict(evidence_origin='native_runtime_preregistered',
                cpu_preparation_only=False, job_id='server11-c5-native-common-cost-gpu03',
                test_scope='CPU_rejection_only'))

    def test_cpu_plan_cannot_claim_new_native_execution(self):
        with self.assertRaisesRegex(ValueError, 'preparation'):
            N.validate_native_plan(None, dict(evidence_origin='CPU_fixture_only',
                cpu_preparation_only=True, job_id=JOB))

    def test_non_exact_origin_rejected(self):
        with self.assertRaises(ValueError):
            N.validate_native_plan(None, dict(evidence_origin='native_gpu_recording',
                cpu_preparation_only=False, job_id=JOB))

    def test_unbound_missing_site_rejects_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            # This CPU rejection input only exercises source/config lookup;
            # there is no live binding, grant, scope, event or model process.
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
                N.validate_native_plan(root, dict(evidence_origin='native_runtime_preregistered',
                    cpu_preparation_only=False, job_id=JOB, project_root=root.as_posix(),
                    test_scope='CPU_missing_site_rejection_only'))

    def test_no_private_receipt_construction_or_old_cost_override(self):
        with self.assertRaisesRegex(ValueError, 'load_verified_single_file'):
            Q.ExactSingleFileReceipt()
        for filename in ('native_conditional_cost.py', 'prepare_and_verify_native_cost.py', 'p4_single_file_receipt.py'):
            source = (HERE / filename).read_text(encoding='utf-8')
            self.assertNotIn('20456415', source)
            self.assertNotIn('12957824', source)

    def test_hardware_signature_and_unknown_condition_boundary_unchanged(self):
        node = functions(HERE / 'p4_single_file_receipt.py')['load_verified_single_file']
        source = ast.unparse(node)
        self.assertIn("condition['gpu_uuid']", source)
        self.assertIn("condition['model_sha256']", source)
        self.assertIn("condition['kv_layout_sha256']", source)
        paired = ast.unparse(functions(HERE / 'native_conditional_cost.py')['analyze_paired'])
        for value in ('zero_observed_holdout_underprediction_no_refit_v1', 'calibration', 'validation',
            'AB', 'BA', 'unknown_conditions', 'reject', 'holdout_used_to_refit'):
            self.assertIn(value, paired)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DomainTests))
    loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p+'.') for p in FORBIDDEN))
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() and not ATTEMPTS and not loaded else 'FAILED',
        scope='CPU_SERVER12_DOMAIN_ONLY', tests=result.testsRun, failed=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), forbidden_import_attempts=ATTEMPTS,
        loaded_forbidden_modules=loaded, GPU_operations=0, actual_model_processes_started=0,
        native_execution_verified=False, conditional_cost_cell_qualified=False,
        production_qualified=False, valid_native_receipt=None, effective_cost_upper_ns=None,
        effective_step_budget_ns=None, normal_runtime_qualified=False), sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and not ATTEMPTS and not loaded else 1)
