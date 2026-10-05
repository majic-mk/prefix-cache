"""Independent CPU authorization/parameter and calibration inheritance review.

Only new metadata modules are imported. GPU/model/native/backend/SSH imports
and subprocess execution are rejected. Fixtures below are declared CPU values,
never persisted as live context, a grant, a reservation or a GPU result.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
NEW = PARENT / 'calibration_v2'
ROOT = PARENT.parents[2]
OLD = ROOT / 'artifacts/prefix_io_v1_server12_candidates/native_recalibration_cpu'
if not OLD.is_dir():
    OLD = ROOT / 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'prefix_io_control', 'pynvml', 'cupy', 'ctypes', 'paramiko')
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_rows():
    return [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
        for p in sorted(NEW.iterdir()) if p.is_file()]


BEFORE = source_rows()


class RejectImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('independent CPU calibration review rejects ' + fullname)


sys.meta_path.insert(0, RejectImports())


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, NEW / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = load('standing_gpu_authorization', 'standing_gpu_authorization.py')
B = load('gpu_entry_binding', 'gpu_entry_binding.py')
C = load('_calibration_review_controller', 'control_native_cost_job.py')


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_bytes()).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def constants(path):
    values = {}
    def value(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return values[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            result = [value(n) for n in node.elts]
            return tuple(result) if isinstance(node, ast.Tuple) else result
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add): return value(node.left) + value(node.right)
        raise ValueError('not literal metadata')
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: values[node.targets[0].id] = value(node.value)
            except (KeyError, ValueError, TypeError): pass
    return values


class StandingAuthorizationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='standing-authorization-cpu-review-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base_ref = dict(path=A.BASE_PERMISSION, bytes=1, sha256='0'*64)
        self.reference = dict(path=A.AMENDMENT, bytes=1114, sha256=A.AMENDMENT_SHA)
        self.document = dict(schema='c5_project_persistent_gpu_authorization_amendment_v1',
            root=str(self.root), issuer='human_user', authorization=True, human_instruction=A.INSTRUCTION,
            original_max_gpu_seconds=28800, original_budget_unchanged=True,
            source='direct_current_user_instruction', base_permission_ref=self.base_ref,
            GPU_authority_scope='current_frozen_project_experiments_with_existing_stage_gates',
            future_gpu_approval_questions_required=False)
        self.revision = dict(path=B.REVISION_LOCK, bytes=1, sha256='1'*64)
        self.context = dict(schema='c5_gpu_entry_live_context_v1', origin='live_site_readonly_context',
            root=str(self.root), label=B.LABEL, revision_lock_ref=self.revision,
            seconds_limit=1200, reserved_seconds=1220, max_attempts=1, max_processes=6,
            resources=dict(gpu_uuid=UUID))

    def bind_fixture(self, context=None, **kwargs):
        # This fixture supplies only previously verified metadata values, and
        # never writes a grant/context or calls real resource telemetry.
        with patch.object(A, 'standing_grant', return_value=(self.document, self.reference)):
            return A.bind_standing_grant(self.root, self.context if context is None else context,
                label=kwargs.get('label', B.LABEL), gpu_uuid=kwargs.get('gpu_uuid', UUID),
                revision_ref=kwargs.get('revision_ref', self.revision))

    def test_direct_existing_user_instruction_and_original_budget_required(self):
        self.assertIs(A.validate_amendment(self.document, self.root, self.base_ref), self.document)
        for key, bad in [('issuer', 'agent'), ('authorization', 1), ('human_instruction', '继续'),
            ('original_max_gpu_seconds', 28801), ('original_max_gpu_seconds', 28800.0),
            ('original_budget_unchanged', False), ('source', 'inferred_continue'),
            ('GPU_authority_scope', 'arbitrary_new_experiments'),
            ('future_gpu_approval_questions_required', True), ('root', str(self.root / 'foreign'))]:
            value = dict(self.document); value[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                A.validate_amendment(value, self.root, self.base_ref)

    def test_base_permission_reference_and_root_are_not_inherited_from_another_site(self):
        for base in ({**self.base_ref, 'sha256': 'f'*64}, {**self.base_ref, 'bytes': 2}):
            with self.assertRaises(ValueError): A.validate_amendment(self.document, self.root, base)
        with self.assertRaises(ValueError):
            A.validate_amendment(self.document, self.root / 'other', self.base_ref)

    def test_metadata_reference_bounds_and_drift_are_rejected_before_parse(self):
        p = self.root / 'value.json'; p.write_bytes(b'{}')
        raw, ref = A.read_reference(self.root, 'value.json', sha(b'{}'))
        self.assertEqual(raw, b'{}')
        self.assertEqual(ref['bytes'], 2)
        for path in ('../value.json', './value.json', 'a//b', 'a\\b', '/etc/passwd', 'C:/outside', ''):
            with self.subTest(path=path), self.assertRaises(ValueError):
                A.read_reference(self.root, path, sha(b'{}'))
        with self.assertRaises(ValueError): A.read_reference(self.root, 'value.json', 'f'*64)
        p.write_bytes(b'x'*(1024**2+1))
        with self.assertRaises(ValueError): A.read_reference(self.root, 'value.json', sha(p.read_bytes()))

    def test_symlink_reference_rejected_without_following_target(self):
        p = self.root / 'value.json'; p.write_bytes(b'{}')
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                A.read_reference(self.root, 'value.json', sha(b'{}'))

    def test_duplicate_and_nonfinite_authorization_json_rejected(self):
        for raw in (b'{"authorization":true,"authorization":false}', b'{"budget":NaN}'):
            with patch.object(A, 'read_reference', side_effect=[(raw, self.reference), (b'{}', self.base_ref)]):
                with self.assertRaises(ValueError): A.standing_grant(self.root)

    def test_binding_is_derived_from_standing_grant_and_keeps_fixed_limits(self):
        grant = self.bind_fixture()
        self.assertEqual(grant['standing_authorization_ref'], self.reference)
        self.assertEqual(grant['binding_origin'], 'derived_from_existing_human_standing_authorization')
        self.assertIs(grant['per_round_reapproval_required'], False)
        for key, expected in [('max_attempts', 1), ('max_processes', 6), ('seconds_limit', 1200), ('reserved_seconds', 1220)]:
            self.assertIs(type(grant[key]), int)
            self.assertEqual(grant[key], expected)
        self.assertIsNone(grant['context_ref'])

    def test_context_origin_scope_revision_uuid_and_limits_reject(self):
        changes = [('origin', 'cpu_fixture'), ('root', str(self.root / 'foreign')),
            ('label', 'foreign-job'), ('revision_lock_ref', {**self.revision, 'sha256':'f'*64}),
            ('seconds_limit', 1201), ('seconds_limit', -1), ('seconds_limit', True),
            ('reserved_seconds', 1221), ('resources', dict(gpu_uuid='GPU-other'))]
        for key, bad in changes:
            context = copy.deepcopy(self.context); context[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError): self.bind_fixture(context)
        with self.assertRaises(ValueError): self.bind_fixture(gpu_uuid=None)

    def test_bound_provenance_rejects_fresh_agent_or_continue_grant(self):
        grant = self.bind_fixture()
        with patch.object(A, 'standing_grant', return_value=(self.document, self.reference)):
            self.assertEqual(A.validate_bound_grant(self.root, grant), self.reference)
            for key, bad in [('standing_authorization_ref', {**self.reference, 'sha256':'f'*64}),
                ('binding_origin', 'new_agent_approval'), ('human_instruction', '继续'),
                ('per_round_reapproval_required', True)]:
                changed = dict(grant); changed[key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    A.validate_bound_grant(self.root, changed)

    def test_production_resource_gate_rejects_bad_uuid_missing_gpu_and_occupied_gpu(self):
        resource = dict(origin='live_linux_readonly_resource_probe', status='LIVE_RESOURCE_READY',
            nvidia_nodes_visible=True, gpu_uuid=UUID, driver_version=B.SITE_DRIVER_VERSION,
            compute_processes=[], cpu_cores=1, cpu_affinity_count=1,
            available_memory_bytes=32*1024**3, gpu_free_mib=28000,
            primary_free_bytes=10*1024**3, cpu_counters_readable=True)
        self.assertEqual(B._resource_shape(resource), UUID)
        for key, bad in [('gpu_uuid', ''), ('gpu_uuid', 'GPU-other'), ('nvidia_nodes_visible', False),
            ('compute_processes', [dict(pid=1)]), ('cpu_cores', .5), ('cpu_cores', True),
            ('available_memory_bytes', 2*1024**3), ('gpu_free_mib', 27999),
            ('primary_free_bytes', 8*1024**3), ('driver_version', 'wrong'), ('cpu_counters_readable', False)]:
            changed = dict(resource); changed[key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError): B._resource_shape(changed)

    def test_context_refuses_active_or_exhausted_original_budget_without_writing(self):
        for ledger in (dict(active_reservation={}, gpu_wall_seconds=0),
            dict(active_reservation=None, gpu_wall_seconds=28800-1219),
            dict(active_reservation=None, gpu_wall_seconds=True),
            dict(active_reservation=None, gpu_wall_seconds=float('nan'))):
            with (patch.object(B, 'probe_resources', return_value={}), patch.object(B, '_resource_shape', return_value=UUID),
                patch.object(B, 'read', return_value=ledger), patch.object(C, 'put', side_effect=AssertionError('must not persist live context'))):
                with self.assertRaises(RuntimeError): C.context(self.root)

    def test_missing_context_blocks_bind_before_resource_probe_or_any_write(self):
        with (patch.object(B, 'probe_resources', side_effect=AssertionError('must read context first')),
            patch.object(C, 'put', side_effect=AssertionError('must not write grant'))):
            with self.assertRaises(FileNotFoundError): C.bind(self.root)

    def bound_memory_fixture(self):
        """Validation-only values; no positive native context/grant is written."""
        def ref(path): return dict(path=path, bytes=1, sha256='a'*64)
        resource = dict(origin='live_linux_readonly_resource_probe', status='LIVE_RESOURCE_READY',
            nvidia_nodes_visible=True, gpu_uuid=UUID, driver_version=B.SITE_DRIVER_VERSION,
            compute_processes=[], cpu_cores=1, cpu_affinity_count=1,
            available_memory_bytes=32*1024**3, gpu_free_mib=28000,
            primary_free_bytes=10*1024**3, cpu_counters_readable=True)
        context = dict(self.context); context['resources'] = resource
        grant = self.bind_fixture(context); grant['context_ref'] = ref(B.CONTEXT)
        binding = dict(schema='c5_gpu_entry_site_binding_v1', status='LIVE_HUMAN_BOUND',
            origin='live_site_binding', synthetic=False, root=str(self.root), label=B.LABEL,
            gpu_uuid=UUID, revision_lock_ref=self.revision, context_ref=ref(B.CONTEXT),
            human_grant_ref=ref(B.GRANT), effective_permission_ref=ref(B.PERMISSION),
            site_lock_relative=B.SITE_LOCK, config_relative=B.CONFIG)
        base = dict(max_gpu_hours=8, approved_experiment_root=str(self.root/'experiments/prefix_io_v1/runs'),
            approved_dependency_root='CPU-FIXTURE-DEPENDENCY-ROOT')
        permission = dict(base, allow_gpu_runs=True, approved_gpu_ids=[UUID], approved_auxiliary_storage=None,
            gpu_entry_binding=dict(label=B.LABEL, revision_lock_ref=self.revision,
                context_ref=ref(B.CONTEXT), human_grant_ref=ref(B.GRANT),
                seconds_limit=1200, reserved_seconds=1220, max_attempts=1, max_processes=6))
        for key in B.DENIED: permission[key] = False
        return dict(binding=binding, context=context, grant=grant, base=base, permission=permission, ref=ref)

    def validate_memory_binding(self, values):
        docs = {B.BINDING:values['binding'], B.CONTEXT:values['context'], B.GRANT:values['grant']}
        def permission(root, path): return values['permission'] if path == B.PERMISSION else values['base']
        with (patch.object(B, 'read', side_effect=lambda root,path: docs[path]),
            patch.object(B, '_revision', return_value={B.BASE_PERMISSION:values['ref'](B.BASE_PERMISSION)}),
            patch.object(B, 'checked_ref', side_effect=lambda root,row: row),
            patch.object(B, 'ref', side_effect=lambda root,path: values['ref'](path)),
            patch.object(B, 'permission_document', side_effect=permission),
            patch.object(A, 'standing_grant', return_value=(self.document,self.reference))):
            return B.load_binding(self.root, B.BINDING)

    def test_real_outer_load_binding_refuses_wrong_uuid_source_and_foreign_context(self):
        values = self.bound_memory_fixture()
        self.assertEqual(self.validate_memory_binding(values), values['binding'])
        for target, key, bad in [('binding','gpu_uuid','GPU-other'), ('context','origin','cpu_fixture'),
            ('context','schema','invalid'), ('context','root',str(self.root/'foreign')),
            ('grant','issuer','agent'), ('grant','authorization',1), ('grant','gpu_uuid','GPU-other'),
            ('grant','human_instruction','继续'), ('grant','binding_origin','agent_approved'),
            ('permission','approved_gpu_ids',['GPU-other']), ('permission','max_gpu_hours',9),
            ('permission','allow_driver_or_system_changes',True)]:
            changed = self.bound_memory_fixture(); changed[target][key] = bad
            with self.subTest(target=target,key=key), self.assertRaises((RuntimeError,ValueError)):
                self.validate_memory_binding(changed)

    def test_real_outer_load_binding_refuses_boolean_float_or_enlarged_grant_counts(self):
        for key, bad in [('max_attempts',True), ('max_attempts',1.0), ('max_attempts',2),
            ('max_processes',True), ('max_processes',6.0), ('max_processes',7),
            ('seconds_limit',1200.0), ('seconds_limit',True), ('seconds_limit',1201),
            ('reserved_seconds',1220.0), ('reserved_seconds',True), ('reserved_seconds',1221)]:
            changed = self.bound_memory_fixture(); changed['grant'][key] = bad
            with self.subTest(key=key,bad=bad), self.assertRaises(RuntimeError):
                self.validate_memory_binding(changed)
        for key in ('max_attempts','max_processes','seconds_limit','reserved_seconds'):
            changed = self.bound_memory_fixture(); del changed['grant'][key]
            with self.subTest(missing=key), self.assertRaises(RuntimeError): self.validate_memory_binding(changed)

    def test_context_counts_cannot_expand_authority_production_grant_stays_fixed(self):
        # Context is read-only resource metadata, not the authority. Missing or
        # oversized advisory counts cannot change the derived fixed 1/6 grant;
        # outer binding requires exact int counts in grant and permission.
        for value in (None, True, 100, 6.0):
            context = copy.deepcopy(self.context)
            if value is None:
                context.pop('max_attempts'); context.pop('max_processes')
            else:
                context['max_attempts'] = context['max_processes'] = value
            grant = self.bind_fixture(context)
            self.assertEqual(grant['max_attempts'], 1)
            self.assertEqual(grant['max_processes'], 6)
            self.assertIs(type(grant['max_attempts']), int)
            self.assertIs(type(grant['max_processes']), int)

    def test_real_outer_permission_job_binding_limits_use_exact_types(self):
        for key,bad in [('max_attempts',1.0), ('max_processes',6.0), ('seconds_limit',1200.0),
            ('reserved_seconds',1220.0), ('max_attempts',True), ('max_processes',7)]:
            changed = self.bound_memory_fixture(); changed['permission']['gpu_entry_binding'][key] = bad
            with self.subTest(key=key,bad=bad), self.assertRaises(RuntimeError):
                self.validate_memory_binding(changed)

    def test_real_bind_rejects_wrong_existing_grant_before_permission_write(self):
        marker = self.root / B.GRANT
        marker.parent.mkdir(parents=True)
        marker.write_text('CPU INVALID PLACEHOLDER; NEVER A GRANT', encoding='utf-8')
        for key, bad in [('gpu_uuid','GPU-other'), ('authorization',1), ('max_attempts',1.0),
            ('max_processes',6.0), ('seconds_limit',1200.0), ('reserved_seconds',1220.0),
            ('human_instruction','继续'), ('binding_origin','agent_approved')]:
            values = self.bound_memory_fixture(); values['grant'][key] = bad
            documents = {B.CONTEXT:values['context'], B.GRANT:values['grant']}
            with (patch.object(B,'read',side_effect=lambda root,path:documents[path]),
                patch.object(B,'probe_resources',return_value=values['context']['resources']),
                patch.object(B,'_revision',return_value={}),
                patch.object(B,'ref',side_effect=lambda root,path:values['ref'](path)),
                patch.object(A,'standing_grant',return_value=(self.document,self.reference)),
                patch.object(C,'put',side_effect=AssertionError('invalid grant cannot write permission or site'))):
                with self.subTest(key=key), self.assertRaises((RuntimeError,ValueError)): C.bind(self.root)


class NativeCalibrationInheritanceReviewTests(unittest.TestCase):
    def test_native_runner_and_cost_math_function_asts_preserved(self):
        self.assertEqual(functions(OLD / 'run_native_cost_experiment.py'), functions(NEW / 'run_native_cost_experiment.py'))
        old = functions(OLD / 'native_conditional_cost.py'); new = functions(NEW / 'native_conditional_cost.py')
        for name in ('original_estimator', 'verify_native', 'calculate'):
            if name in old:
                self.assertEqual(old[name], new[name], name)
        self.assertEqual(functions(OLD / 'p4_single_file_receipt.py')['_calibration_a_budget'],
            functions(NEW / 'p4_single_file_receipt.py')['_calibration_a_budget'])

    def test_new_prompt_and_seed_families_match_serializer_without_importing_runner(self):
        runner = constants(NEW / 'run_native_cost_experiment.py')
        self.assertEqual(runner['PROMPT_FIRST'], (40100, 41100, 42100))
        self.assertEqual(runner['SEEDS'], (4029, 4030, 4031))
        self.assertEqual(runner['ORDER'], (('A','B'), ('B','A'), ('A','B')))
        serializer = (NEW / 'prepare_and_verify_native_cost.py').read_text(encoding='utf-8')
        self.assertIn('h["PROMPT_FIRST"] == (40100, 41100, 42100)', serializer)
        self.assertIn('h["SEEDS"] == (4029, 4030, 4031)', serializer)
        self.assertIn('split="calibration" if index < 2 else "validation"', serializer)
        self.assertNotIn('(18100, 19100, 20100)', serializer)
        self.assertNotIn('(1829, 1830, 1831)', serializer)

    def test_receipt_verifier_and_serializer_file_pins_match_actual_new_sources(self):
        tree = ast.parse((NEW / 'p4_single_file_receipt.py').read_bytes())
        values = constants(NEW / 'p4_single_file_receipt.py')
        for name, filename in [('_VERIFIER', 'native_conditional_cost.py'), ('_SERIALIZER', 'prepare_and_verify_native_cost.py')]:
            node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == name)
            self.assertIsInstance(node, ast.Call)
            self.assertEqual(ast.literal_eval(node.args[1]), (NEW / filename).stat().st_size)
            self.assertEqual(ast.literal_eval(node.args[2]), sha((NEW / filename).read_bytes()))
            self.assertEqual(values['_D6'], B.DIR + '/')

    def test_original_ancestor_guard_sdk_and_common_sources_still_fixed(self):
        self.assertEqual(B.GUARD_SHA, '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a')
        self.assertEqual(B.ANCESTOR_SHA, '53fe06db2572b6d07b5dfdaac31ca46aed62fe0710bbb63e006a4d7d089d8e22')
        self.assertEqual(B.COMMON, 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate')
        self.assertIn('--seconds', B.guard_command())
        self.assertIn('1200', B.guard_command())
        self.assertEqual((NEW / 'NATIVE_SOURCE_INHERITANCE.json').read_bytes(),
                         (OLD / 'NATIVE_SOURCE_INHERITANCE.json').read_bytes())

    def test_new_candidate_source_bytes_unchanged_and_no_gpu_imports(self):
        self.assertEqual(source_rows(), BEFORE)
        self.assertFalse([n for n in sys.modules if n.split('.')[0] in FORBIDDEN])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in
        (StandingAuthorizationReviewTests, NativeCalibrationInheritanceReviewTests)])
    with (patch('subprocess.run', side_effect=AssertionError('CPU review forbids subprocess execution')),
        patch('subprocess.Popen', side_effect=AssertionError('CPU review forbids native/GPU launch'))):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = dict(status='PASS_CPU_STANDING_AND_CALIBRATION_REVIEW' if result.wasSuccessful() else 'FAIL',
        tests=dict(run=result.testsRun, errors=len(result.errors), failures=len(result.failures), skipped=len(result.skipped)),
        gpu_operations=0, native_backend_submissions=0, test_log=stream.getvalue(),
        source_before=BEFORE, source_after=source_rows(),
        limitations=['CPU metadata fixtures never persisted as live grant or context.',
            'On and cost/performance qualification remain false.',
            'Production resource, source, budget and original guard gates still required.'])
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(record, out, indent=2); out.write('\n')
    sys.stdout.write(stream.getvalue())
    print(json.dumps({k:record[k] for k in ('status','tests','gpu_operations')}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
