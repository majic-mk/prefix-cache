"""Independent negative evidence for new C5 native entry, never GPU proof."""
from __future__ import annotations
import ast
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

STAGE = Path(os.environ['C5_GPU_ENTRY_REVIEW_CANDIDATE_ROOT']).resolve()
TEMP = Path(os.environ['C5_GPU_ENTRY_REVIEW_TEMP_ROOT']).resolve()
ENTRY = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-revision-20261004'
CONTROL = Path('common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control')
LABEL = 'server11-c5-native-common-cost-gpu01'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixture_ref(root, name, document):
    """Explicitly synthetic, temporary negative metadata; never a grant."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(document, allow_nan=False) + '\n').encode()
    path.write_bytes(raw)
    return dict(path=name, bytes=len(raw), sha256=sha256(raw).hexdigest())


def functions(path):
    return {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}


class NewNativeBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N = load('_independent_new_native_contract', STAGE / 'native_conditional_cost.py')
        cls.S = load('_independent_new_native_serializer', STAGE / 'prepare_and_verify_native_cost.py')
        cls.R = load('_independent_new_native_runner', STAGE / 'run_native_cost_experiment.py')
        cls.B = load('_independent_new_gpu_entry_binding', STAGE / 'gpu_entry_binding.py')
        sys.path.insert(0, str(STAGE))
        cls.J = load('_independent_new_gpu_entry_controller', STAGE / 'control_native_cost_job.py')
        if Path(cls.J.B.__file__).resolve() != (STAGE / 'gpu_entry_binding.py').resolve():
            raise ValueError('controller binding escaped actual new entry revision')
        package = STAGE / CONTROL.parent
        if any(name == 'prefix_io_control' or name.startswith('prefix_io_control.') for name in sys.modules):
            raise ValueError('independent review requires fresh canonical policy modules')
        sys.path.insert(0, str(package))
        cls.P = importlib.import_module('prefix_io_control.p4_policy')
        cls.T = importlib.import_module('prefix_io_control.p4_types')
        cls.F = importlib.import_module('prefix_io_control.p4_single_file_receipt')
        if Path(cls.F.__file__).resolve() != (STAGE / CONTROL / 'p4_single_file_receipt.py').resolve():
            raise ValueError('canonical receipt escaped actual new candidate')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='n', dir=TEMP)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_original_c5_engine_collector_policy_bridge_bytes(self):
        rows = {
            'common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py':
                'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
            'common_candidate/native_full_step_collector.py':
                'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf',
            (CONTROL / 'p4_policy.py').as_posix():
                'bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673',
            (CONTROL / 'p4_bridge.py').as_posix():
                '6bec5dd57e25cfc2c0e62540742a781d656ba528cb6c8057cd2402197db93e1a',
        }
        for relative, expected in rows.items():
            with self.subTest(path=relative):
                self.assertEqual(sha256((STAGE / relative).read_bytes()).hexdigest(), expected)

    def test_original_formula_capture_io_and_drain_ast(self):
        old = STAGE.parent / 'native_cost_v6/native_conditional_cost.py'
        if not old.exists():
            # Server path: the frozen parent source is in project artifacts.
            old = Path(os.environ['C5_GPU_ENTRY_REVIEW_PROJECT_ROOT']) / \
                'artifacts/prefix_io_v1/server11-native-cost-v6-20261003/native_conditional_cost.py'
        self.assertEqual(sha256(old.read_bytes()).hexdigest(),
                         'e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291')
        old_funcs, new_funcs = functions(old), functions(STAGE / 'native_conditional_cost.py')
        for name in ('original_estimator', 'validate_capture', 'validate_io',
                     'original_post_shutdown_drain', 'analyze_paired'):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old_funcs[name], include_attributes=False),
                                 ast.dump(new_funcs[name], include_attributes=False))

    def test_native_six_window_design_is_fixed(self):
        self.assertEqual(self.R.ORDER, (('A', 'B'), ('B', 'A'), ('A', 'B')))
        self.assertEqual((self.R.PROMPT_TOKENS, self.R.OPERATION_COUNT, self.R.FILE_BYTES,
                          self.R.MAX_ACCEPTED_PARENTS), (129, 1, 917504, 8))
        self.assertEqual(self.R.PROMPT_FIRST, (18100, 19100, 20100))
        self.assertEqual(self.R.SEEDS, (1829, 1830, 1831))
        self.assertEqual(self.R.LABEL, LABEL)
        self.assertEqual(self.R.DELIVERY, ENTRY)

    def test_canonical_pins_are_actual_new_verifier_and_serializer(self):
        for pin, filename in ((self.F._VERIFIER, 'native_conditional_cost.py'),
                              (self.F._SERIALIZER, 'prepare_and_verify_native_cost.py')):
            raw = (STAGE / filename).read_bytes()
            self.assertEqual(pin.path, ENTRY + '/' + filename)
            self.assertEqual((pin.bytes, pin.sha256), (len(raw), sha256(raw).hexdigest()))

    def test_private_constructor_rejects_dict_and_pass(self):
        for value in (None, {}, {'status': 'PASS', 'native_execution_verified': True}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'source-bound'):
                    self.F.ExactSingleFileReceipt(value)

    def test_policy_rejects_pass_dict_and_subclass(self):
        config = self.T.P4Config(mode='interference', sample_max_age_ns=100000000,
                                 max_wait_ns=100000000, internal_step_budget_ns=100)
        class WrongReceipt(self.F.ExactSingleFileReceipt):
            pass
        # No exact receipt is constructed. A wrong class object is rejected
        # before any receipt field can be read or attached.
        values = ({'status': 'PASS', 'cost_upper_ns': 1}, object.__new__(WrongReceipt))
        for value in values:
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'verified exact'):
                    self.P.P4Policy('independent-negative', config, single_file=value)

    def test_synthetic_plan_rejected_before_binding_import(self):
        plans = (
            {'evidence_origin': 'synthetic_cpu_contract', 'cpu_preparation_only': True, 'job_id': LABEL},
            {'evidence_origin': 'native_runtime_preregistered', 'cpu_preparation_only': True, 'job_id': LABEL},
            {'evidence_origin': 'native_runtime_preregistered', 'cpu_preparation_only': 0, 'job_id': LABEL},
        )
        with patch.object(self.N, 'binding_api', side_effect=AssertionError('binding reached')):
            for plan in plans:
                with self.subTest(plan=plan):
                    with self.assertRaisesRegex(ValueError, 'synthetic, preparation and old'):
                        self.N.validate_native_plan(self.root, plan)

    def test_old_job_rejected_before_binding_import(self):
        for job in ('server11-native-cost-six-window-06',
                    'server11-c5-native-common-cost-preparation01'):
            plan = {'evidence_origin': 'native_runtime_preregistered', 'cpu_preparation_only': False,
                    'job_id': job}
            with patch.object(self.N, 'binding_api', side_effect=AssertionError('binding reached')):
                with self.assertRaisesRegex(ValueError, 'synthetic, preparation and old'):
                    self.N.validate_native_plan(self.root, plan)

    def test_wrong_project_plan_rejected_before_binding_import(self):
        plan = {'evidence_origin': 'native_runtime_preregistered', 'cpu_preparation_only': False,
                'job_id': LABEL, 'project_root': '/other/project'}
        with patch.object(self.N, 'binding_api', side_effect=AssertionError('binding reached')):
            with self.assertRaisesRegex(ValueError, 'actual project root'):
                self.N.validate_native_plan(self.root, plan)

    def test_serializer_refuses_cpu_fixture_before_raw_read(self):
        plan = {'evidence_origin': 'synthetic_cpu_contract', 'cpu_preparation_only': True, 'job_id': LABEL}
        plan_ref = fixture_ref(self.root, 'negative_plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'synthetic, preparation and old'):
            self.S.serialize_runtime_record(self.root, runtime_relative='must_not_be_read.json',
                plan_ref=plan_ref, source_verification_refs={})

    def test_window_checker_refuses_cpu_fixture_before_raw_read(self):
        plan = {'evidence_origin': 'synthetic_cpu_contract', 'cpu_preparation_only': True, 'job_id': LABEL}
        plan_ref = fixture_ref(self.root, 'negative_plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'synthetic, preparation and old'):
            self.S.verify_raw_window(self.root, plan_ref=plan_ref, child_relative='must_not_be_read.json')

    def test_verify_cell_expected_refs_cannot_be_self_substituted(self):
        with self.assertRaisesRegex(ValueError, 'independently expected'):
            self.N.verify_native_cell(self.root, plan_ref={}, measurements_ref={}, guard_ref={},
                expected_plan_ref={'path': 'other'}, expected_guard_ref={}, original_source_path=None)

    def test_source_read_detects_same_length_mutation(self):
        row = fixture_ref(self.root, 'negative_source.json', {'negative_fixture': 'A'})
        path = self.root / row['path']
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"A"', b'"B"'))
        with self.assertRaisesRegex(ValueError, 'contents'):
            self.F.FileRef.from_mapping(row).read(self.root)

    def test_source_ref_rejects_traversal_and_boolean_size(self):
        row = fixture_ref(self.root, 'negative_source.json', {'negative_fixture': True})
        with self.assertRaisesRegex(ValueError, 'size'):
            self.F.FileRef.from_mapping(dict(row, bytes=True))
        with self.assertRaisesRegex(ValueError, 'relative'):
            self.F.FileRef.from_mapping(dict(row, path='../negative_source.json')).read(self.root)

    def test_new_binding_does_not_accept_old_scope_path(self):
        for value in (None, 'artifacts/prefix_io_v1/server11-native-cost-v6-20261003/SIX_PROCESS_AUTHORIZED_SCOPE.json',
                      'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004/GPU_ENTRY_BINDING.json'):
            with self.subTest(path=value):
                with self.assertRaisesRegex(RuntimeError, 'wrong revision binding path'):
                    self.B.load_binding(self.root, value)

    def test_unbound_template_cannot_be_binding(self):
        fixture_ref(self.root, self.B.BINDING, {'schema': 'c5_gpu_entry_unbound_review_template_v1',
            'status': 'UNBOUND', 'gpu_uuid': None, 'gpu_launch_allowed': False})
        with self.assertRaisesRegex(RuntimeError, 'UNBOUND/CPU preparation'):
            self.B.load_binding(self.root, self.B.BINDING)

    def test_cpu_binding_origin_is_rejected_before_ancestry(self):
        fixture_ref(self.root, self.B.BINDING, {'schema': 'c5_gpu_entry_site_binding_v1',
            'status': 'LIVE_HUMAN_BOUND', 'origin': 'synthetic_cpu_contract', 'synthetic': True})
        with patch.object(self.B, '_revision', side_effect=AssertionError('ancestry reached')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic or unknown'):
                self.B.load_binding(self.root, self.B.BINDING)

    def test_synthetic_false_boolean_is_required(self):
        fixture_ref(self.root, self.B.BINDING, {'schema': 'c5_gpu_entry_site_binding_v1',
            'status': 'LIVE_HUMAN_BOUND', 'origin': 'live_site_binding', 'synthetic': 0})
        with patch.object(self.B, '_revision', side_effect=AssertionError('ancestry reached')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic or unknown'):
                self.B.load_binding(self.root, self.B.BINDING)

    def test_configuration_actual_path_reaches_missing_binding_gate(self):
        fixture_ref(self.root, self.B.CONFIG, {})
        # A valid real Path and the equivalent relative name must normalize to
        # the same config, then reject its missing binding before any GPU I/O.
        for path in (self.B.CONFIG, self.root / self.B.CONFIG):
            with self.subTest(path=str(path)):
                with self.assertRaisesRegex(RuntimeError, 'invalid immutable reference'):
                    self.B.verify_configuration(self.root, path)

    def test_foreign_configuration_path_rejected(self):
        for path in ('artifacts/prefix_io_v1/server11-native-cost-v6-20261003/NATIVE_COST_CONFIG.json',
                     self.root / 'foreign' / 'NATIVE_COST_CONFIG.json'):
            with self.subTest(path=str(path)):
                with self.assertRaisesRegex(RuntimeError, 'configuration|config path'):
                    self.B.verify_configuration(self.root, path)

    def test_new_checked_ref_rejects_source_drift(self):
        row = fixture_ref(self.root, 'negative_source.json', {'test': 'A'})
        path = self.root / row['path']
        path.write_bytes(path.read_bytes().replace(b'"A"', b'"B"'))
        with self.assertRaisesRegex(RuntimeError, 'bytes/source drift'):
            self.B.checked_ref(self.root, row)

    def test_new_checked_rows_reject_duplicate_and_bool_size(self):
        row = fixture_ref(self.root, 'negative_source.json', {'test': 'A'})
        with self.assertRaisesRegex(RuntimeError, 'duplicate source path'):
            self.B.checked_rows(self.root, [row, row])
        with self.assertRaisesRegex(RuntimeError, 'invalid immutable reference'):
            self.B.checked_ref(self.root, dict(row, bytes=True))

    def test_resource_shape_rejects_unknown_uuid_and_cpu_origin(self):
        # These are value-only synthetic negative inputs. No resource probe,
        # binding, permission, guard or valid execution is created from them.
        resource = {'origin': 'live_linux_readonly_resource_probe', 'status': 'LIVE_RESOURCE_READY',
                    'nvidia_nodes_visible': True, 'compute_processes': [],
                    'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000001',
                    'cpu_cores': 1, 'cpu_affinity_count': 1, 'available_memory_bytes': 32 * 1024**3,
                    'gpu_free_mib': 28000, 'primary_free_bytes': 10 * 1024**3,
                    'cpu_counters_readable': True}
        mutations = ({'gpu_uuid': None}, {'gpu_uuid': 'GPU-unknown'},
                     {'origin': 'cpu_fixture'}, {'nvidia_nodes_visible': False},
                     {'compute_processes': [123]}, {'cpu_cores': 0.5},
                     {'cpu_counters_readable': False}, {'available_memory_bytes': 2 * 1024**3})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(RuntimeError):
                    self.B._resource_shape(dict(resource, **mutation))

    def test_resource_shape_rejects_bool_numeric_fields(self):
        resource = {'origin': 'live_linux_readonly_resource_probe', 'status': 'LIVE_RESOURCE_READY',
                    'nvidia_nodes_visible': True, 'compute_processes': [],
                    'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000001',
                    'cpu_cores': 1, 'cpu_affinity_count': 1, 'available_memory_bytes': 32 * 1024**3,
                    'gpu_free_mib': 28000, 'primary_free_bytes': 10 * 1024**3,
                    'cpu_counters_readable': True}
        for key in ('cpu_cores', 'cpu_affinity_count', 'available_memory_bytes',
                    'gpu_free_mib', 'primary_free_bytes'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, 'invalid number'):
                    self.B._resource_shape(dict(resource, **{key: True}))

    def test_new_json_parser_rejects_duplicate_and_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError, 'duplicate JSON key'):
            self.B.parse('{"gpu_uuid":null,"gpu_uuid":"old"}')
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            self.B.parse('{"cpu_cores":NaN}')

    def test_metadata_only_rows_cannot_escape_root(self):
        row = {'path': '../outside.py', 'bytes': 1, 'sha256': '0' * 64}
        with self.assertRaisesRegex(RuntimeError, 'project-relative'):
            self.B.checked_rows(self.root, [row], verify_bytes=False)

    def test_no_card_prepare_only_writes_inert_template(self):
        output = self.J.B.DIR + '/UNBOUND_REVIEW_TEMPLATE.json'
        with patch.object(self.J.B, 'probe_resources', side_effect=AssertionError('GPU probe reached')):
            result = self.J.prepare(self.root, output)
        self.assertIs(result['gpu_launch_allowed'], False)
        self.assertEqual(result['actual_gpu_runs'], 0)
        template = self.J.B.read(self.root, output)
        self.assertEqual(template['status'], 'UNBOUND')
        for key in ('gpu_uuid', 'valid_config', 'site_binding', 'human_grant', 'native_receipt',
                    'effective_cost_upper_ns', 'effective_step_budget_ns'):
            self.assertIsNone(template[key])
        for relative in (self.J.B.CONFIG, self.J.B.BINDING, self.J.B.GRANT,
                         self.J.B.PERMISSION, self.J.B.SITE_LOCK):
            self.assertFalse((self.root / relative).exists())

    def test_unbound_template_plan_command_reaches_metadata_gate(self):
        output = self.J.B.DIR + '/UNBOUND_REVIEW_TEMPLATE.json'
        self.J.prepare(self.root, output)
        command = self.J.B.read(self.root, output)['commands']['plan']
        self.assertEqual(command[:3], ['.venv/bin/python', '-B',
            self.J.B.DIR + '/prepare_and_verify_native_cost.py'])
        # Parsing the promised argv reaches the plan metadata gate; the gate
        # itself is isolated to reject, so no effective plan/authority is made.
        with patch.object(self.S, 'create_plan', side_effect=RuntimeError('independent plan metadata stop')):
            with self.assertRaisesRegex(RuntimeError, 'independent plan metadata stop'):
                self.S.main(command[3:])

    def test_binding_exact_types_reject_boolean_numeric_aliases(self):
        for value, expected in ((True, 1), (False, 0), (1, True), (0, False), (1.0, 1)):
            with self.subTest(value=value, expected=expected):
                self.assertFalse(self.B.exact(value, expected))

    def test_prelaunch_plan_self_ref_cannot_replace_original_intent(self):
        supplied = fixture_ref(self.root, 'negative_plan.json', {'synthetic_rejection_input': 'after'})
        different = dict(supplied, sha256='0' * 64)
        fixture_ref(self.root, ENTRY + '/GPU_LAUNCH_INTENT.json', {'plan_ref': different})
        with self.assertRaisesRegex(ValueError, 'independent prelaunch intent'):
            self.N.validate_prelaunch_plan(self.root, supplied)

