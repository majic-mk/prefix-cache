"""Independent negative call-site checks, explicitly synthetic CPU contracts.

Source binding fixtures below model immutable values only. No model, torch Event,
receipt issuance, I/O completion, cost qualification or guarded process is made.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from contextlib import redirect_stdout, ExitStack
from hashlib import sha256
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

STAGE = Path(os.environ['C5_NATIVE_REVIEW_STAGE_ROOT'])
NATIVE = Path(os.environ['C5_NATIVE_REVIEW_NATIVE_ROOT'])
CANDIDATE = Path(os.environ['C5_NATIVE_REVIEW_CANDIDATE_ROOT'])
ORIGINAL = Path(os.environ['C5_NATIVE_REVIEW_ORIGINAL_ESTIMATOR'])
PREP = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
OVERLAY = PREP + '/common_candidate/source'
REACTOR = OVERLAY + '/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
COLLECTOR = PREP + '/common_candidate/native_full_step_collector.py'
POLICY_REL = Path('source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control')
BLOCKED = 'GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY'
C5_R = 'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'
C5_C = 'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def top_functions(path):
    return {node.name: node for node in ast.parse(path.read_bytes()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


class Poison:
    def __getattribute__(self, name):
        raise AssertionError('blocked call attempted value access: ' + name)
    def __str__(self):
        raise AssertionError('blocked call attempted string conversion')
    def __fspath__(self):
        raise AssertionError('blocked call attempted path conversion')


class IndependentNativePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serializer = load('_review_c5_serializer', STAGE / 'prepare_and_verify_native_cost.py')
        cls.verifier = cls.serializer.C
        cls.calibrator = load('_review_c5_calibrator', STAGE / 'run_native_cost_experiment.py')
        cls.control = load('_review_c5_controller', STAGE / 'control_native_cost_job.py')
        cls.contract = load('_review_c5_contract', STAGE / 'native_preparation_contract.py')
        package_root = STAGE / 'common_candidate' / POLICY_REL.parent
        # Only modules from this exact canonical overlay are permitted to remain.
        for name in tuple(sys.modules):
            if name == 'prefix_io_control' or name.startswith('prefix_io_control.'):
                del sys.modules[name]
        sys.path.insert(0, str(package_root))
        cls.receipt = importlib.import_module('prefix_io_control.p4_single_file_receipt')
        cls.policy = importlib.import_module('prefix_io_control.p4_policy')
        cls.bridge = importlib.import_module('prefix_io_control.p4_bridge')
        cls.types = importlib.import_module('prefix_io_control.p4_types')
        cls.source_map = {
            REACTOR: dict(path=REACTOR, bytes=(STAGE / 'common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py').stat().st_size, sha256=C5_R),
            COLLECTOR: dict(path=COLLECTOR, bytes=(STAGE / 'common_candidate/native_full_step_collector.py').stat().st_size, sha256=C5_C),
        }

    def assert_blocked(self, fn, *args, **kwargs):
        """Real public entry point must fail before any file/process side effect."""
        with ExitStack() as stack:
            for target in ('builtins.open', 'pathlib.Path.read_bytes', 'pathlib.Path.read_text',
                           'pathlib.Path.write_bytes', 'pathlib.Path.write_text',
                           'pathlib.Path.mkdir', 'subprocess.run', 'subprocess.Popen'):
                stack.enter_context(patch(target, side_effect=AssertionError('side effect before block: ' + target)))
            with self.assertRaisesRegex(RuntimeError, '^' + BLOCKED + '$'):
                fn(*args, **kwargs)

    def binding_values(self):
        native, collector = deepcopy(self.source_map[REACTOR]), deepcopy(self.source_map[COLLECTOR])
        plan = dict(cpu_preparation_only=True, evidence_origin='synthetic_cpu_contract',
            job_id=self.calibrator.LABEL, gpu_uuid=None, common_overlay_relative=OVERLAY, native_source_ref=native,
            native_source_sha256=C5_R, collector_source_ref=collector,
            common_owner_parameters=dict(max_accepted_parents=8, bridge_is_none=True))
        child = dict(native_source_binding=dict(actual_source_ref=native, original_run_code_verified=True,
            max_accepted_parents=8, bridge_is_none=True,
            loaded_native_modules=[dict(module='py_kvcache.reactor', source_ref=native)]),
            collector_native_source_binding=dict(
                legacy_lookup_key='third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
                actual_source_ref=native, adapter_native_source_sha256_before=C5_R,
                adapter_native_source_sha256_after=C5_R, same_adapter_all_frames=True, full_frame_count=128))
        return plan, child

    def check_binding(self, plan, child):
        # Explicit synthetic source lookup: no physical operation is asserted.
        with patch.object(self.serializer, 'ref', side_effect=lambda root, name: deepcopy(self.source_map[name])):
            return self.serializer.validate_common_source_binding(Poison(), plan, child)

    def test_01_create_plan_blocks_before_paths(self):
        self.assert_blocked(self.serializer.create_plan, Poison(), source_lock_relative=Poison(),
                            config_relative=Poison(), output_relative=Poison())

    def test_02_final_native_verifier_blocks_even_matching_old_refs(self):
        p = Poison()
        self.assert_blocked(self.verifier.verify_native_cell, p, plan_ref=p, measurements_ref=p,
                            guard_ref=p, expected_plan_ref=p, expected_guard_ref=p, original_source_path=p)

    def test_03_receipt_loader_blocks_before_binding(self):
        self.assert_blocked(self.receipt.load_verified_single_file, Poison(), Poison())

    def test_04_acquisition_configuration_guard_and_parent_blocks(self):
        p = Poison()
        cases = [(self.calibrator.load_configuration, (p,)), (self.calibrator.verify_guard, (p,)),
                 (self.calibrator.execute_window, (p, p, p, p, p)), (self.calibrator.execute_parent, (p, p, p, p))]
        for fn, values in cases:
            with self.subTest(entry=fn.__name__):
                self.assert_blocked(fn, *values)

    def test_05_controller_all_mutation_and_status_entries_block(self):
        for name, values in [('prepare', ()), ('scope', (Poison(),)), ('launch', ()),
                             ('status', ()), ('check_sources', (Poison(),))]:
            with self.subTest(entry=name):
                self.assert_blocked(getattr(self.control, name), *values)

    def test_06_cli_main_does_not_parse_or_touch_old_scope(self):
        for module in (self.serializer, self.calibrator, self.control, self.contract, self.verifier, self.receipt):
            with self.subTest(module=module.__name__), redirect_stdout(io.StringIO()) as stream:
                with patch.object(sys, 'argv', ['entry.py', '--launch', '--old-authorized-scope']):
                    self.assertEqual(module.main(), 2)
                document = json.loads(stream.getvalue())
                self.assertEqual(document['status'], BLOCKED)
                self.assertIs(document['gpu_launch_allowed'], False)
                self.assertIsNone(document['valid_native_receipt'])

    def test_07_uuid_is_unbound_in_both_real_entries(self):
        self.assertIsNone(self.calibrator.GPU_UUID)
        self.assertIsNone(self.control.GPU)
        self.assertIsNone(self.control.PERMISSION)
        self.assertIsNone(self.control.ANCESTOR)
        self.assertIsNone(self.control.ANCESTOR_SHA)
        self.assertIsNone(self.calibrator.STORAGE)
        self.assertIsNone(self.contract.draft_common_calibration()['gpu_uuid'])

    def test_08_all_new_runtime_paths_share_same_overlay(self):
        self.assertEqual(self.calibrator.DELIVERY, PREP)
        self.assertEqual(self.control.D, PREP)
        self.assertEqual(self.calibrator.CANDIDATE, PREP + '/common_candidate')
        self.assertEqual(self.control.CANDIDATE, self.calibrator.CANDIDATE)
        self.assertEqual(self.calibrator.REACTOR, REACTOR)
        self.assertEqual(self.calibrator.SCRIPT, PREP + '/run_native_cost_experiment.py')
        self.assertEqual(self.receipt._COMMON_OVERLAY, OVERLAY)
        self.assertEqual(self.receipt._NATIVE, REACTOR)

    def test_09_real_c5_reactor_collector_are_byte_identical(self):
        self.assertEqual(sha(STAGE / 'common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'), C5_R)
        self.assertEqual(sha(STAGE / 'common_candidate/native_full_step_collector.py'), C5_C)

    def test_10_original_policy_and_bridge_are_byte_identical(self):
        for name in ('p4_policy.py', 'p4_bridge.py'):
            with self.subTest(source=name):
                self.assertEqual((STAGE / 'common_candidate' / POLICY_REL / name).read_bytes(),
                                 (CANDIDATE / POLICY_REL / name).read_bytes())

    def test_11_policy_bridge_share_exact_canonical_module(self):
        self.assertIs(self.bridge.P4Policy, self.policy.P4Policy)
        self.assertIs(sys.modules['prefix_io_control.p4_single_file_receipt'], self.receipt)
        for module in (self.policy, self.bridge, self.receipt):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to((STAGE / 'common_candidate').resolve()))
        node = top_functions(Path(self.policy.__file__))
        tree = ast.parse(Path(self.policy.__file__).read_bytes())
        init = next(n for n in next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'P4Policy').body
                    if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == 'p4_single_file_receipt'
                            for n in ast.walk(init)))
        self.assertTrue(any(isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.IsNot)
                            and isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Name) and n.left.func.id == 'type'
                            and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == 'ExactSingleFileReceipt'
                            for n in ast.walk(init)))

    def test_12_external_same_named_receipt_and_dict_rejected_by_actual_policy(self):
        config = self.types.P4Config('interference', 100_000_000, 100_000_000, 100)
        class ExactSingleFileReceipt:
            step_budget_ns = 100
            cost_upper_ns = 1
        class Derived(self.receipt.ExactSingleFileReceipt):
            pass
        for value in ({'status': 'PASS'}, ExactSingleFileReceipt(), object.__new__(Derived)):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'verified exact single-file'):
                    self.policy.make_p4_policy('synthetic-review', config, single_file=value)

    def test_13_inherited_old_receipt_cannot_pass_original_exact_type_check(self):
        old = load('_review_old_c4_receipt', CANDIDATE / POLICY_REL / 'p4_single_file_receipt.py')
        old_value = object.__new__(old.ExactSingleFileReceipt)
        self.assertIsNot(type(old_value), self.receipt.ExactSingleFileReceipt)
        config = self.types.P4Config('interference', 100_000_000, 100_000_000, 100)
        with self.assertRaisesRegex(ValueError, 'verified exact single-file'):
            self.policy.make_p4_policy('synthetic-review', config, single_file=old_value)

    def test_14_receipt_constructor_does_not_accept_dict_pass(self):
        with self.assertRaisesRegex(ValueError, 'source-bound'):
            self.receipt.ExactSingleFileReceipt(status='PASS', production_qualified=False)

    def test_15_new_serializer_and_verifier_dependency_pins_match(self):
        for ref, name in ((self.receipt._VERIFIER, 'native_conditional_cost.py'),
                          (self.receipt._SERIALIZER, 'prepare_and_verify_native_cost.py')):
            with self.subTest(source=name):
                self.assertEqual(ref.path, PREP + '/' + name)
                self.assertEqual(ref.bytes, (STAGE / name).stat().st_size)
                self.assertEqual(ref.sha256, sha(STAGE / name))

    def test_16_original_numerical_capture_and_io_ast_unchanged(self):
        before, after = top_functions(NATIVE / 'native_conditional_cost.py'), top_functions(STAGE / 'native_conditional_cost.py')
        for name in ('original_estimator', 'validate_capture', 'validate_io', 'original_post_shutdown_drain', 'analyze_paired'):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(before[name], include_attributes=False), ast.dump(after[name], include_attributes=False))
        result = self.contract.assert_original_algorithms(NATIVE / 'native_conditional_cost.py')
        self.assertIs(result['numerical_ast_changed'], False)
        self.assertEqual(sha(ORIGINAL), '3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac')

    def test_17_a_only_calibration_budget_ast_unchanged(self):
        before = top_functions(CANDIDATE / POLICY_REL / 'p4_single_file_receipt.py')['_calibration_a_budget']
        after = top_functions(Path(self.receipt.__file__))['_calibration_a_budget']
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_18_fixed_six_window_design_remains_unchanged(self):
        draft = self.contract.draft_common_calibration()
        self.assertEqual(self.calibrator.ORDER, (('A', 'B'), ('B', 'A'), ('A', 'B')))
        self.assertEqual(draft['fixed_order'], ['A0', 'B0', 'B1', 'A1', 'A2', 'B2'])
        self.assertEqual((draft['calibration_pairs'], draft['validation_pairs']), (2, 1))
        self.assertEqual((draft['prompt_tokens'], draft['cached_prompt_tokens'], draft['required_output_tokens_each']), (129, 128, 128))
        self.assertEqual((draft['measured_offset'], draft['warmup_offsets']), (16, [1]))
        self.assertEqual((draft['action_operations'], draft['action_physical_bytes']), (1, 917504))
        self.assertEqual(self.calibrator.FILE_BYTES, 917504)
        self.assertIs(draft['bridge_is_none'], True)
        self.assertEqual(self.calibrator.MAX_ACCEPTED_PARENTS, 8)

    def test_19_draft_has_no_plan_job_scope_receipt_or_qualification(self):
        draft = self.contract.draft_common_calibration()
        self.assertEqual(draft['scope'], 'c5_native_common_source_draft_v1')
        self.assertEqual(draft['status'], 'CPU_DRAFT_GPU_UUID_UNRESOLVED')
        for name in ('native_plan_created', 'gpu_scope_created', 'gpu_job_created', 'gpu_launch_allowed',
                     'native_execution_verified', 'native_cost_qualified', 'full_runtime_cost_qualified', 'on_observation_cost_measured'):
            self.assertIs(draft[name], False)
        for name in ('gpu_uuid', 'valid_native_receipt', 'effective_cost_upper_ns', 'effective_step_budget_ns'):
            self.assertIsNone(draft[name])

    def test_20_actual_serializer_default_native_calls_refused_before_input_access(self):
        p = Poison()
        for value in (False, None, 1, 'true'):
            with self.subTest(synthetic_cpu=value):
                with self.assertRaisesRegex(ValueError, 'explicit synthetic CPU'):
                    self.serializer.serialize_runtime_record(p, runtime_relative=p, plan_ref=p,
                                                            source_verification_refs=p, synthetic_cpu=value)
                with self.assertRaisesRegex(ValueError, 'explicit synthetic CPU'):
                    self.serializer.verify_raw_window(p, plan_ref=p, child_relative=p, synthetic_cpu=value)

    def test_21_cpu_envelope_rejects_native_origin_before_file_access(self):
        for origin in ('native_gpu_recording', 'PASS', None):
            envelope = dict(origin=origin, runtime_relative=Poison(), plan_ref=Poison(), source_verification_refs=Poison())
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, 'explicit synthetic CPU'):
                self.contract.audit_synthetic_raw(Poison(), envelope=envelope, original_estimator_path=Poison())

    def test_22_common_source_binding_accepts_only_correct_synthetic_shape(self):
        plan, child = self.binding_values()
        self.assertIsNone(self.check_binding(plan, child))

    def test_23_old_plan_or_native_origin_rejected(self):
        for key, value in [('cpu_preparation_only', False), ('cpu_preparation_only', 1),
                           ('evidence_origin', 'native_gpu_recording'), ('job_id', 'server11-native-cost-six-window-06'),
                           ('common_overlay_relative', 'artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003/source')]:
            plan, child = self.binding_values(); plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check_binding(plan, child)

    def test_24_wrong_reactor_hash_or_actual_source_binding_rejected(self):
        for section in ('native_source_ref', 'collector_source_ref'):
            plan, child = self.binding_values(); plan[section]['sha256'] = '0' * 64
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.check_binding(plan, child)
        plan, child = self.binding_values(); child['native_source_binding']['actual_source_ref'] = dict(self.source_map[COLLECTOR])
        with self.assertRaises(ValueError): self.check_binding(plan, child)

    def test_25_shared_owner_limits_and_common_bridge_required(self):
        for which, key, value in [('plan', 'max_accepted_parents', 9), ('plan', 'max_accepted_parents', True),
                                  ('plan', 'bridge_is_none', False), ('child', 'max_accepted_parents', 9),
                                  ('child', 'bridge_is_none', False)]:
            plan, child = self.binding_values()
            (plan['common_owner_parameters'] if which == 'plan' else child['native_source_binding'])[key] = value
            with self.subTest(which=which, key=key, value=value), self.assertRaises(ValueError):
                self.check_binding(plan, child)

    def test_26_loaded_module_reactor_absent_duplicate_or_escaped_rejected(self):
        mutations = [[], [dict(module='torch', source_ref=self.source_map[REACTOR])],
                     [dict(module='py_kvcache.reactor', source_ref=self.source_map[REACTOR])] * 2,
                     [dict(module='prefix_io_control.p4_policy', source_ref=self.source_map[REACTOR])]]
        for rows in mutations:
            plan, child = self.binding_values(); child['native_source_binding']['loaded_native_modules'] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError): self.check_binding(plan, child)

    def test_27_same_adapter_identity_requires_all_128_frames(self):
        for key, value in [('full_frame_count', 127), ('full_frame_count', True), ('same_adapter_all_frames', False),
                           ('adapter_native_source_sha256_before', '0' * 64), ('adapter_native_source_sha256_after', '0' * 64),
                           ('legacy_lookup_key', REACTOR)]:
            plan, child = self.binding_values(); child['collector_native_source_binding'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.check_binding(plan, child)

    def test_28_collector_alias_points_to_actual_new_c5_ref_without_mutation(self):
        original = deepcopy(self.source_map)
        result = self.calibrator.collector_source_view(original)
        self.assertEqual(original, self.source_map)
        self.assertEqual(result[self.calibrator.LEGACY_COLLECTOR_REACTOR_KEY], self.source_map[REACTOR])
        self.assertEqual(result[REACTOR], self.source_map[REACTOR])
        wrong = deepcopy(original); wrong[REACTOR]['path'] = self.calibrator.LEGACY_COLLECTOR_REACTOR_KEY
        with self.assertRaises(ValueError): self.calibrator.collector_source_view(wrong)

    def runtime_values(self):
        def f(name): return dict(path=name, bytes=1, sha256='1' * 64)
        common = [self.source_map[REACTOR], f('model-plan.json'), f('model-config.json'), f('event.py'), f(self.receipt._RUNNER)]
        overlay = [self.source_map[COLLECTOR]]
        plan = dict(source_lock_ref=f('source-lock.json'), collector_source_ref=self.source_map[COLLECTOR],
                    model_plan_ref=common[1], model_config_ref=common[2], cuda_event_source_ref=common[3])
        binding = dict(runtime_common_refs=common, runtime_overlay_refs=overlay)
        return plan, binding

    def runtime_check(self, plan, binding, locked_rows=None):
        lock = dict(files=locked_rows if locked_rows is not None else
                    [*self.source_map.values(), *binding['runtime_common_refs'], *binding['runtime_overlay_refs']])
        # Synthetic immutable source metadata only; no original receipt is issued.
        def read(ref, project):
            return json.dumps(lock).encode() if ref.path == 'source-lock.json' else b'x'
        with patch.object(self.receipt.FileRef, 'read', read):
            return self.receipt._runtime_refs(Poison(), binding, plan)

    def test_29_collector_exactly_once_in_overlay_and_reactor_common(self):
        plan, binding = self.runtime_values()
        common, overlay = self.runtime_check(plan, binding)
        self.assertEqual(sum(ref.path == COLLECTOR for ref in overlay), 1)
        self.assertFalse(any(ref.path == COLLECTOR for ref in common))
        self.assertTrue(any(ref.path == REACTOR for ref in common))

    def test_30_collector_common_duplicate_absent_or_old_ref_rejected(self):
        for kind in ('common', 'duplicate', 'absent', 'old'):
            plan, binding = self.runtime_values()
            if kind == 'common':
                binding['runtime_common_refs'].append(binding['runtime_overlay_refs'].pop())
                binding['runtime_overlay_refs'].append(dict(path='another.py', bytes=1, sha256='1' * 64))
            elif kind == 'duplicate': binding['runtime_overlay_refs'] *= 2
            elif kind == 'absent': binding['runtime_overlay_refs'] = [dict(path='another.py', bytes=1, sha256='1' * 64)]
            else: plan['collector_source_ref'] = dict(path='old/collector.py', bytes=1, sha256=C5_C)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.runtime_check(plan, binding)

    def test_31_canonical_a_budget_ignores_holdout_and_action(self):
        plan = dict(entries=[dict(pair_id='p0', split='calibration'), dict(pair_id='p1', split='calibration'),
                             dict(pair_id='p2', split='validation')])
        rows = [dict(arm='baseline', pair_id='p0', selected_gpu_elapsed_ns=10),
                dict(arm='baseline', pair_id='p1', selected_gpu_elapsed_ns=14),
                dict(arm='baseline', pair_id='p2', selected_gpu_elapsed_ns=10**12),
                dict(arm='action', pair_id='p0', selected_gpu_elapsed_ns=10**12)]
        verified = dict(windows=rows, calibration_only_cost=dict(baseline_ns=12))
        self.assertEqual(self.receipt._calibration_a_budget(plan, verified), 14)
        verified['windows'][2]['selected_gpu_elapsed_ns'] *= 2
        verified['windows'][3]['selected_gpu_elapsed_ns'] *= 3
        self.assertEqual(self.receipt._calibration_a_budget(plan, verified), 14)

    def test_32_imports_do_not_load_model_packages(self):
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') or name == 'vllm' or name.startswith('vllm.')
                             for name in sys.modules))

    def test_33_cpu_preparation_cannot_invent_or_reuse_a_gpu_uuid(self):
        for value in ('GPU-CPU-TEST', 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9', '', 0, False):
            plan, child = self.binding_values(); plan['gpu_uuid'] = value
            with self.subTest(uuid=value), self.assertRaisesRegex(ValueError, 'resolved GPU UUID'):
                self.check_binding(plan, child)

    def test_34_inheritance_keeps_all_ancestor_python_except_canonical_receipt(self):
        inheritance = json.loads((STAGE / 'SOURCE_INHERITANCE.json').read_bytes())
        self.assertIs(inheritance['gpu_launch_allowed'], False)
        self.assertEqual(inheritance['actual_gpu_runs'], 0)
        rows = inheritance['source_parent_files']
        self.assertGreaterEqual(len(rows), 64)
        self.assertEqual(len({row['path'] for row in rows}), len(rows))
        replacements = []
        for row in rows:
            relative = Path(row['path'])
            parent = CANDIDATE / relative
            copied = STAGE / 'common_candidate' / relative
            self.assertEqual(parent.stat().st_size, row['bytes'])
            self.assertEqual(sha(parent), row['sha256'])
            if copied.read_bytes() != parent.read_bytes():
                replacements.append(row['path'])
        self.assertEqual(replacements, [(POLICY_REL / 'p4_single_file_receipt.py').as_posix()])

    def test_35_runtime_sources_changed_or_not_in_actual_common_lock_rejected(self):
        plan, binding = self.runtime_values()
        locked = deepcopy([*self.source_map.values(), *binding['runtime_common_refs'], *binding['runtime_overlay_refs']])
        for row in locked:
            if row['path'] == 'model-plan.json': row['sha256'] = '2' * 64
        with self.assertRaisesRegex(ValueError, 'common source differs'):
            self.runtime_check(plan, binding, locked)


if __name__ == '__main__':
    unittest.main()
