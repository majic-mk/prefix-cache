"""CPU route/source/restore tests. No model, fake CostTable or GPU evidence.

Spies test call contracts and AST branches only. Positive spy JSON is explicitly
CPU-only and is never written as a production reserve, frame or GPU receipt.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PREP_BASE = HERE.parent.parent
PREP = PREP_BASE / 'gpu_prerental_preparation_20261004'
if not PREP.is_dir():
    PREP = PREP_BASE / 'server12-gpu-prerental-preparation-20261004'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load('_CPU_only_formal_native_runtime_v3', HERE / 'native_runtime_v3.py')
OLD = load('_CPU_only_original_native_runtime_v2', PREP / 'runner/native_runtime_v2.py')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def gates(role=('effect', 'off', 'U'), *, formal=True):
    phase, mode, arm = role
    config = dict(phase=phase, mode=mode, arm=arm, run_id='CPU_ONLY_ROUTE_NOT_GPU',
                  runtime_ref=dict(path='CPU/runner/native_runtime_v3.py'), gpu_uuid='CPU_NOT_A_GPU')
    if formal:
        config['formal_trace_binding_ref'] = dict(path='CPU_NOT_REAL_FORMAL_INPUT')
    result = dict(config=config, refs={},
                  pair={'U': {'engine': {'kv_transfer_config': {'kv_connector_extra_config': {
                      'prefix_io_p4_policy': {'mode': 'off', 'cost_table': None}}}}}},
                  workload=dict(schema='natural_trace_workload_v1' if formal else
                                'controlled_original_p3_qualification_workload_v1'),
                  finite_activation=dict(phase=phase, descriptor={}, independent_budget={}))
    return result


def metadata_driver(value, *, fake_role=None, eligible=False, attached=True):
    def preflight(root, actual):
        summary = dict(schema='formal_trace_phase_CPU_preflight_v1',
                       role=list(fake_role or value), gpu_eligible=eligible,
                       formal_goodput_allowed=False, CPU_TEST_SPY_ONLY=True)
        if attached:
            actual['formal_phase_preflight'] = summary
        return summary
    return NS(require=require, preflight_formal_runtime=preflight)


def reserve_call_fixture():
    item = gates()
    finite = item['finite_activation']
    refs = item['refs']
    for key in ('development_reserve_ref', 'development_budget_ref', 'reserve_verification_ref',
                'reserve_join_source_ref', 'development_plan_ref', 'development_guard_ref',
                'development_observation_ref', 'development_native_result_ref'):
        row = dict(path='CPU_CALL_CONTRACT/' + key + '.json', bytes=0, sha256='0' * 64)
        finite['descriptor'][key] = row
        refs[row['path']] = row
    activation = dict(path='CPU/runner/activation_request.py', bytes=0, sha256='0' * 64)
    refs[activation['path']] = activation
    item['config']['activation_ref'] = dict(path='CPU_CALL_CONTRACT_ONLY_ACTIVATION')
    finite.update(activation_source_path='CPU_CALL_CONTRACT_NOT_RUNTIME', calibration_plan={},
                  runtime_refs=refs, expected_common_domain_sha256='CPU_CALL_CONTRACT_NOT_COMMON_DOMAIN')
    return item


class FormalRuntimeCPU(unittest.TestCase):
    def test_all_four_formal_roles_have_strict_metadata_gate_and_U_no_controller(self):
        roles = [('development', 'off', 'U'), ('development', 'shadow', 'I'),
                 ('effect', 'off', 'U'), ('effect', 'on', 'I')]
        for role in roles:
            item = gates(role)
            route = N.preflight_runtime_route(Path('.'), item, driver=metadata_driver(role))
            self.assertTrue(route['formal'])
            self.assertTrue(route['require_private_table'])
            self.assertEqual(route['install_finite_controller'], role[2] == 'I')
            self.assertFalse(item['formal_phase_preflight']['gpu_eligible'])

    def test_old_qualification_and_tableless_shadow_do_not_call_formal_replay(self):
        def forbidden(*args, **kwargs):
            self.fail('original qualification/shadow must not invoke formal contract')
        for role in [('qualification', 'off', 'U'), ('shadow', 'shadow', 'I')]:
            route = N.preflight_runtime_route(Path('.'), gates(role, formal=False),
                                             driver=NS(require=require, preflight_formal_runtime=forbidden))
            self.assertFalse(route['formal'])
            self.assertFalse(route['require_private_table'])
            self.assertFalse(route['install_finite_controller'])

    def test_formal_missing_or_promoted_CPU_gate_rejects_before_output_or_model(self):
        for bad in ('missing', 'eligible', 'wrong_role', 'detached'):
            item = gates()
            if bad == 'missing':
                item['config'].pop('formal_trace_binding_ref')
            driver = metadata_driver(('effect', 'off', 'U'),
                eligible=bad == 'eligible', fake_role=('qualification', 'off', 'U') if bad == 'wrong_role' else None,
                attached=bad != 'detached')
            # No output, storage or reservation ID is supplied. The actual
            # execute function must reject at the CPU route gate first.
            with self.assertRaises(ValueError):
                N.execute(Path('.'), item, dict(label=item['config']['run_id']), driver=driver)

    def test_formal_U_never_accepts_qualification_rows_or_an_on_policy(self):
        for role in [('development', 'off', 'U'), ('effect', 'off', 'U')]:
            with self.assertRaisesRegex(ValueError, 'qualification relabeling'):
                N.preflight_runtime_route(Path('.'), gates(role, formal=False), driver=NS(require=require))
            item = gates(role)
            item['pair']['U']['engine']['kv_transfer_config']['kv_connector_extra_config']['prefix_io_p4_policy']['mode'] = 'interference'
            with self.assertRaisesRegex(ValueError, 'unchanged U off'):
                N.preflight_runtime_route(Path('.'), item, driver=metadata_driver(role))

    def test_U_controller_branch_returns_before_any_module_import_or_startup_load(self):
        def forbidden(*args, **kwargs):
            self.fail('U must not load or construct an I controller')
        for role in [('development', 'off', 'U'), ('effect', 'off', 'U')]:
            with patch.object(N.importlib, 'import_module', forbidden):
                actual = N.prepare_finite_controller(Path('.'), gates(role), native_module=None,
                    table=None, issuer=None, binding_module=None, driver=NS(require=require, load=forbidden))
            self.assertIsNone(actual)

    def test_I_controller_retains_original_args_and_development_observation_only(self):
        for role in [('development', 'shadow', 'I'), ('effect', 'on', 'I')]:
            calls = []
            item = gates(role)
            finite = item['finite_activation']
            finite['descriptor']['startup_ref'] = dict(path='CPU_CALL_CONTRACT_STARTUP')
            finite['independent_budget']['internal_step_budget_ns'] = 123
            subset = (('CPU_CALL_CONTRACT_ONLY_NO_CELL',),)
            finite['reserve_covered_cell_signatures'] = subset if role[0] == 'effect' else None
            marker = object()
            def constructor(**kwargs):
                calls.append(kwargs)
                return marker
            driver = NS(require=require, load=lambda *args: NS(FiniteStartup=constructor))
            # None stands for a call argument only. No CostTable or private
            # issuer is fabricated, and no constructor from production is run.
            with patch.object(N.importlib, 'import_module', lambda name: 'CPU_MODULE_CALL_SPY'):
                actual = N.prepare_finite_controller(Path('.'), item, native_module=None, table=None,
                    issuer=None, binding_module=None, driver=driver)
            self.assertIs(actual, marker)
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call['budget_ns'], 123)
            self.assertEqual(call['observation_only'], role[0] == 'development')
            self.assertIs(call['validate_drained'], N.validate_drained_snapshot)
            self.assertEqual(call['reserve_covered_cell_signatures'], subset if role[0] == 'effect' else None)

    def test_missing_actual_development_refs_reject_before_loading_policy(self):
        def forbidden(*args, **kwargs):
            self.fail('missing actual reserve must reject before private policy source import')
        for key in ('development_reserve_ref', 'development_guard_ref', 'reserve_verification_ref'):
            item = reserve_call_fixture()
            item['finite_activation']['descriptor'].pop(key)
            with self.assertRaisesRegex(ValueError, 'missing actual development prerequisite'):
                N.preflight_formal_reserve_replay(Path('.'), item,
                    driver=NS(require=require, check_ref=lambda *args: None, load=forbidden))

    def test_source_ref_drift_rejects_before_CPU_private_issuer(self):
        item = reserve_call_fixture()
        item['refs'][item['finite_activation']['descriptor']['development_guard_ref']['path']] = dict(path='CPU_DRIFT')
        with self.assertRaisesRegex(ValueError, 'missing actual development prerequisite'):
            N.preflight_formal_reserve_replay(Path('.'), item,
                driver=NS(require=require, check_ref=lambda *args: None,
                          load=lambda *args: self.fail('drift before import')))

    def test_CPU_issue_call_cleanup_runs_on_success_and_failure_without_table_escape(self):
        for fail in (False, True):
            item = reserve_call_fixture()
            before_path, before_modules = list(sys.path), set(sys.modules)
            calls = []
            def verify(*args, **kwargs):
                calls.append('original_verify')
                return deepcopy(item['finite_activation'])
            def issue(*args, **kwargs):
                calls.append('original_issue_call_contract_only')
                sys.modules['prefix_io_control'] = ModuleType('prefix_io_control')
                sys.modules['prefix_io_control.CPU_CALL_SPY'] = ModuleType('prefix_io_control.CPU_CALL_SPY')
                if fail:
                    raise ValueError('CPU_call_contract_failure')
                return None, None
            def replay(*args, **kwargs):
                calls.append('original_replay_call_contract_only')
                return dict(CPU_CALL_SPY_ONLY=True, GPU_QUALIFICATION=False, returned_private_table=False)
            driver = NS(require=require, check_ref=lambda *args: None,
                        load=lambda *args: NS(verify=verify, issue=issue))
            with patch.object(N, '_replay_original_reserve', replay):
                if fail:
                    with self.assertRaisesRegex(ValueError, 'CPU_call_contract_failure'):
                        N.preflight_formal_reserve_replay(Path('.'), item, driver=driver)
                else:
                    result = N.preflight_formal_reserve_replay(Path('.'), item, driver=driver)
                    self.assertTrue(result['CPU_CALL_SPY_ONLY'])
                    self.assertFalse(result['GPU_QUALIFICATION'])
                    self.assertFalse(result['returned_private_table'])
            self.assertEqual(sys.path, before_path)
            self.assertFalse(any(name == 'prefix_io_control' or name.startswith('prefix_io_control.')
                                 for name in set(sys.modules) - before_modules))
            self.assertEqual(calls[0], 'original_verify')
            self.assertEqual(calls[1], 'original_issue_call_contract_only')

    def test_existing_private_package_cannot_be_replaced_by_CPU_preflight(self):
        item = reserve_call_fixture()
        module = ModuleType('prefix_io_control')
        driver = NS(require=require, check_ref=lambda *args: None,
                    load=lambda *args: NS(verify=lambda *args, **kwargs: deepcopy(item['finite_activation']),
                                         issue=lambda *args, **kwargs: self.fail('existing private package must reject')))
        with patch.dict(sys.modules, {'prefix_io_control': module}):
            with self.assertRaisesRegex(ValueError, 'fresh sole CPU private'):
                N.preflight_formal_reserve_replay(Path('.'), item, driver=driver)
            self.assertIs(sys.modules['prefix_io_control'], module)

    def test_UNBOUND_or_UNKNOWN_capture_never_calls_original_verifier(self):
        with patch.object(N, '_load_original_reserve_join', lambda *args, **kwargs: self.fail('unknown before import')):
            for capture in (None, {}, {'valid': False}, {'valid': True, 'frames': []}):
                with self.assertRaisesRegex(ValueError, 'whole-stream CUDA observation UNKNOWN'):
                    N.validate_formal_capture(Path('.'), gates(), capture, {}, source_ref={}, driver=NS(require=require))

    def test_UNBOUND_formal_off_parent_with_flags_only_rejects(self):
        prerequisite = dict(status='PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE',
                            phase='development', arm='U', mode='off',
                            original_engine_shutdown_returned=True, native_tail_drained=True, os_session_drained=True)
        with self.assertRaisesRegex(ValueError, 'closed actual formal native peer evidence'):
            N.verify_formal_off_native_prerequisite(Path('.'), prerequisite,
                config=dict(phase='development', arm='U', mode='off'), refs={},
                driver=NS(require=require, read=lambda *args: self.fail('missing before raw import')))

    def test_development_off_prerequisite_cannot_accept_a_closed_I_peer(self):
        item = dict(phase='effect', arm='I', mode='on')
        with patch.object(N, 'verify_formal_peer_native_prerequisite',
                          lambda *args, **kwargs: self.fail('development U prerequisite cannot delegate I')):
            with self.assertRaisesRegex(ValueError, 'Uoff native prerequisite only'):
                N.verify_formal_off_native_prerequisite(Path('.'), item, config=item, refs={}, driver=NS(require=require))

    def test_generic_effect_I_peer_needs_real_raw_refs_and_original_private_replay(self):
        item = dict(phase='effect', arm='I', mode='on', original_engine_shutdown_returned=True,
                    native_tail_drained=True, os_session_drained=True)
        with self.assertRaisesRegex(ValueError, 'closed actual formal native peer evidence'):
            N.verify_formal_peer_native_prerequisite(Path('.'), item, config=item, refs={},
                driver=NS(require=require, read=lambda *args: self.fail('flags-only peer cannot import policy')))
        tree = ast.parse((HERE / 'native_runtime_v3.py').read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == 'verify_formal_peer_native_prerequisite')
        names = [node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Name)]
        self.assertEqual(names.count('preflight_formal_reserve_replay'), 1)
        self.assertEqual(names.count('validate_formal_capture'), 1)
        self.assertEqual(names.count('verify_original_tail'), 1)
        source = ast.unparse(function)
        self.assertIn('finite_running_identity_verified', source)
        self.assertIn('finite_startup_restored_after_original_shutdown', source)
        self.assertIn('owner_thread_install', source)

    def test_actual_execute_U_I_flags_and_status_branch_are_distinct(self):
        tree = ast.parse((HERE / 'native_runtime_v3.py').read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
        branches = [node for node in ast.walk(function) if isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].value, ast.Name) and
                    node.targets[0].value.id == 'result' and isinstance(node.targets[0].slice, ast.Constant) and
                    node.targets[0].slice.value in ('strategy_runtime', 'status') and
                    (node.targets[0].slice.value == 'strategy_runtime' or isinstance(node.value, ast.IfExp))]
        self.assertEqual(len(branches), 2)
        for role in [('development', 'off', 'U'), ('effect', 'off', 'U'),
                     ('development', 'shadow', 'I'), ('effect', 'on', 'I')]:
            phase, mode, arm = role
            namespace = dict(config=dict(phase=phase, mode=mode, arm=arm), result={}, bridge=None,
                             finite_startup=None if arm == 'U' else object(), qualified_table=object())
            exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(branches), type_ignores=[])),
                         'CPU_only_actual_strategy_status_AST', 'exec', dont_inherit=True), namespace)
            report = namespace['result']
            self.assertEqual(report['strategy_runtime']['actual_I_strategy_activated'], phase == 'effect' and arm == 'I')
            self.assertEqual(report['strategy_runtime']['real_interference_table_installed'], arm == 'I')
            if arm == 'U':
                self.assertIn('U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE', report['status'])

    def test_original_planner_collector_tail_and_finally_remain_exact_AST(self):
        old_tree = ast.parse((PREP / 'runner/native_runtime_v2.py').read_bytes())
        new_tree = ast.parse((HERE / 'native_runtime_v3.py').read_bytes())
        old = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
        new = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual([name for name in old if ast.dump(old[name]) != ast.dump(new[name])], ['execute'])
        old_try = next(node for node in old['execute'].body if isinstance(node, ast.Try))
        new_try = next(node for node in new['execute'].body if isinstance(node, ast.Try))
        self.assertEqual([ast.dump(node) for node in old_try.finalbody], [ast.dump(node) for node in new_try.finalbody])
        self.assertEqual(hashlib.sha256((PREP / 'runner/native_runtime_v2.py').read_bytes()).hexdigest(),
                         '26b6b8b2e704c09d066963013d4124edf19098dfe1a7085d9de3535a9cddd1c5')
        body_calls = [node for node in ast.walk(new['execute']) if isinstance(node, ast.Call)]
        drive = [node for node in body_calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'drive_original_engine']
        self.assertEqual(len(drive), 1)
        self.assertFalse(any(name == 'torch' or name.startswith(('torch.', 'vllm', 'py_kvcache')) for name in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
