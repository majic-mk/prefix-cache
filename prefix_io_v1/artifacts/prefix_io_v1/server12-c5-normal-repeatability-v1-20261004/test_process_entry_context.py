"""CPU-only process-context rejection tests; no native capability fixtures."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_context_cpu_tests_actual_source', HERE/'process_entry_context.py')
P = importlib.util.module_from_spec(spec); sys.modules[spec.name] = P; spec.loader.exec_module(P)


class ProcessContextRejectTests(unittest.TestCase):
    def test_constructor_cannot_promote_fixture(self):
        with self.assertRaisesRegex(ValueError, 'source-verified create'):
            P.ProcessEntryContext()

    def test_constructor_rejects_foreign_token(self):
        with self.assertRaisesRegex(ValueError, 'source-verified create'):
            P.ProcessEntryContext(token=object(), receipt={'PASS': True})

    def test_fake_controller_rejected_before_loader(self):
        with self.assertRaisesRegex(ValueError, 'actual independently frozen'):
            P.ProcessEntryContext.create(object(), HERE, 'anything.json')

    def test_identity_comparison_is_only_metadata(self):
        actual = dict(pid=13, sid=11, process_group=11)
        self.assertIsNone(P.validate_identity(actual, dict(actual)))

    def test_pid_change_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PID/session/process group changed'):
            P.validate_identity(dict(pid=13,sid=11,process_group=11),dict(pid=14,sid=11,process_group=11))

    def test_session_change_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PID/session/process group changed'):
            P.validate_identity(dict(pid=13,sid=11,process_group=11),dict(pid=13,sid=12,process_group=11))

    def test_group_change_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PID/session/process group changed'):
            P.validate_identity(dict(pid=13,sid=11,process_group=11),dict(pid=13,sid=11,process_group=12))

    def test_bool_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PID/session/process group changed'):
            P.validate_identity(dict(pid=1,sid=1,process_group=1),dict(pid=True,sid=1,process_group=1))

    def test_missing_identity_coordinate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PID/session/process group changed'):
            P.validate_identity(dict(pid=1,sid=1,process_group=1),dict(pid=1,sid=1))

    def guard(self):
        return dict(id='explicitly_synthetic_guard_metadata',session_id=11,process_group=11,runner_pid=5,
            label='fixture_only',gpu_uuid='fixture_only',permissions=dict(path='fixture',bytes=1,sha256='0'*64))

    def test_guard_metadata_retains_no_qualification(self):
        value = P.guard_identity(self.guard())
        self.assertNotIn('PASS', value); self.assertNotIn('qualified', value)

    def test_empty_original_reservation_rejected(self):
        value=self.guard();value['id']=''
        with self.assertRaisesRegex(ValueError, 'actual original reservation'):
            P.guard_identity(value)

    def test_bool_guard_process_rejected(self):
        value=self.guard();value['runner_pid']=True
        with self.assertRaisesRegex(ValueError, 'actual original reservation'):
            P.guard_identity(value)

    def test_guard_missing_permission_rejected(self):
        value=self.guard();del value['permissions']
        with self.assertRaisesRegex(ValueError, 'complete original guard identity'):
            P.guard_identity(value)

    def test_every_guard_identity_coordinate_rechecked(self):
        base=self.guard();expected=P.guard_identity(base)
        for key in expected:
            with self.subTest(key=key):
                value=deepcopy(base)
                value[key] = value[key]+1 if type(value[key]) is int else 'wrong' if type(value[key]) is str else dict(value[key],sha256='1'*64)
                with self.assertRaises(ValueError):P.same_guard(expected,value)

    def test_source_flow_has_one_public_loader(self):
        tree=ast.parse((HERE/'process_entry_context.py').read_bytes())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='load_receipt']
        self.assertEqual(len(calls),1)

    def test_reuse_has_actual_metadata_and_current_guard_checks(self):
        tree=ast.parse((HERE/'process_entry_context.py').read_bytes())
        fn=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='configuration')
        attrs={node.func.attr for node in ast.walk(fn) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertTrue({'verify_metadata_configuration','_verify_active_guard_metadata','validate_guard','ref'}<=attrs)
        self.assertNotIn('load_receipt',attrs)

    def test_no_original_function_or_module_patch(self):
        tree=ast.parse((HERE/'process_entry_context.py').read_bytes())
        prohibited=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and node.func.id in ('setattr','exec','eval')]
        self.assertEqual(prohibited,[])
        writes=[target for node in ast.walk(tree) if isinstance(node,(ast.Assign,ast.AnnAssign))
            for target in (node.targets if isinstance(node,ast.Assign) else [node.target])
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='controller']
        self.assertEqual(writes,[])


if __name__=='__main__':unittest.main()
