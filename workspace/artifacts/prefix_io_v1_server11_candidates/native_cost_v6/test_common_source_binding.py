"""CPU contract checks for recalibration of a changed common implementation."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import types
import unittest
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v6_common_wrapper_cpu', HERE/'run_native_cost_experiment.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class CommonSourceTests(unittest.TestCase):
    def test_actual_module_path_binding_rejects_old_source_even_with_new_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            source=root/M.REACTOR;source.parent.mkdir(parents=True);source.write_text('# CPU fixture',encoding='utf-8')
            module=types.ModuleType('py_kvcache.reactor');module.__file__=str(source)
            owner_type=type('IoReactor',(),{'__module__':module.__name__})
            owner=owner_type();owner._prefix_p4_bridge=None;owner._max_accepted_parents=8
            calls=[];runtime=types.SimpleNamespace(original_method=lambda *args:calls.append(args[1]))
            refs={M.REACTOR:M.file_ref(root,M.REACTOR)}
            with patch.dict(sys.modules,{module.__name__:module}):
                value=M.native_source_binding(root,refs,owner,runtime)
                self.assertEqual(calls,['_run'])
                self.assertEqual(value['actual_source_ref'],refs[M.REACTOR])
                module.__file__=str(root/M.LEGACY_COLLECTOR_REACTOR_KEY)
                with self.assertRaisesRegex(ValueError,'actual common reactor source path'):
                    M.native_source_binding(root,refs,owner,runtime)

    def test_collector_alias_is_local_view_of_actual_overlay(self):
        actual = dict(path=M.REACTOR,bytes=123,sha256='a'*64)
        old = dict(path=M.LEGACY_COLLECTOR_REACTOR_KEY,bytes=99,sha256='b'*64)
        refs = {M.REACTOR:actual,M.LEGACY_COLLECTOR_REACTOR_KEY:old}
        view = M.collector_source_view(refs)
        self.assertEqual(view[M.LEGACY_COLLECTOR_REACTOR_KEY],actual)
        self.assertEqual(refs[M.LEGACY_COLLECTOR_REACTOR_KEY],old)
        self.assertEqual(actual['path'],M.REACTOR)

    def test_collector_alias_cannot_promote_legacy_reference(self):
        with self.assertRaisesRegex(ValueError,'actual collector'):
            M.collector_source_view({M.REACTOR:dict(path=M.LEGACY_COLLECTOR_REACTOR_KEY)})

    def test_actual_factory_keeps_strategy_off_and_parent_cap_eight(self):
        tree = ast.parse((HERE/'run_native_cost_experiment.py').read_bytes())
        function = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='factory')
        calls=[]
        namespace=dict(coordinators=[],require=M.require,original_factory=lambda **kw:(calls.append(kw) or object()),
            journal=object(),sink=object(),LABEL=M.LABEL,MAX_ACCEPTED_PARENTS=M.MAX_ACCEPTED_PARENTS)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(HERE),'exec'),namespace)
        namespace['factory'](file_store='existing')
        self.assertEqual(calls[0]['max_accepted_parents'],8)
        self.assertIsNone(calls[0]['p4_bridge'])
        self.assertEqual(calls[0]['progress_run_id'],M.LABEL)
        with self.assertRaises(ValueError):namespace['factory']()

    def test_real_collector_installed_with_new_view_then_verified_all_frames(self):
        tree=ast.parse((HERE/'run_native_cost_experiment.py').read_bytes())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and
            isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
            n.func.value.id=='collector' and n.func.attr=='install')
        view=next(k.value for k in call.keywords if k.arg=='refs')
        self.assertIsInstance(view,ast.Call);self.assertEqual(view.func.id,'collector_source_view')
        raw=(HERE/'run_native_cost_experiment.py').read_text(encoding='utf-8')
        self.assertIn('adapter_native_source_sha256_before',raw)
        self.assertIn('adapter_native_source_sha256_after',raw)
        self.assertIn('len(scalar_adapter.frames)==128',raw)
        self.assertIn('active_capture.observer.scalar._adapter is scalar_adapter',raw)

    def test_original_cost_contract_unchanged(self):
        self.assertEqual(hashlib.sha256((HERE/'native_conditional_cost.py').read_bytes()).hexdigest(),
            'e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291')

    def test_original_v5_harness_is_unchanged(self):
        project=os.environ.get('SERVER11_AUTHOR_SOURCE_ROOT')
        path=(Path(project)/'artifacts/prefix_io_v1/server11-native-cost-v5-20261003/run_native_cost_experiment.py'
              if project else HERE.parent/'native_cost_v5/run_native_cost_experiment.py')
        # Pin the actual historical source; this is a drift check, not GPU evidence.
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            'feb1e3dead5da65141b7f0faadbb857bdcf654e0c5b2291c67ca5b1913bc53dc')


if __name__=='__main__':unittest.main(verbosity=2)
