"""CPU-only execution of the exact new runtime finally prefix, no backend."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest

PATH=Path(__file__).with_name("g3_calibration_runtime_metrics_v2.py")
RAW=PATH.read_bytes()
assert len(RAW)==29978 and hashlib.sha256(RAW).hexdigest()=="0c04287325fd84c31aaea35dc6ff7431229511a04f3ab71bf9aa96a2ce83f1cc"
TREE=ast.parse(RAW)
FUNCTION=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="execute_original_acquisition")
TRY=next(n for n in FUNCTION.body if isinstance(n,ast.Try))
# Include the actual following receipt assignment, to verify cleanup failures
# continue through the original finally instead of replacing the main error.
NODES=TRY.finalbody[:4]
assert len(NODES)==4 and isinstance(NODES[0],ast.Expr) and isinstance(NODES[1],ast.If) and isinstance(NODES[2],ast.If) and isinstance(NODES[3],ast.Assign)
CODE=compile(ast.Module(body=NODES,type_ignores=[]),str(PATH),"exec",dont_inherit=True)
SPEC=importlib.util.spec_from_file_location("_cpu_exact_metrics_cleanup_runtime",PATH)
R=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class Cleanup(unittest.TestCase):
    def namespace(self,handle):
        faults=[];calls=[]
        taps=types.SimpleNamespace(restore=lambda:calls.append("restore"),fault=lambda error:faults.append(error),requests=[])
        name="_cpu_exact_metrics_cleanup_fixture"
        helper=types.SimpleNamespace(__name__=name)
        sys.modules[name]=helper
        namespace=dict(taps=taps,stats_handle=handle,stats_compat=helper,scalar_copy=R.scalar_copy,
            receipt={"common_stats_dictionary_fix":{"installed":True,"restored":False,"witness":None}},sys=sys)
        return namespace,faults,calls,name

    def test_successful_dict_detach_restores_and_removes_helper(self):
        state={"installed":True};calls=[]
        def detach():
            calls.append("detach");state["installed"]=False
            return dict(state)
        def witness():calls.append("witness");return dict(state)
        namespace,faults,restores,name=self.namespace(types.SimpleNamespace(detach=detach,witness=witness))
        try:
            exec(CODE,namespace)
            self.assertEqual(calls,["detach","witness"])
            self.assertEqual(restores,["restore"])
            self.assertEqual(faults,[])
            self.assertIs(namespace["receipt"]["common_stats_dictionary_fix"]["restored"],True)
            self.assertIs(namespace["receipt"]["common_stats_dictionary_fix"]["witness"]["installed"],False)
            self.assertEqual(namespace["receipt"]["requests"],[])
            self.assertNotIn(name,sys.modules)
        finally:sys.modules.pop(name,None)

    def test_detach_error_preserves_original_error_and_later_cleanup(self):
        original=ValueError("original acquisition fixture failure")
        secondary=RuntimeError("foreign override fixture")
        def detach():raise secondary
        def witness():raise AssertionError("witness must not hide detach failure")
        namespace,faults,restores,name=self.namespace(types.SimpleNamespace(detach=detach,witness=witness))
        try:
            try:
                try:raise original
                finally:exec(CODE,namespace)
            except BaseException as actual:self.assertIs(actual,original)
            else:self.fail("original error was lost")
            self.assertEqual(faults,[secondary])
            self.assertEqual(restores,["restore"])
            self.assertIs(namespace["receipt"]["common_stats_dictionary_fix"]["restored"],False)
            self.assertEqual(namespace["receipt"]["requests"],[])
            self.assertNotIn(name,sys.modules)
        finally:sys.modules.pop(name,None)


if __name__=="__main__":unittest.main(verbosity=2)
