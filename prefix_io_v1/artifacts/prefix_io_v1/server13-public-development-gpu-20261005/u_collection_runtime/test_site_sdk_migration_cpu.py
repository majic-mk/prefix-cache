"""Six small CPU source/metadata checks; no SDK audit, compiler or CUDA call."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def source_module(path,name):
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),"exec",dont_inherit=True),module.__dict__)
    return module


S=source_module(HERE/"site_sdk_migration.py","_SDK_migration_CPU_fixture")


def helper_source():
    for root in [Path.cwd(),*HERE.parents]:
        for relative in ("artifacts/prefix_io_v1/server09-cuda13-toolchain-v2-20261001/cuda13_sdk_overlay.py",
                         "artifacts/prefix_io_v1_server12_candidates/native_gpu_delivery/recorded_files/98a1bf951d72_cuda13_sdk_overlay.py"):
            path=root/relative
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==S.HELPER_SHA:
                return path
    raise RuntimeError("UNBOUND: exact original CUDA13 helper source required")


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="SDK_migration_CPU_fixture_")
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.audit=dict(old_driver=dict(S.HISTORICAL_DRIVER),SDK_size_drift=[],actual_driver_dynamic_soname=[],
            old_cudart=dict(path=str(self.root.resolve()/".venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13"),
                bytes=704288,sha256="96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"),
            current_driver_paths={path:dict(resolved_path=S.CURRENT_DRIVER["path"],bytes=S.CURRENT_DRIVER["bytes"],
                sha256=S.CURRENT_DRIVER["sha256"],is_symlink=path in S.ALIASES)
                for path in S.ALIASES+(S.CURRENT_DRIVER["path"],)})
        self.audit["current_driver_paths"][S.HISTORICAL_DRIVER["path"]]=dict(resolved_path=S.HISTORICAL_DRIVER["path"],
            bytes=0,sha256=hashlib.sha256(b"").hexdigest(),is_symlink=False)

    def test_new_driver_never_relabels_historical_compile_or_GPU_authority(self):
        value=S._validate_audit(self.audit,self.root)
        self.assertEqual(value["current_driver_ref"],S.CURRENT_DRIVER)
        self.assertEqual(value["historical_compiler_driver_ref"],S.HISTORICAL_DRIVER)
        self.assertFalse(value["compiler_proof_current_driver_matched"])
        self.assertFalse(value["GPU_model_or_JIT_runtime_qualified"])
        self.assertEqual(value["current_driver_compile_link_executions"],0)
        self.assertEqual(value["actual_GPU_operations"],0)

    def test_old_receipt_relabel_unknown_driver_and_extra_authority_fail(self):
        for mutation in (lambda d:d.update(old_driver=dict(S.CURRENT_DRIVER)),
                         lambda d:d["current_driver_paths"][S.ALIASES[0]].update(sha256="0"*64),
                         lambda d:d.update(GPU_qualified=True)):
            changed=deepcopy(self.audit);mutation(changed)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):S._validate_audit(changed,self.root)

    def test_source_loader_uses_actual_bytes_and_rejects_drift(self):
        path=self.root/"fixture.py";raw=b"value=7\n";path.write_bytes(raw)
        row=dict(path="fixture.py",bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        module,actual,_=S._load_source(self.root,{"fixture.py":row},"fixture.py",row["sha256"])
        self.assertEqual(module.value,7);self.assertEqual(actual,raw)
        path.write_bytes(b"value=8\n")
        with self.assertRaisesRegex(ValueError,"actual dependency bytes"):
            S._load_source(self.root,{"fixture.py":row},"fixture.py",row["sha256"])
        with self.assertRaisesRegex(ValueError,"actual fixed migration audit"):
            S.validate_migration_binding(self.root,{},dict(S.MIGRATION_REF,sha256="0"*64))

    def test_original_asset_AST_retains_all_checks_except_one_explicit_driver_join(self):
        path=helper_source();raw=path.read_bytes();helper=source_module(path,"_original_SDK_AST_fixture")
        actual_compile=compile;trees=[]
        def record(value,*args,**kwargs):
            if isinstance(value,ast.Module):trees.append(deepcopy(value))
            return actual_compile(value,*args,**kwargs)
        with mock.patch("builtins.compile",side_effect=record):
            fn=S._derive_asset_verifier(helper,raw)
        self.assertIs(fn.__globals__["verify_tree"],helper.verify_tree)
        expected=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=="verify_assets")
        expected=deepcopy(expected);expected.name="_verify_current_assets_with_historical_compile_evidence"
        target=ast.dump(ast.parse('proof.get("driver_ref") == pin["driver"]',mode="eval").body)
        changed=0
        for node in ast.walk(expected):
            if isinstance(node,ast.Compare) and ast.dump(node)==target:
                node.comparators[0]=ast.Name(id="_actual_historical_compile_driver",ctx=ast.Load());changed+=1
        self.assertEqual(changed,1)
        self.assertEqual(ast.dump(trees[0]),ast.dump(ast.Module(body=[expected],type_ignores=[])))
        self.assertEqual(fn.__globals__["_actual_historical_compile_driver"],S.HISTORICAL_DRIVER)
        with self.assertRaisesRegex(ValueError,"exact immutable original"):
            S._derive_asset_verifier(helper,raw+b"\n#changed")

    def test_original_environment_never_puts_stubs_on_runtime_path(self):
        helper=source_module(helper_source(),"_original_SDK_environment_fixture")
        overlay=self.root/"fixture-cuda13"
        env=helper.planned_environment(overlay,{"PATH":"/usr/bin","LD_LIBRARY_PATH":"/usr/lib"})
        self.assertNotIn("stubs",env["LD_LIBRARY_PATH"])
        with self.assertRaisesRegex(ValueError,"stubs never"):
            helper.planned_environment(overlay,{"LD_LIBRARY_PATH":"/usr/lib/stubs"})

    def test_runtime_V6_only_changes_collection_SDK_dispatch_and_not_finally(self):
        old=ast.parse((HERE/"native_runtime_v5.py").read_bytes());new=ast.parse((HERE/"native_runtime_v6.py").read_bytes())
        functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        o,n=functions(old),functions(new)
        self.assertEqual([name for name in o if ast.dump(o[name])!=ast.dump(n[name])],["execute"])
        old_try=next(x for x in o["execute"].body if isinstance(x,ast.Try))
        new_try=next(x for x in n["execute"].body if isinstance(x,ast.Try))
        self.assertEqual(ast.dump(ast.Module(body=old_try.finalbody,type_ignores=[])),
                         ast.dump(ast.Module(body=new_try.finalbody,type_ignores=[])))
        branch=next(x for x in new_try.body if isinstance(x,ast.If) and ast.unparse(x.test)=="collection is None")
        index=next(i for i,x in enumerate(old_try.body) if isinstance(x,ast.Assign) and
                   isinstance(x.targets[0],ast.Name) and x.targets[0].id=="sdk")
        self.assertEqual(ast.dump(ast.Module(body=branch.body,type_ignores=[])),
                         ast.dump(ast.Module(body=old_try.body[index:index+2],type_ignores=[])))


if __name__=="__main__":unittest.main(verbosity=2)
