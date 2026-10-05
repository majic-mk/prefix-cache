"""Three new CPU SDK-dispatch tests only; all returned asset summaries are mocks.

Uses actual frozen collector/SDK/driver-audit source leaves for static closure.
No actual SDK asset preflight/library/framework/GPU/RPC/old test suite is run.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def source_module(path,name):
    module=types.ModuleType(name)
    module.__file__=str(path.resolve())
    sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),"exec",dont_inherit=True),module.__dict__)
    return module


# Import fixture constructors only, never discover/run the previous nine tests.
OLD=source_module(HERE/"test_u_collection_runner_cpu.py","_old_fixture_constructor_only")
R=source_module(HERE/"strong_trace_runner_v6.py","_V6_sdk_wire_CPU")
B=source_module(HERE/"u_collection_bridge_v2.py","_V6_sdk_bridge_CPU")


class SDKWiringTests(unittest.TestCase):
    def setUp(self):
        self.fixture=OLD.NewCollectionTests(methodName="test_wrong_role_old_activation_or_prior_U_not_collection")
        self.fixture.setUp()
        self.descriptor=self.fixture.stage()
        self.fixture.config["runtime_ref"]={"path":"prep/runner/native_runtime_v6.py"}
        self.fixture.source("prep/u_collection_bridge/u_collection_bridge_v2.py",HERE/"u_collection_bridge_v2.py")
        self.descriptor["site_sdk_adapter_ref"]=self.fixture.source("prep/sdk_migration/site_sdk_migration.py",HERE/"site_sdk_migration.py")
        candidates=[HERE.parent/"ACTUAL_DRIVER_MIGRATION_AUDIT_01.json"]
        candidates += [root/B.SDK_MIGRATION_AUDIT["path"] for root in [Path.cwd(),*HERE.parents]]
        actual=next((p for p in candidates if p.is_file()),None)
        if actual is None:raise RuntimeError("UNBOUND actual cloned driver audit bytes")
        self.descriptor["site_sdk_migration_ref"]=self.fixture.source(B.SDK_MIGRATION_AUDIT["path"],actual)
        self.fixture.config["collection_ref"]=self.fixture.put("synthetic-SDK-collection.json",self.descriptor)

    def tearDown(self):
        self.fixture.tearDown()

    def gate(self):
        f=self.fixture
        return R.u_collection_gate(f.root,f.config,f.refs,f.pair)

    def test_sdk_descriptor_sources_closed_and_old_adapter_or_audit_rejected(self):
        f=self.fixture
        self.assertEqual(self.gate()["descriptor"]["site_sdk_migration_ref"],B.SDK_MIGRATION_AUDIT)
        for key,value in (("site_sdk_adapter_ref",dict(self.descriptor["site_sdk_adapter_ref"],sha256="0"*64)),
                          ("site_sdk_migration_ref",dict(B.SDK_MIGRATION_AUDIT,bytes=0))):
            changed=deepcopy(self.descriptor);changed[key]=value
            f.config["collection_ref"]=f.put("synthetic-wrong-SDK.json",changed)
            with self.subTest(key=key),self.assertRaises(ValueError):self.gate()
        missing={k:v for k,v in self.descriptor.items() if k!="site_sdk_migration_ref"}
        f.config["collection_ref"]=f.put("synthetic-no-SDK.json",missing)
        with self.assertRaises(ValueError):self.gate()

    def test_production_CPU_sdk_preflight_calls_pure_API_and_rejects_false_GPU_or_compile_claims(self):
        f=self.fixture
        gate=self.gate()
        summary=dict(schema="existing_CUDA13_current_driver_CPU_asset_preflight_v1",CPU_assets_verified=True,
            compiler_proof_current_driver_matched=False,current_driver_compile_link_executions=0,
            GPU_model_or_JIT_runtime_qualified=False,shared_objects_loaded=False,overlay_created=False,
            actual_GPU_operations=0)
        sdk=types.SimpleNamespace(preflight_site_sdk=mock.Mock(return_value=summary))
        with mock.patch.object(R,"load",return_value=sdk) as loading:
            self.assertEqual(R.preflight_collection_sdk(f.root,gate,f.refs),summary)
            sdk.preflight_site_sdk.assert_called_once_with(f.root,f.refs,B.SDK_MIGRATION_AUDIT)
            self.assertEqual(loading.call_args.args[1],self.descriptor["site_sdk_adapter_ref"])
        for key,value in (("compiler_proof_current_driver_matched",True),("current_driver_compile_link_executions",True),
                          ("GPU_model_or_JIT_runtime_qualified",True),("shared_objects_loaded",True),
                          ("overlay_created",True),("actual_GPU_operations",True)):
            sdk.preflight_site_sdk.return_value=dict(summary,**{key:value})
            with self.subTest(key=key),mock.patch.object(R,"load",return_value=sdk),self.assertRaises(ValueError):
                R.preflight_collection_sdk(f.root,gate,f.refs)
        with mock.patch.object(R,"load",side_effect=AssertionError("old route SDK changed")):
            self.assertIsNone(R.preflight_collection_sdk(f.root,None,f.refs))

    def test_old_drive_guard_source_and_I_qualification_helpers_unchanged(self):
        old=ast.parse((HERE/"strong_trace_runner_v5.py").read_bytes())
        new=ast.parse((HERE/"strong_trace_runner_v6.py").read_bytes())
        for name in ("drive_original_engine","verify_guard","guard_command","verify_formal_off_prerequisite",
                     "check_phase","source_rows","common_domain_sha","activation_gate_module"):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        path=HERE/"site_sdk_migration.py"
        source=path.read_bytes()
        import hashlib
        self.assertEqual(hashlib.sha256(source).hexdigest(),B.SITE_SDK_ADAPTER_SHA)


if __name__=="__main__":
    unittest.main(verbosity=2)
