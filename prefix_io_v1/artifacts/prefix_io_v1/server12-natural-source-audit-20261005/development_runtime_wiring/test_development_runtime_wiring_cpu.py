"""Portable stdlib composition/rejection tests; no actual GPU/receipt fixtures.

Direct python -B -I -S invocation works locally and on the cloned server.
Source files are loaded from bytes, never via an ambient site/helper cache.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def original_base():
    candidates = []
    if os.environ.get("PREFIX_IO_PROJECT_ROOT"):
        candidates.append(Path(os.environ["PREFIX_IO_PROJECT_ROOT"]))
    candidates.extend([Path.cwd(), *HERE.parents])
    names = ("artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004",
             "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004")
    for root in candidates:
        for name in names:
            base = root / name
            if (base / "formal_trace_binding/formal_trace_binding.py").is_file():
                return base.resolve()
    raise RuntimeError("UNBOUND: actual cloned original prep source directory required")


BASE = original_base()


def from_source(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


B = from_source(HERE / "development_diagnostic_bridge.py", "_development_bridge_CPU_test")
R = from_source(HERE / "strong_trace_runner_v4.py", "_development_runner_CPU_test")
V = from_source(HERE / "native_runtime_v4.py", "_development_runtime_CPU_test")


def node(path, name):
    return next(item for item in ast.parse(path.read_bytes()).body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)


def json_leaf(root, path, value):
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(json.dumps(value, sort_keys=True).encode())
    return R.ref(root, path)


def fresh_module(root, name, filename, original):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original.read_bytes())
    row = R.ref(root, filename)
    return R.load(root, row, name), row


def metadata_only_manifest(namespace):
    records = [dict(request_id=i, prompt_sha256=hashlib.sha256(("fixture-"+str(i)).encode()).hexdigest(),
                    prompt_token_ids=[i+1], scheduled_ns=i, max_tokens=128, min_tokens=128,
                    seed=0, split=part, prefix_family="fixture-source-only-"+str(i))
               for i, part in enumerate(namespace.PARTITIONS)]
    value = dict(schema="natural_trace_workload_v1", origin="frozen_actual_existing_trace_no_new_gpu_outcomes",
        dataset_sha256="1"*64, ordered_prompt_digest="2"*64, author_trace_sha256=namespace.AUTHOR_TRACE_SHA,
        author_common_sha256=namespace.AUTHOR_COMMON_SHA, tokenizer_receipt_digest="3"*64,
        declaration_digest="4"*64, model_manifest_sha256="5"*64, records=records,
        partition_counts={name:1 for name in namespace.PARTITIONS}, max_concurrency=1,
        arrival_rate=None, schedule_seed=0, schedule_origin="unchanged_author_build_global_specs",
        recorded_natural_arrival_claim=False, no_prefix_injection=True, no_per_request_cache_reset=True,
        no_request_drops_or_reorder=True, initial_cache_state=namespace.INITIAL,
        cost_domain_covered=False, gpu_effect_qualified=False, gpu_operations=0)
    value["workload_sha256"] = namespace.digest(value)
    return value


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.refs = {}
        self.config = dict(phase="development", mode="off", arm="U", run_id="fixture-U",
            runtime_ref=dict(path="prep/runner/native_runtime_v4.py"),
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000000")

    def tearDown(self):
        self.temp.cleanup()

    def put_source(self, relative, origin):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(origin.read_bytes())
        row = R.ref(self.root, relative)
        self.refs[relative] = row
        return row

    def put_json(self, relative, value):
        row = json_leaf(self.root, relative, value)
        self.refs[relative] = row
        return row

    def stage_modules(self, schema=B.BINDING_SCHEMA):
        self.put_source("prep/formal_trace_binding/formal_trace_binding.py", BASE/"formal_trace_binding/formal_trace_binding.py")
        self.put_source("prep/formal_runtime_bridge/namespace_bridge.py", BASE/"formal_runtime_bridge/namespace_bridge.py")
        self.put_source("prep/runner/activation_request.py", BASE/"runner/activation_request.py")
        self.put_source("prep/development_runtime_bridge/development_diagnostic_bridge.py", HERE/"development_diagnostic_bridge.py")
        self.config["formal_trace_binding_ref"] = self.put_json("fixture-binding.json", dict(schema=schema))
        return self.config

    def test_actual_runner_loads_scoped_development_module(self):
        self.stage_modules()
        module = R.formal_binding_module(self.root, self.config, self.refs)
        self.assertEqual(module.BINDING_FIELDS, set(B_BASE_FIELDS())-{"independent_deadline_ref"})
        self.assertEqual(module.validate_formal_workload.__code__.co_filename,
                         str(self.root/"prep/development_runtime_bridge/development_diagnostic_bridge.py"))
        self.assertEqual(module.load_pure.__code__.co_filename,
                         str(self.root/"prep/development_runtime_bridge/development_diagnostic_bridge.py"))
        with self.assertRaisesRegex(ValueError, "full prospective development partition"):
            module.validate_formal_workload(None,root=self.root,workload_ref=None,binding_ref=None,
                refs=self.refs,pair=None,partition="evaluation")

    def test_development_descriptor_effect_off_and_on_rejected(self):
        for mode, arm in (("off","U"),("on","I")):
            self.stage_modules()
            config=dict(self.config, phase="effect",mode=mode,arm=arm)
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,"no evaluation/effect/ordinary I"):
                R.formal_binding_module(self.root,config,self.refs)

    def test_development_descriptor_on_and_qualification_rejected(self):
        self.stage_modules()
        for phase,mode,arm in (("development","on","I"),("qualification","off","U")):
            config=dict(self.config,phase=phase,mode=mode,arm=arm)
            with self.subTest(phase=phase),self.assertRaises(ValueError):
                R.activation_gate_module(self.root,config,self.refs)

    def test_effect_original_descriptor_keeps_original_F_and_activation(self):
        self.stage_modules("formal_natural_trace_binding_v1")
        config=dict(self.config,phase="effect",mode="on",arm="I")
        module=R.formal_binding_module(self.root,config,self.refs)
        self.assertIn("independent_deadline_ref",module.BINDING_FIELDS)
        self.assertEqual(module.validate_formal_workload.__code__.co_filename,
                         str(self.root/"prep/formal_trace_binding/formal_trace_binding.py"))
        self.assertEqual(module.load_pure.__code__.co_filename,
                         str(self.root/"prep/development_runtime_bridge/development_diagnostic_bridge.py"))
        activation=R.activation_gate_module(self.root,config,self.refs)
        self.assertEqual(activation.SCHEMA,"finite_gpu_cell_activation_request_v1")
        self.assertIn("independent_deadline_ref",activation.FIELDS)
        self.assertEqual(activation.verify.__code__.co_filename,str(self.root/"prep/runner/activation_request.py"))

    def test_development_activation_retains_exact_original_issue_function(self):
        module,row=fresh_module(self.root,"_original_activation_exact_issue_test","prep/runner/activation_request.py",
                                BASE/"runner/activation_request.py")
        self.refs[row["path"]]=row
        wrapper=B.bind_activation(module,root=self.root,source_ref=row,config=self.config,refs=self.refs,driver=R)
        self.assertIs(wrapper.issue,module.issue)
        self.assertEqual(ast.dump(node(BASE/"runner/activation_request.py","issue")),
                         ast.dump(node(BASE/"runner/activation_request.py","issue")))
        with self.assertRaisesRegex(ValueError,"cannot be reused by effect"):
            wrapper.verify(self.root,None,refs=self.refs,pair=None,gpu_uuid=self.config["gpu_uuid"],driver=R,phase="effect")

    def test_U_no_preview_budget_and_shadow_null_refused(self):
        self.config["pair_config_ref"]={"path":"fixture-pair.json","bytes":0,"sha256":"0"*64}
        pair={"I":{"engine":{"kv_transfer_config":{"kv_connector_extra_config":{
            "prefix_io_p4_policy":{"internal_step_budget_ns":None}}}}}}
        budget=B.diagnostic_budget(self.config,pair)
        self.assertIsNone(budget["internal_step_budget_ns"])
        self.assertFalse(budget["ordinary_I_authorized"])
        self.assertFalse(budget["effect_budget_reuse_allowed"])
        with self.assertRaisesRegex(ValueError,"UNBOUND_SHADOW_ENGINEERING_PREVIEW_BUDGET"):
            B.diagnostic_budget(dict(self.config,mode="shadow",arm="I"),pair)

    def test_shadow_positive_engineering_setting_never_SLO_or_ordinary_I(self):
        cfg=dict(self.config,mode="shadow",arm="I",pair_config_ref={"path":"fixture-pair.json"})
        pair={"I":{"engine":{"kv_transfer_config":{"kv_connector_extra_config":{
            "prefix_io_p4_policy":{"internal_step_budget_ns":123}}}}}}
        budget=B.diagnostic_budget(cfg,pair)
        self.assertEqual(budget["internal_step_budget_ns"],123)
        self.assertTrue(budget["observation_only"])
        self.assertIsNone(budget["service_SLO"])
        for flag in ("ordinary_I_authorized","preview_is_service_safety_claim","effect_budget_reuse_allowed"):
            self.assertFalse(budget[flag])
        for val in (True,0,-1,1.5,"123"):
            pair["I"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["prefix_io_p4_policy"]["internal_step_budget_ns"]=val
            with self.subTest(val=val),self.assertRaises(ValueError):B.diagnostic_budget(cfg,pair)

    def test_external_SLO_and_deadline_cannot_enter_adapter(self):
        for key in ("service_SLO","independent_deadline_ref","authority_ref","full_control_window_deadline_ns"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                B.require_role(dict(self.config,**{key:1}))

    def test_factory_binding_closes_real_bytes_but_does_not_declare_ready(self):
        leaf=self.put_json("fixture-not-a-tokenizer-receipt.json",{"synthetic_fixture":True})
        inputs={key:leaf for key in B.INPUT_REF_FIELDS}
        result=B.prepare_binding_descriptor(self.root,inputs,refs=self.refs,config=self.config,driver=R)
        self.assertEqual(result["descriptor"]["schema"],B.BINDING_SCHEMA)
        self.assertNotIn("independent_deadline_ref",result["descriptor"])
        self.assertFalse(result["ready_for_GPU"])
        self.assertTrue(result["original_full_input_replay_required"])
        with self.assertRaisesRegex(ValueError,"exact real natural input"):
            B.prepare_binding_descriptor(self.root,dict(inputs,independent_deadline_ref=None),
                refs=self.refs,config=self.config,driver=R)
        inputs["dataset_ref"]={"path":"missing-fixture","bytes":1,"sha256":"0"*64}
        with self.assertRaisesRegex(ValueError,"UNBOUND actual input leaf"):
            B.prepare_binding_descriptor(self.root,inputs,refs=self.refs,config=self.config,driver=R)

    def test_factory_activation_no_deadline_no_mock_reserve_no_ready(self):
        leaf=self.put_json("fixture-not-GPU-calibration.json",{"synthetic_fixture":True})
        inputs={key:(None if key in B.POST_DEVELOPMENT else leaf) for key in B.ACTIVATION_REF_FIELDS}
        result=B.prepare_activation_descriptor(self.root,inputs,refs=self.refs,config=self.config,driver=R)
        self.assertEqual(result["descriptor"]["schema"],B.ACTIVATION_SCHEMA)
        self.assertNotIn("independent_deadline_ref",result["descriptor"])
        self.assertFalse(result["ready_for_GPU"])
        inputs["development_reserve_ref"]=leaf
        with self.assertRaisesRegex(ValueError,"cannot claim a measured reserve"):
            B.prepare_activation_descriptor(self.root,inputs,refs=self.refs,config=self.config,driver=R)

    def test_budget_extra_authority_flags_and_origin_refused(self):
        cfg=dict(self.config,pair_config_ref={"path":"fixture-pair"})
        budget=B.diagnostic_budget(cfg,{})
        for mutation in ({"ordinary_I_authorized":True},{"effect_budget_reuse_allowed":True},
                         {"origin":"independent_service_deadline"},{"independent_deadline_ref":{}}):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                B.verify_diagnostic_budget(cfg,dict(budget,**mutation))

    def test_source_only_runner_ignores_valid_malicious_pyc(self):
        file=self.root/"fixture_helper.py"
        good=b"VALUE = 'GOOD'\n"
        bad=b"VALUE = 'EVIL'\n"
        file.write_bytes(bad)
        py_compile.compile(str(file),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        stamp=file.stat().st_mtime_ns
        file.write_bytes(good);os.utime(file,ns=(stamp,stamp))
        spec=importlib.util.spec_from_file_location("_fixture_default_cache_probe",file)
        default=importlib.util.module_from_spec(spec);spec.loader.exec_module(default)
        self.assertEqual(default.VALUE,"EVIL")
        row=R.ref(self.root,"fixture_helper.py")
        actual=R.load(self.root,row,"_fixture_actual_V4_source_helper")
        self.assertEqual(actual.VALUE,"GOOD")
        self.assertTrue(Path(importlib.util.cache_from_source(str(file))).is_file())

    def test_new_private_module_names_unique_even_if_clock_repeats(self):
        self.stage_modules()
        with mock.patch.object(R.time,"monotonic_ns",return_value=123):
            first=R.development_bridge_module(self.root,self.config,self.refs)
            second=R.development_bridge_module(self.root,self.config,self.refs)
        self.assertNotEqual(first.__name__,second.__name__)

    def test_source_only_F_pure_loader_ignores_valid_malicious_pyc(self):
        formal,row=fresh_module(self.root,"_fixture_F_pure_source_cache_test","formal.py",BASE/"formal_trace_binding/formal_trace_binding.py")
        file=self.root/"fixture_protocol.py"
        file.write_bytes(b"VALUE = 'EVIL'\n")
        py_compile.compile(str(file),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        stamp=file.stat().st_mtime_ns
        file.write_bytes(b"VALUE = 'GOOD'\n");os.utime(file,ns=(stamp,stamp))
        leaf=R.ref(self.root,"fixture_protocol.py")
        actual=B.source_only_pure_loader(formal)(self.root,leaf,{leaf["path"]:leaf},leaf["sha256"])
        self.assertEqual(actual.VALUE,"GOOD")

    def test_source_only_pure_loader_rejects_backend_import_before_execution(self):
        formal,row=fresh_module(self.root,"_fixture_F_backend_import_test","formal.py",BASE/"formal_trace_binding/formal_trace_binding.py")
        file=self.root/"fixture_protocol.py";file.write_text("import torch\n",encoding="utf-8")
        leaf=R.ref(self.root,"fixture_protocol.py")
        with self.assertRaisesRegex(ValueError,"pure helper import boundary"):
            B.source_only_pure_loader(formal)(self.root,leaf,{leaf["path"]:leaf},leaf["sha256"])

    def test_runner_detects_source_mutation_during_execution(self):
        path=self.root/"fixture_drift.py"
        path.write_text("from pathlib import Path\nPath(__file__).write_text('drift')\n",encoding="utf-8")
        leaf=R.ref(self.root,"fixture_drift.py")
        with self.assertRaisesRegex(ValueError,"source/evidence byte drift"):
            R.load(self.root,leaf,"_fixture_V4_postload_byte_drift")

    def test_natural_input_mandatory_and_real_leaf_closure_kept(self):
        self.stage_modules()
        formal=R.formal_binding_module(self.root,self.config,self.refs)
        manifest=metadata_only_manifest(formal)
        mref=self.put_json("fixture-manifest.json",manifest)
        missing={"path":"missing-fixture.json","bytes":1,"sha256":"0"*64}
        desc={key:missing for key in formal.BINDING_FIELDS-{"schema"}}
        desc.update(schema=B.BINDING_SCHEMA,manifest_ref=mref)
        bref=self.put_json("fixture-complete-binding.json",desc)
        with self.assertRaisesRegex(ValueError,"source leaf outside frozen closure"):
            formal.validate_formal_workload(manifest,root=self.root,workload_ref=mref,binding_ref=bref,
                refs=self.refs,pair=None,partition="development")

    def test_closed_peer_namespace_adapter_not_replaced(self):
        self.stage_modules()
        formal=R.formal_binding_module(self.root,self.config,self.refs)
        self.assertEqual(formal.validate_namespace.__code__.co_filename,
                         str(self.root/"prep/formal_runtime_bridge/namespace_bridge.py"))
        original=node(BASE/"formal_runtime_bridge/namespace_bridge.py","verify_closed_peer_document")
        self.assertIn("validate_guard",ast.unparse(original))
        self.assertIn("verify_formal_peer_native_prerequisite",ast.unparse(original))
        self.assertIn("actual_run_config_ref",ast.unparse(original))

    def test_U_runtime_does_not_construct_optional_controller(self):
        driver=types.SimpleNamespace(require=R.require,load=lambda *a,**k:(_ for _ in ()).throw(AssertionError("U must not load startup")))
        result=V.prepare_finite_controller(self.root,{"config":self.config,"finite_activation":{}},
            native_module=None,table=None,issuer=None,binding_module=None,driver=driver)
        self.assertIsNone(result)

    def test_development_on_runtime_controller_rejected_before_loading(self):
        driver=types.SimpleNamespace(require=R.require,load=lambda *a,**k:(_ for _ in ()).throw(AssertionError("no startup")))
        with self.assertRaisesRegex(ValueError,"same finite development-shadow"):
            V.prepare_finite_controller(self.root,{"config":dict(self.config,arm="I",mode="on"),"finite_activation":{}},
                native_module=None,table=None,issuer=None,binding_module=None,driver=driver)

    def test_original_drive_guard_and_qualification_functions_AST_unchanged(self):
        for name in ("drive_original_engine","verify_guard","validate_workload","check_phase","common_domain_sha"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(node(BASE/"runner/strong_trace_runner_v3.py",name)),
                                 ast.dump(node(HERE/"strong_trace_runner_v4.py",name)))

    def test_runtime_original_finally_and_effect_native_verifiers_AST_unchanged(self):
        old=node(BASE/"runner/native_runtime_v3.py","execute")
        new=node(HERE/"native_runtime_v4.py","execute")
        old_final=next(n for n in old.body if isinstance(n,ast.Try)).finalbody
        new_final=next(n for n in new.body if isinstance(n,ast.Try)).finalbody
        self.assertEqual(ast.dump(ast.Module(body=old_final,type_ignores=[])),ast.dump(ast.Module(body=new_final,type_ignores=[])))
        for name in ("preflight_formal_reserve_replay","verify_formal_peer_native_prerequisite","validate_formal_capture",
                     "verify_original_tail","prepare_finite_controller"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(node(BASE/"runner/native_runtime_v3.py",name)),
                                 ast.dump(node(HERE/"native_runtime_v4.py",name)))

    def test_CPU_preflight_modules_import_no_model_or_native_backend(self):
        original_import=builtins.__import__
        blocked={"torch","vllm","py_kvcache","tokenizers","transformers","prefix_io_control"}
        def bound(name,*a,**kw):
            if name.split(".")[0] in blocked:raise AssertionError("backend imported during CPU source preflight")
            return original_import(name,*a,**kw)
        with mock.patch("builtins.__import__",side_effect=bound):
            self.stage_modules()
            R.formal_binding_module(self.root,self.config,self.refs)
            R.activation_gate_module(self.root,self.config,self.refs)

    def test_original_calibration_guard_rejects_GPU_boolean_fixture(self):
        validator=from_source(BASE/"activation/native_conditional_cost.py","_wiring_original_guard_negative")
        with self.assertRaisesRegex(ValueError,"actual GPU guard/natural session drain"):
            validator.validate_guard({"GPU_verified":True},gpu_uuid="GPU-fixture",job_id="fixture",wrapper_path="fixture.py")

    def test_original_private_mock_cost_table_never_qualifies(self):
        control=BASE/"activation/source/prefix_io_control"
        saved={name:module for name,module in sys.modules.items()
               if name=="prefix_io_control" or name.startswith("prefix_io_control.")}
        for name in saved:sys.modules.pop(name)
        try:
            package=types.ModuleType("prefix_io_control")
            package.__path__=[str(control)];package.__file__=str(control/"__init__.py")
            sys.modules["prefix_io_control"]=package
            dispatch=from_source(control/"dispatch_budget.py","prefix_io_control.dispatch_budget")
            cost=from_source(control/"p4_cost_table.py","prefix_io_control.p4_cost_table")
            issuer=from_source(control/"gpu_cell_issuer.py","prefix_io_control.gpu_cell_issuer")
            package.p4_cost_table=cost
            signature=("GPU-fixture","a"*64,"b"*64,"c"*64,1,1,0,527,917504)
            cell=cost.CostCell(signature,dispatch.ZERO,"ssd_read",917504,"existing_io_plus_delta",100,20,1)
            table=cost.CostTable((cell,),scope="mock_only",source_sha256="d"*64)
            self.assertFalse(table.production_qualified)
            self.assertIsNone(issuer.qualified_identity(table))
            self.assertIsNone(table.lookup(signature,dispatch.ZERO,"ssd_read",917504,execution="production"))
            with self.assertRaises(ValueError):cost.CostTable((cell,),scope="gpu_verified_exact_cells",source_sha256="d"*64)
            with self.assertRaises(ValueError):cost._from_verified_gpu_capability(object())
            self.assertFalse(issuer._ISSUED)
        finally:
            for name in list(sys.modules):
                if name=="prefix_io_control" or name.startswith("prefix_io_control."):sys.modules.pop(name)
            sys.modules.update(saved)

    def test_original_native_capture_rejects_old_host_elapsed_clock(self):
        validator=from_source(BASE/"activation/native_conditional_cost.py","_wiring_original_clock_negative")
        prepared=dict(native_step_ordinal=0,batch=1,active_decode=0,prefill_tokens=1,context_length=128,
            step_kind="prefill",context_basis="pre_computed_tokens",rows=[dict(request_id="fixture",pre_context=128,
                prompt_tokens=129,scheduled_tokens=1)],input_seq_lens_from_cpu_inputs=[129])
        frame=dict(native_step_ordinal=0,start_ns=10,end_ns=20,intended_timing_scope="full_decode_step",
                   gpu_elapsed_ns=None,existing_io=None,new_io=None,prepared=prepared,outputs=[["fixture",[0]]])
        witness=dict(native_step_ordinal=0,start_record_before_ns=1,start_record_after_ns=2,start_completed_query_ns=3,
                     end_record_before_ns=20,end_record_after_ns=21,end_completed_query_ns=30,
                     event_elapsed_source="time.perf_counter_ns",gpu_elapsed_ns=1)
        capture=dict(scope="server11_full_step_native_capture_v1",run_id="fixture",origin="native_gpu_recording",
                     valid=True,frames=[frame]+[{}]*127,event_witnesses=[witness]+[{}]*127,failures=[],pending_event_pairs=0,
                     open_event_pair=False,no_added_synchronization=True,cross_clock_absolute_mapping=False,selected_offsets=[16])
        with self.assertRaisesRegex(ValueError,"actual CUDA elapsed source required"):
            validator.validate_capture(capture,run_id="fixture",request_id="fixture",output_ids=list(range(128)),
                prompt_tokens=129,measured_offset=16,warmup_offsets=[1])


def B_BASE_FIELDS():
    # Source-only load of the original exact input constants; no real receipt.
    return from_source(BASE/"formal_trace_binding/formal_trace_binding.py","_wiring_original_F_constants").BINDING_FIELDS


if __name__ == "__main__":
    unittest.main()
