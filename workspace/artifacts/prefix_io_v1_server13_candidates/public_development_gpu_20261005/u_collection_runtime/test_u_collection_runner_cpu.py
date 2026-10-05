"""Small stdlib fixture tests of new U collection/source dispatch only.

Fixture JSON is never an actual tokenizer/calibration/native receipt. The
source leaves and historical V14 lock/proof are actual unchanged byte copies.
No old suite, model, GPU, native backend, download or external RPC is run.
"""
from __future__ import annotations
import ast
import builtins
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def source_module(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def original_roots():
    roots = [Path(os.environ["PREFIX_IO_PROJECT_ROOT"])] if os.environ.get("PREFIX_IO_PROJECT_ROOT") else []
    roots += [Path.cwd(), *HERE.parents]
    for root in roots:
        for prep, audit in (
            ("artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004",
             "artifacts/prefix_io_v1/server12-natural-source-audit-20261005"),
            ("artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004",
             "artifacts/prefix_io_v1_server12_candidates/natural_source_audit_20261005")):
            if (root/prep/"formal_trace_binding/formal_trace_binding.py").is_file() and (root/audit/"PRERENT_SOURCE_LOCK_V14.json").is_file():
                return root/prep, root/audit
    raise RuntimeError("UNBOUND actual cloned prep/V14 source bytes")


BASE, AUDIT = original_roots()
R = source_module(HERE/"strong_trace_runner_v5.py", "_V5_runner_CPU_test")
B = source_module(HERE/"u_collection_bridge.py", "_V5_collection_CPU_test")


class NewCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.refs = {}
        self.config = dict(phase="development", mode="off", arm="U", run_id="synthetic-CPU-U",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            runtime_ref={"path":"prep/runner/native_runtime_v5.py"}, formal_peer_closed_ref=None,
            off_qualification_ref=None)
        extra = dict(shared_storage_path=str(self.root/"synthetic-storage"),
            prefix_io_parent_admission=dict(run_id=self.config["run_id"]),
            prefix_io_p4_policy=dict(mode="off"))
        self.pair = dict(U=dict(engine=dict(kv_transfer_config=dict(kv_connector_extra_config=extra))),
                         I=dict(synthetic_fixture=True))

    def tearDown(self):
        self.temp.cleanup()

    def source(self, path, original):
        destination=self.root/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(original.read_bytes())
        row=R.ref(self.root,path)
        self.refs[path]=row
        return row

    def put(self,path,value):
        destination=self.root/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(json.dumps(value,sort_keys=True).encode())
        row=R.ref(self.root,path)
        self.refs[path]=row
        return row

    def stage(self):
        self.source("prep/u_collection_bridge/u_collection_bridge.py",HERE/"u_collection_bridge.py")
        self.config["workload_ref"]=self.put("synthetic-manifest.json",dict(schema="natural_trace_workload_v1",synthetic_fixture=True))
        self.config["formal_trace_binding_ref"]=self.put("synthetic-binding.json",dict(schema="development_natural_trace_binding_v1",synthetic_fixture=True))
        self.config["pair_config_ref"]=self.put("synthetic-pair.json",dict(configurations=self.pair,synthetic_fixture=True))
        fields={}
        for key,path in (("collector_ref","runner/bounded_native_full_step_collector_v2.py"),
                         ("original_collector_source_ref","runner/bounded_native_full_step_collector.py"),
                         ("reserve_join_source_ref","activation/control_observation/reserve_join.py"),
                         ("native_journal_source_ref","activation/source/prefix_io_control/p4_native_window_journal.py")):
            fields[key]=self.source("prep/"+path,BASE/path)
        self.source("prep/activation/control_observation/host_control_observer.py",
                    BASE/"activation/control_observation/host_control_observer.py")
        descriptor=dict(schema=B.SCHEMA,gpu_uuid=self.config["gpu_uuid"],common_runtime_domain_sha256=R.common_domain_sha(self.pair),
            workload_ref=self.config["workload_ref"],formal_trace_binding_ref=self.config["formal_trace_binding_ref"],
            pair_config_ref=self.config["pair_config_ref"],table_issued=False,cost_qualified=False,
            ordinary_I_authorized=False,formal_goodput_allowed=False,**fields)
        self.config["collection_ref"]=self.put("synthetic-collection.json",descriptor)
        return descriptor

    def gate(self):
        return R.u_collection_gate(self.root,self.config,self.refs,self.pair)

    def test_new_U_gate_actual_collector_journal_sources_no_old_cal_import(self):
        descriptor=self.stage()
        original_import=builtins.__import__
        forbidden=[]
        def guarded(name,*args,**kwargs):
            if name.startswith(("torch","vllm","py_kvcache","prefix_io_control")) or "activation_request" in name:
                forbidden.append(name)
                raise AssertionError("forbidden CPU import: "+name)
            return original_import(name,*args,**kwargs)
        with mock.patch("builtins.__import__",side_effect=guarded),mock.patch.object(R,"activation_gate_module",side_effect=AssertionError("old private issuer")):
            value=self.gate()
        self.assertEqual(value["descriptor"],descriptor)
        self.assertFalse(value["table_issued"])
        self.assertFalse(value["cost_qualified"])
        self.assertEqual(forbidden,[])
        # Source-only repeated metadata gate does not demand a still-fresh namespace.
        (self.root/"synthetic-storage").mkdir()
        self.assertEqual(self.gate(),value)

    def test_wrong_role_old_activation_or_prior_U_not_collection(self):
        self.stage()
        for phase,mode,arm in (("development","shadow","I"),("development","on","I"),
                               ("effect","off","U"),("effect","on","I"),("qualification","off","U")):
            with self.subTest(role=(phase,mode,arm)),self.assertRaises(ValueError):
                B.gate(self.root,dict(self.config,phase=phase,mode=mode,arm=arm),self.refs,self.pair,driver=R)
        for key,value in (("activation_ref",dict(path="old-calibration")),("off_qualification_ref",dict(path="old-U")),
                          ("formal_peer_closed_ref",dict(path="old-peer")),("authority_ref",dict(path="fake-deadline"))):
            with self.subTest(key=key),self.assertRaises(ValueError):
                B.gate(self.root,dict(self.config,**{key:value}),self.refs,self.pair,driver=R)

    def test_missing_journal_old_device_authority_flag_and_source_drift_rejected(self):
        original=self.stage()
        mutations=[dict(original,gpu_uuid="GPU-old-calibration"),dict(original,cost_qualified=True),
                   {k:v for k,v in original.items() if k!="native_journal_source_ref"},
                   dict(original,calibration_plan_ref={"path":"old-plan"})]
        for i,descriptor in enumerate(mutations):
            self.config["collection_ref"]=self.put("synthetic-mutated-%s.json"%i,descriptor)
            with self.subTest(i=i),self.assertRaises(ValueError):self.gate()
        self.config["collection_ref"]=self.put("synthetic-original.json",original)
        path=self.root/original["native_journal_source_ref"]["path"]
        path.write_bytes(path.read_bytes()+b"\n# synthetic drift\n")
        with self.assertRaisesRegex(ValueError,"byte drift"):self.gate()

    def stage_phase(self):
        self.stage()
        inputs=dict(schema="development_natural_token_id_partition_CPU_binding_v1",service_SLO=None,
            independent_deadline_ref=None,gpu_eligible=False,partition="development",complete_selected_record_count=5,
            records=[dict(request_id=i,prompt_token_ids=[i+1]) for i in range(5)],synthetic_fixture=True)
        calls=[]
        def original_input(*args,**kwargs):
            calls.append(kwargs)
            return deepcopy(inputs)
        f=types.SimpleNamespace(validate_formal_workload=original_input,
            json_leaf=lambda root,row,refs:(row,R.read(R.check_ref(root,row))),
            exact=lambda a,b:type(a) is type(b) and a==b,require=B.require)
        B.bind_phase(f,config=self.config,refs=self.refs,driver=R)
        return f,inputs,calls

    def test_phase_replays_original_full_input_and_never_issues_calibration(self):
        f,inputs,calls=self.stage_phase()
        with mock.patch.object(R,"activation_gate_module",side_effect=AssertionError("old cal issuer")):
            phase=f.validate_formal_phase(self.config,root=self.root,refs=self.refs,pair=self.pair,
                formal_workload=inputs,finite_activation=None,development_replay=None)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]["partition"],"development")
        self.assertTrue(phase["input_bindings_validated"])
        for field in ("cost_qualified","table_issued","ordinary_I_authorized","gpu_eligible","formal_goodput_allowed"):
            self.assertIs(phase[field],False)
        self.assertEqual(phase["actual_GPU_operations"],0)

    def test_phase_rejects_forged_ids_or_qualification_gate(self):
        f,inputs,calls=self.stage_phase()
        wrong=deepcopy(inputs)
        wrong["records"][0]["prompt_token_ids"]=[999]
        with self.assertRaisesRegex(ValueError,"full real partition"):
            f.validate_formal_phase(self.config,root=self.root,refs=self.refs,pair=self.pair,formal_workload=wrong)
        for key in ("finite_activation","development_replay"):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"old private table"):
                f.validate_formal_phase(self.config,root=self.root,refs=self.refs,pair=self.pair,
                    formal_workload=inputs,**{key:dict(synthetic_fixture=True)})

    def test_runner_preflight_dispatch_uses_new_gate_then_original_input_phase(self):
        f,inputs,calls=self.stage_phase()
        inputs["common_runtime_domain_sha256"]=R.common_domain_sha(self.pair)
        gates=dict(config=self.config,refs=self.refs,pair=self.pair,
            formal_original_workload=dict(schema=R.FORMAL_SCHEMA,synthetic_fixture=True),
            workload=dict(records=inputs["records"],synthetic_fixture=True),formal_workload_binding=inputs,
            finite_activation=None,uncalibrated_u_collection=self.gate())
        with mock.patch.object(R,"select_formal_workload",return_value=(gates["workload"],inputs)),\
             mock.patch.object(R,"formal_binding_module",return_value=f),\
             mock.patch.object(R,"activation_gate_module",side_effect=AssertionError("calibration gate")):
            result=R.preflight_formal_runtime(self.root,gates)
        self.assertIs(result,gates["formal_phase_preflight"])
        self.assertIsNone(gates["development_reserve_replay"])
        self.assertEqual(len(calls),1)
        gates["uncalibrated_u_collection"]["cost_qualified"]=True
        with mock.patch.object(R,"select_formal_workload",return_value=(gates["workload"],inputs)),self.assertRaises(ValueError):
            R.preflight_formal_runtime(self.root,gates)

    def migration(self):
        self.stage()
        self.source(B.ANCESTOR_LOCK["path"],AUDIT/"PRERENT_SOURCE_LOCK_V14.json")
        self.source(B.ANCESTOR_PROOF["path"],AUDIT/"PRERENT_SOURCE_PROOF_V14.json")
        ancestor={r["path"]:r for r in R.read(self.root/B.ANCESTOR_LOCK["path"])["files"]}
        # Ancestor proof/lock are references outside their own file denominator.
        self.refs.pop(B.ANCESTOR_LOCK["path"])
        self.refs.pop(B.ANCESTOR_PROOF["path"])
        current={**ancestor,**self.refs}
        required=tuple(self.refs)[:2]
        config=dict(self.config,source_lock_ref=dict(path="synthetic-current-lock.json",bytes=0,sha256="0"*64))
        proof=dict(schema=B.PROOF_SCHEMA,status="PASS_INHERITED_V14_PLUS_TARGETED_CPU_BYTES",
            ancestor_source_lock_ref=B.ANCESTOR_LOCK,ancestor_source_proof_ref=B.ANCESTOR_PROOF,
            source_lock_ref=config["source_lock_ref"],source_count=len(current),ancestor_source_count=5014,
            targeted_source_refs=[current[p] for p in sorted(set(current)-set(ancestor))],
            required_source_refs=[current[p] for p in required],actual_gpu_runs=0,
            full_source_verified=False,current_host_whole_source_hash_performed=False)
        driver=types.SimpleNamespace(REQUIRED=required,read=R.read,check_ref=mock.Mock(side_effect=R.check_ref))
        return config,current,proof,driver

    def test_actual_V14_ancestry_only_targeted_leaves_hashed_honest_labels(self):
        config,current,proof,driver=self.migration()
        result=B.verify_migrated_source_proof(self.root,config,current,proof,driver=driver)
        self.assertEqual(result["ancestor_source_count"],5014)
        self.assertFalse(result["full_source_verified"])
        self.assertFalse(result["current_host_whole_source_hash_performed"])
        expected=[B.ANCESTOR_LOCK,B.ANCESTOR_PROOF]+proof["targeted_source_refs"]+proof["required_source_refs"]
        self.assertEqual([call.args[1] for call in driver.check_ref.call_args_list],expected)
        self.assertFalse(any("models/" in call.args[1]["path"] for call in driver.check_ref.call_args_list))

    def test_migration_cannot_lose_alter_or_replace_ancestor_rows_or_claim_full(self):
        config,current,proof,driver=self.migration()
        inherited=next(p for p in current if p not in self.refs)
        for changed in ({p:r for p,r in current.items() if p!=inherited},
                        dict(current,**{inherited:dict(current[inherited],sha256="0"*64)})):
            with self.assertRaisesRegex(ValueError,"inherited source/model/SDK"):
                B.verify_migrated_source_proof(self.root,config,changed,proof,driver=driver)
        for key,value in (("full_source_verified",True),("current_host_whole_source_hash_performed",True),
                          ("actual_gpu_runs",True),("targeted_source_refs",proof["targeted_source_refs"][:-1])):
            with self.subTest(key=key),self.assertRaises(ValueError):
                B.verify_migrated_source_proof(self.root,config,current,dict(proof,**{key:value}),driver=driver)
        with self.assertRaises(ValueError):
            B.verify_migrated_source_proof(self.root,dict(config,phase="effect"),current,proof,driver=driver)

    def test_collection_role_no_standing_on_I_and_original_drive_guard_preserved(self):
        self.stage()
        self.assertEqual(R.formal_role(self.config),"development")
        for mode,arm in (("shadow","I"),("on","I")):
            with self.assertRaises(ValueError):R.formal_role(dict(self.config,mode=mode,arm=arm))
        old=AUDIT/"development_runtime_wiring/strong_trace_runner_v4.py"
        # Server deploy copies that stable folder intact alongside canonical prep.
        for path in (old,BASE/"runner/strong_trace_runner_v4.py"):
            if path.is_file():old=path;break
        old_ast=ast.parse(old.read_bytes())
        new_ast=ast.parse((HERE/"strong_trace_runner_v5.py").read_bytes())
        for name in ("drive_original_engine","verify_guard","guard_command","verify_formal_off_prerequisite",
                     "check_phase","source_rows","common_domain_sha"):
            a=next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
