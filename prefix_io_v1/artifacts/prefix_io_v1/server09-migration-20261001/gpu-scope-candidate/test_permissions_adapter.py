"""CPU contract tests. All authorization files are synthetic temporary fixtures.
No backend import, GPU probe, subprocess launch, or real budget update is allowed.
"""
import contextlib,copy,hashlib,importlib.abc,importlib.util,io,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/"baseline"))
if os.name=="nt":
    sys.modules.setdefault("fcntl",types.SimpleNamespace(LOCK_EX=2,LOCK_NB=4,flock=lambda *a:None))
class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in ("torch","vllm","py_kvcache"):
            raise AssertionError("Forbidden backend import during CPU test: "+fullname)
        return None
BLOCKER=NoGPUImports()
sys.meta_path.insert(0,BLOCKER)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
q=load_module("candidate_qualifier",BASE/"candidate/qualify_p4_native_gpu.py")
g=load_module("candidate_guard",BASE/"candidate/run_gpu_stage.py")
oldq=load_module("baseline_qualifier",BASE/"baseline/qualify_p4_native_gpu.py")
import yaml
GPU="GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"
OLDGPU="GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8"
EXPLICIT="experiments/prefix_io_v1/configs/permissions.server09.yaml"
SCOPE="artifacts/server09/TEST_ONLY_GPU_SCOPE.json"
LOCK="artifacts/server09/TEST_ONLY_SOURCE_LOCK.json"

class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.primary=self.root/"experiments/prefix_io_v1/runs";self.primary.mkdir(parents=True)
        self.ledger=self.root/"experiments/prefix_io_v1/gpu-budget-ledger.json"
        self.write_ledger(dict(gpu_wall_seconds=16520.562886646483,active_reservation=None,events=[]))
        self.permission=dict(schema_version=1,allow_gpu_runs=True,max_gpu_hours=8,
            max_model_download_gib=20,approved_gpu_ids=[GPU],
            approved_experiment_root=str(self.primary),approved_dependency_root=str(self.root),
            allow_model_downloads=False,allow_driver_or_system_changes=False,
            allow_shared_data_deletion=False,allow_payment=False,allow_new_cloud_rental=False)
        old=copy.deepcopy(self.permission);old["approved_gpu_ids"]=[OLDGPU]
        old["allow_model_downloads"]=True
        self.write_yaml(q.PERMISSION,old);self.write_yaml(EXPLICIT,self.permission)
        self.write_json(LOCK,dict(files=[]))
        self.args=types.SimpleNamespace(mode="off",name="server09-test-off",scope_record=SCOPE,
                                         source_lock=LOCK,permissions_path=EXPLICIT)
        self.scope=dict(schema_version=1,status="USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION",
            allow_gpu_initialization=True,allow_gpu_runs=True,allowed_modes=["off","shadow"],
            new_executor=False,allow_model_downloads=False,gpu_uuid=GPU,
            base_permissions=self.ref(EXPLICIT),source_lock=LOCK,
            source_lock_sha256=self.ref(LOCK)["sha256"],qualification_context=q.qualification_context(),
            fixture_only=True)
        self.write_json(SCOPE,self.scope)
        self.source_mock=patch.object(q,"source_refs",side_effect=lambda root,lock:{EXPLICIT:self.ref(EXPLICIT)})
        self.source_mock.start();self.addCleanup(self.source_mock.stop)
    def write_yaml(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(yaml.safe_dump(value,sort_keys=False))
    def write_json(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def write_ledger(self,value):self.ledger.write_text(json.dumps(value))
    def ref(self,name):
        raw=(self.root/name).read_bytes();return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def gates(self):
        return q.scope_source_gates(self.root,self.args)[0]
    def test_missing_permission_option_keeps_default_and_original_argv(self):
        bare=types.SimpleNamespace()
        self.assertEqual(q.permission_relative(bare),q.PERMISSION)
        new=q.preview(self.root,"off","test")
        old=oldq.preview(self.root,"off","test")
        for k in ("child_argv","guard_argv","launch_argv","shell_preview"):
            self.assertEqual(new[k],old[k])
        self.assertNotIn("--permissions-path",new["child_argv"])
    def test_guard_default_reads_original_file_without_override_restrictions(self):
        permission,ref=g.permission_source(self.root)
        self.assertEqual(permission["approved_gpu_ids"],[OLDGPU])
        self.assertTrue(permission["allow_model_downloads"])
        self.assertEqual(ref,self.ref(q.PERMISSION))
    def test_legitimate_explicit_permission_path_reads_real_bytes(self):
        permission,ref=g.permission_source(self.root,EXPLICIT)
        self.assertEqual(permission,self.permission)
        self.assertEqual(ref,self.ref(EXPLICIT))
        gates,reason=q.scope_source_gates(self.root,self.args)
        self.assertEqual(reason,"SCOPE_SOURCE_VALIDATED_WITHOUT_GPU")
        self.assertEqual(gates["scope"]["base_permissions"],ref)
    def test_explicit_path_propagates_to_launch_guard_and_child(self):
        view=q.preview(self.root,"off","test",scope_record=SCOPE,permissions_path=EXPLICIT)
        for key in ("launch_argv","guard_argv","child_argv"):
            argv=view[key];i=argv.index("--permissions-path")
            self.assertEqual(argv[i+1],EXPLICIT)
        completed=types.SimpleNamespace(returncode=0)
        with patch("subprocess.run",return_value=completed) as run:
            self.assertEqual(q.invoke_existing_guard(self.root,self.args,self.gates()),0)
        command=run.call_args.args[0]
        self.assertEqual(command.count("--permissions-path"),2)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],GPU)
    def test_permission_paths_reject_escape_absolute_and_malformed(self):
        for relative in ("../outside.yaml",str(self.root/"outside.yaml"),"a//b","a/../b","a\\b","./a"):
            with self.subTest(relative=relative):
                with self.assertRaises((ValueError,RuntimeError)):g.permission_source(self.root,relative)
                with self.assertRaises(ValueError):q.safe(self.root,relative)
    def test_symlink_permission_file_rejected(self):
        link=self.root/"linked.yaml"
        try:link.symlink_to(self.root/EXPLICIT)
        except OSError as exc:self.skipTest("Local OS cannot create symlink: "+str(exc))
        with self.assertRaises(RuntimeError):g.permission_source(self.root,"linked.yaml")
        with self.assertRaises(ValueError):q.safe(self.root,"linked.yaml")
    def test_symlink_permission_ancestor_rejected(self):
        link=self.root/"linked-configs"
        try:link.symlink_to(self.root/"experiments/prefix_io_v1/configs",target_is_directory=True)
        except OSError as exc:self.skipTest("Local OS cannot create symlink: "+str(exc))
        with self.assertRaises(RuntimeError):g.permission_source(self.root,"linked-configs/permissions.server09.yaml")
        with self.assertRaises(ValueError):q.safe(self.root,"linked-configs/permissions.server09.yaml")
    def test_scope_binding_cannot_name_old_file_with_new_bytes(self):
        self.scope["base_permissions"]["path"]=q.PERMISSION;self.write_json(SCOPE,self.scope)
        with self.assertRaisesRegex(ValueError,"scope/permissions binding"):self.gates()
    def test_permission_changed_after_scope_freeze_fails(self):
        self.permission["max_gpu_hours"]=7;self.write_yaml(EXPLICIT,self.permission)
        with self.assertRaisesRegex(ValueError,"scope/permissions binding"):self.gates()
    def test_effective_permission_must_be_source_locked(self):
        with patch.object(q,"source_refs",return_value={}):
            with self.assertRaisesRegex(ValueError,"effective permissions"):self.gates()
    def test_wrong_device_scope_fails_before_backend(self):
        self.scope["gpu_uuid"]=OLDGPU;self.write_json(SCOPE,self.scope)
        with self.assertRaisesRegex(ValueError,"same approved GPU scope"):self.gates()
    def test_override_cannot_raise_budget(self):
        self.permission["max_gpu_hours"]=9;self.write_yaml(EXPLICIT,self.permission)
        with self.assertRaisesRegex(RuntimeError,"exceeds base"):g.permission_source(self.root,EXPLICIT)
    def test_override_cannot_enable_download_system_delete_payment_or_rental(self):
        for key in ("allow_model_downloads","allow_driver_or_system_changes",
                    "allow_shared_data_deletion","allow_payment","allow_new_cloud_rental"):
            changed=copy.deepcopy(self.permission);changed[key]=True;self.write_yaml(EXPLICIT,changed)
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError,key):g.permission_source(self.root,EXPLICIT)
    def test_override_cannot_change_roots_or_borrow_auxiliary_grant(self):
        for key in ("approved_experiment_root","approved_dependency_root"):
            changed=copy.deepcopy(self.permission);changed[key]=str(self.root/"other");self.write_yaml(EXPLICIT,changed)
            with self.assertRaisesRegex(RuntimeError,"root differs"):g.permission_source(self.root,EXPLICIT)
        changed=copy.deepcopy(self.permission)
        changed["approved_auxiliary_storage"]=dict(root="/tmp/other",authorization_record="old-grant")
        self.write_yaml(EXPLICIT,changed)
        with self.assertRaisesRegex(RuntimeError,"PRIMARY-only"):g.permission_source(self.root,EXPLICIT)
    def test_launch_budget_and_storage_checks_remain_same_single_ledger(self):
        gates,reason=q.launch_gates(self.root,self.args,lambda p:types.SimpleNamespace(free=12*1024**3))
        self.assertEqual(reason,"READY_FOR_AUTHORIZED_OLD_GUARD")
        self.assertEqual(gates["out"],self.primary/self.args.name/"details")
        with self.assertRaisesRegex(ValueError,"storage"):
            q.launch_gates(self.root,self.args,lambda p:types.SimpleNamespace(free=q.FLOOR))
        self.write_ledger(dict(gpu_wall_seconds=28700,active_reservation=None))
        with self.assertRaisesRegex(ValueError,"budget"):q.launch_gates(self.root,self.args,lambda p:types.SimpleNamespace(free=12*1024**3))
    def test_execution_requires_same_child_scope_and_actual_guard_binding(self):
        view=q.preview(self.root,self.args.mode,self.args.name,LOCK,SCOPE,EXPLICIT)
        active=dict(label=self.args.name,gpu_uuid=GPU,session_id=123,seconds_limit=q.TIME_LIMIT,
                    reserved_seconds=q.TIME_LIMIT+20,command=view["child_argv"],permissions=self.ref(EXPLICIT))
        self.write_ledger(dict(gpu_wall_seconds=16520,active_reservation=active))
        with patch.object(q.os,"getsid",return_value=123,create=True),patch.dict(os.environ,
                CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1"):
            gates,reason=q.execution_gates(self.root,self.args)
            self.assertEqual(reason,"READY_FOR_AUTHORIZED_NATIVE_GPU")
            self.assertEqual(gates["permit"]["approved_gpu_ids"],[GPU])
            active["permissions"]["path"]=q.PERMISSION
            self.write_ledger(dict(gpu_wall_seconds=16520,active_reservation=active))
            with self.assertRaisesRegex(ValueError,"effective-permissions"):q.execution_gates(self.root,self.args)
    def test_execution_rejects_child_permission_argument_change(self):
        view=q.preview(self.root,self.args.mode,self.args.name,LOCK,SCOPE,EXPLICIT)
        child=view["child_argv"][:-2]
        active=dict(label=self.args.name,gpu_uuid=GPU,session_id=123,seconds_limit=q.TIME_LIMIT,
                    reserved_seconds=q.TIME_LIMIT+20,command=child,permissions=self.ref(EXPLICIT))
        self.write_ledger(dict(gpu_wall_seconds=16520,active_reservation=active))
        with patch.object(q.os,"getsid",return_value=123,create=True):
            with self.assertRaisesRegex(ValueError,"child argv"):q.execution_gates(self.root,self.args)
    def test_explicit_storage_uses_scope_primary_without_legacy_aux(self):
        gates=self.gates();gates["out"]=self.primary/self.args.name/"details"
        with patch("shutil.disk_usage",return_value=types.SimpleNamespace(free=12*1024**3)):
            result=q.storage_contract(self.root,self.args,gates)
        self.assertFalse(result["auxiliary_scope_used"])
        self.assertEqual(result["permissions"],self.ref(EXPLICIT))
        gates["out"]=self.root/"aux/details"
        with self.assertRaisesRegex(ValueError,"PRIMARY"):q.storage_contract(self.root,self.args,gates)
    def test_default_storage_still_uses_original_preflight(self):
        args=types.SimpleNamespace(name="old")
        sentinel=object()
        fake=types.SimpleNamespace(preflight=lambda *a,**kw:sentinel)
        with patch.dict(sys.modules,experiment_storage=fake):
            self.assertIs(q.storage_contract(self.root,args,{"out":self.primary/"old/details"}),sentinel)
    def test_missing_gpu_scope_launch_fails_before_any_probe_or_guard(self):
        before=self.ledger.read_bytes()
        with patch.object(q,"invoke_existing_guard",side_effect=AssertionError("guard called")),\
             patch.object(q,"qualify",side_effect=AssertionError("GPU qualification called")),\
             contextlib.redirect_stdout(io.StringIO()) as output:
            rc=q.main(["--project",str(self.root),"--mode","off","--name","server09-test-denied",
                       "--permissions-path",EXPLICIT,"--scope-record","missing-scope.json",
                       "--source-lock",LOCK,"--launch"])
        self.assertEqual(rc,78);self.assertIn("BLOCKED_CPU_ONLY",output.getvalue())
        self.assertEqual(before,self.ledger.read_bytes());self.assertFalse((self.primary/"server09-test-denied").exists())
    def test_invalid_permission_guard_cli_fails_before_process_or_ledger_mutation(self):
        before=self.ledger.read_bytes()
        fake_script=self.root/"experiments/prefix_io_v1/scripts/run_gpu_stage.py"
        with patch.object(g,"__file__",str(fake_script)),\
             patch.object(sys,"argv",["guard","--permissions-path","../escape","--label","test","--seconds","1","--","unused"]),\
             patch.object(g.subprocess,"Popen",side_effect=AssertionError("process launched")):
            with self.assertRaisesRegex(RuntimeError,"permission path"):g.main()
        self.assertEqual(before,self.ledger.read_bytes());self.assertFalse((self.primary/"test").exists())

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary=dict(schema_version=1,status="PASS" if result.wasSuccessful() else "FAIL",
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
        actual_GPU_runs=0,GPU_probes=0,backend_imports_forbidden=True,
        remote_project_changes=0,real_budget_mutations=0,
        note="Synthetic CPU contract fixtures; not GPU or performance qualification.")
    (BASE/"CPU_TEST_RESULT.json").write_text(json.dumps(summary,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)

