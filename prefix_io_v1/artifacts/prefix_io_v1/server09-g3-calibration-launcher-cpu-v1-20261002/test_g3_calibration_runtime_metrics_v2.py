"""CPU contracts: genuine pinned acquisition AST, fake backend, no GPU claims."""
import argparse
import ast
import collections
import dataclasses
import gc
import hashlib
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from unittest import mock
import weakref

ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root",type=Path)
ARGS,REST=parser.parse_known_args()
SOURCE=ARGS.source_root or ROOT/"artifacts/prefix_io_v1_server08_primary_qualification/contents"
spec=importlib.util.spec_from_file_location("_g3_runtime_test_module",Path(__file__).with_name("g3_calibration_runtime_metrics_v2.py"))
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)


def result(prompt=(1,),token=42):
    return [types.SimpleNamespace(prompt_token_ids=list(prompt),outputs=[types.SimpleNamespace(token_ids=[token])],
        num_cached_tokens=0,request_id="fixture-request")]


class RuntimeContracts(unittest.TestCase):
    def test_import_is_stdlib_and_does_not_create_cache(self):
        source=Path(R.__file__).read_text(encoding="utf-8")
        tree=ast.parse(source)
        top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        names={item.name.split(".")[0] for node in top_imports for item in (node.names if isinstance(node,ast.Import) else [types.SimpleNamespace(name=node.module)])}
        self.assertFalse(names & {"torch","vllm","py_kvcache"})
        with mock.patch.object(Path,"mkdir",side_effect=AssertionError("import mkdir")):
            exec(compile(tree,"cpu-import","exec"),{"__name__":"cpu_only_import"})

    def test_fixed_argv_raw_three_modes_and_no_curve_or_diagnostics(self):
        for mode in R.MODES:
            argv=R.fixed_argv(mode,Path("out"),Path("storage"),Path("model"),Path("plan"))
            self.assertEqual(argv[argv.index("--mode")+1],mode)
            self.assertEqual(argv[argv.index("--sizes")+1],"128")
            self.assertEqual(argv[argv.index("--reps")+1],"3")
            self.assertNotIn("--curves",argv);self.assertNotIn("--diagnostic-logprobs",argv)
        with self.assertRaises(ValueError):R.fixed_argv("planned",1,2,3,4)

    def test_missing_guard_rejects_before_source_or_backend(self):
        with mock.patch.object(R,"preflight_runtime",side_effect=AssertionError("read source")):
            with self.assertRaises(ValueError):R.execute_original_acquisition(ROOT,"cold",ROOT,ROOT,{}, {})

    def test_scalar_ownership_and_capacity(self):
        owner=types.SimpleNamespace()
        with self.assertRaises(ValueError):R.scalar_copy({"owner":owner})
        with self.assertRaises(ValueError):R.scalar_copy(float("nan"))
        self.assertEqual(len(R.scalar_copy(list(range(129)))),129)
        with self.assertRaises(ValueError):R.scalar_copy(list(range(257)))

    def test_generate_same_identity_args_once_and_restore(self):
        output=result(tuple(range(129)));calls=[]
        class LLM:
            def generate(self,*args,**kwargs):calls.append((args,kwargs));return output
        original=LLM.generate;taps=R.ScalarTaps();taps.install_generate(LLM)
        arg=object();kw=object();instance=LLM()
        self.assertIs(instance.generate(arg,flag=kw),output)
        self.assertEqual(len(calls),1);self.assertIs(calls[0][0][0],arg);self.assertIs(calls[0][1]["flag"],kw)
        self.assertEqual(taps.requests[0]["output_token_ids"],[42])
        taps.restore();self.assertIs(LLM.generate,original)

    def test_generate_original_error_object_and_control_signal(self):
        for error in (RuntimeError("original"),KeyboardInterrupt()):
            calls=[]
            class LLM:
                def generate(self,*args,**kwargs):calls.append(1);raise error
            taps=R.ScalarTaps();original=LLM.generate;taps.install_generate(LLM)
            try:
                with self.assertRaises(type(error)) as caught:LLM().generate()
                self.assertIs(caught.exception,error);self.assertEqual(calls,[1]);self.assertEqual(taps.requests,[])
            finally:taps.restore()
            self.assertIs(LLM.generate,original)

    def test_generate_observation_fault_preserves_success_and_capacity(self):
        output=object();calls=[]
        class LLM:
            def generate(self):calls.append(1);return output
        taps=R.ScalarTaps();taps.install_generate(LLM)
        try:
            for _ in range(20):self.assertIs(LLM().generate(),output)
            self.assertEqual(len(calls),20);self.assertLessEqual(len(taps.failures),8)
        finally:taps.restore()

    def test_unknown_output_metadata_is_never_retained(self):
        class Owner:pass
        owner=Owner();reference=weakref.ref(owner);output=result()
        output[0].num_cached_tokens=owner
        class LLM:
            def generate(self):return output
        taps=R.ScalarTaps();taps.install_generate(LLM)
        self.assertIs(LLM().generate(),output);self.assertEqual(taps.requests,[])
        taps.restore();del owner,output;gc.collect();self.assertIsNone(reference())

    def test_shutdown_same_result_or_exception_once(self):
        output=object();calls=[]
        class Core:
            def shutdown(self,*args,**kwargs):calls.append((args,kwargs));return output
        original=Core.shutdown;core=Core();taps=R.ScalarTaps();taps.install_engine_shutdown(core,original)
        self.assertIs(core.shutdown(timeout=15),output);self.assertEqual(len(calls),1)
        self.assertTrue(taps.engine_shutdown_returned);taps.restore()
        self.assertNotIn("shutdown",vars(core))
        error=SystemExit(7)
        def failure(owner,*args,**kwargs):raise error
        taps=R.ScalarTaps();taps.install_engine_shutdown(core,failure)
        try:
            with self.assertRaises(SystemExit) as caught:core.shutdown()
            self.assertIs(caught.exception,error);self.assertFalse(taps.engine_shutdown_returned)
        finally:taps.restore()

    def test_taps_do_not_retain_core_handler_or_output(self):
        class Core:
            def shutdown(self):return None
        core=Core();ref=weakref.ref(core);taps=R.ScalarTaps();taps.install_engine_shutdown(core,Core.shutdown)
        del core;gc.collect();self.assertIsNone(ref());taps.restore()
        class Output:
            pass
        output=Output();ref=weakref.ref(output)
        class LLM:
            def generate(self):return output
        taps=R.ScalarTaps();taps.install_generate(LLM);LLM().generate();taps.restore()
        del output;gc.collect();self.assertIsNone(ref())

    def _tail(self):
        thread=types.SimpleNamespace(is_alive=lambda:False)
        ring=types.SimpleNamespace(_closed=True,_drained=True,_fatal=None,_worker=thread,
            _stats=dict(accepted=7,completed=7,reaped=7),_ops={},_pending=collections.deque(),
            _ready=collections.deque(),_done=collections.deque())
        native=dict(active_parents=0,ring_ops=0,pending_copies=0,copy_ready=0,ready_load_fds=0,ready_preload_fds=0)
        snapshot=dict(native_shutdown_read=True,owner_capture=False,native=native,
            aio=dict(accepted=7,completed=7,reaped=7,outstanding=0,pending=0,ready=0,unreaped=0),stage_accounting=None)
        reactor=types.SimpleNamespace(_closed=True,_worker=thread,ring=ring,_active={},_inflight={},
            _pending_copies=[],_copy_ready=[],_ready_fds_load=[],_ready_fds_preload=[],
            _prefix_stage_accounting=None,_observation_failures=0)
        calls=[]
        def inspect():calls.append("inspect");return snapshot
        handler=types.SimpleNamespace(is_shutdown=True,_active={},coordinator=types.SimpleNamespace(reactor=reactor,inspect_snapshot=inspect))
        return handler,snapshot,calls

    def test_tail_one_snapshot_and_unknown_not_invented_fourstage(self):
        handler,snapshot,calls=self._tail();tail=R.post_shutdown_snapshot(handler)
        self.assertEqual(calls,["inspect"]);self.assertIsNone(tail["actual_stage_counters"])
        self.assertEqual(tail["stage_accounting_status"],"UNKNOWN_NOT_ENABLED")
        self.assertFalse(tail["cost_qualified"]);self.assertFalse(tail["native_release_qualified"])
        self.assertEqual(tail["actual_aio"]["accepted"],7)

    def test_tail_before_join_or_published_drift_reject(self):
        handler,snapshot,calls=self._tail();handler.is_shutdown=False
        with self.assertRaises(ValueError):R.post_shutdown_snapshot(handler)
        self.assertEqual(calls,[])
        handler.is_shutdown=True;snapshot["aio"]["reaped"]=999
        with self.assertRaises(ValueError):R.post_shutdown_snapshot(handler)
        self.assertEqual(calls,["inspect"])

    def test_handler_fault_cannot_replace_original_return(self):
        class Handler:
            def shutdown(self):return sentinel
        sentinel=object();handler=Handler();taps=R.ScalarTaps()
        taps.install_handler_shutdown(handler,Handler.shutdown)
        self.assertIs(handler.shutdown(),sentinel);self.assertEqual(taps.handler_shutdown_calls,1)
        self.assertEqual(taps.tails,[]);self.assertEqual(taps.failures,["ValueError"]);taps.restore()
        self.assertNotIn("shutdown",vars(handler))

    def test_source_binding_rejects_drift_and_instance_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();path=root/"fixture.py";raw=b'class Owner:\n    def shutdown(self): return 7\n'
            path.write_bytes(raw);name="_source_fixture_owner";module=types.ModuleType(name);module.__file__=str(path)
            sys.modules[name]=module
            try:
                exec(compile(raw,str(path),"exec",dont_inherit=True),module.__dict__)
                refs={"fixture.py":dict(path="fixture.py",bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())}
                owner=module.Owner();self.assertIs(R.original_method(owner,"shutdown",root,refs),module.Owner.shutdown)
                owner.shutdown=lambda:7
                with self.assertRaises(ValueError):R.original_method(owner,"shutdown",root,refs)
                del owner.shutdown;path.write_bytes(raw+b'\n')
                with self.assertRaises(ValueError):R.original_method(owner,"shutdown",root,refs)
            finally:sys.modules.pop(name,None)

    def test_source_path_traversal_rejects_before_read(self):
        with mock.patch.object(Path,"read_bytes",side_effect=AssertionError("read")):
            for relative in ("../owner.py","a/../owner.py","/owner.py","a\\owner.py"):
                with self.assertRaises(ValueError):R.checked_source(ROOT,{},relative)

    def test_foreign_helper_override_not_overwritten_on_restore(self):
        class Owner:
            def generate(self):return result()
        taps=R.ScalarTaps();taps.install_generate(Owner);foreign=lambda owner:None;Owner.generate=foreign
        taps.restore();self.assertIs(Owner.generate,foreign);self.assertEqual(taps.restorations,[False])

    def test_guard_side_conditions_fail_closed(self):
        witness=dict(schema_version=1,purpose=R.PURPOSE,gpu_uuid=R.GPU_UUID,label="server09-g3-calibration-cold-02",
            source_lock_sha256="a"*64,scope_sha256="b"*64,permissions_ref=dict(path="permit",bytes=1,sha256="c"*64),
            session_id=88,guard_command=["python","--execute"],seconds_limit=300,reserved_seconds=320,
            authorized_scope_verified=True,active_reservation_verified=True)
        with mock.patch.object(R.os,"name","posix"),mock.patch.object(R.os,"getsid",return_value=88,create=True),mock.patch.dict(os.environ,CUDA_VISIBLE_DEVICES=R.GPU_UUID):
            R.validate_guard(witness,"cold")
            for field,value in (("session_id",89),("gpu_uuid","GPU-wrong"),("active_reservation_verified",False),("reserved_seconds",999)):
                with self.assertRaises(ValueError):R.validate_guard(dict(witness,**{field:value}),"cold")


class OriginalMainReplay(unittest.TestCase):
    def test_exact_original_main_three_children_eighteen_requests(self):
        path=SOURCE/R.ACQUIRE;raw=path.read_bytes()
        self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),R.PINNED[R.ACQUIRE])
        tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("prompt","trace_summary","main")]
        self.assertEqual(len(nodes),3)
        all_requests=[];published=set();results=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();storage=root/"storage"
            for mode in R.MODES:
                out=root/mode;calls=[];taps=R.ScalarTaps();profile={}
                @dataclasses.dataclass
                class Metrics:
                    first_token_latency:float=0.125
                    is_corrupted:bool=False
                class Sampling:
                    def __init__(self,**kw):self.values=kw
                class LLM:
                    def __init__(self,model,**config):
                        self.config=config;self.cache=set();self.counter=0
                        self.llm_engine=types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=lambda **kw:calls.append("shutdown")))
                    def reset_prefix_cache(self,**kw):calls.append(("reset",kw));return True
                    def collective_rpc(self,method,timeout,args):return [method(self,*args)]
                    def generate(self,inputs,sampling,use_tqdm):
                        tokens=inputs[0]["prompt_token_ids"];key=tuple(tokens);n=len(tokens)-1;external=mode!="cold"
                        cached=external and (key in published or key in self.cache)
                        mem=cached and key in self.cache;self.cache.add(key)
                        if external and n:published.add(key)
                        self.counter+=1;rid="request-%d"%self.counter
                        profile.update(rid=rid,n=n,cached=cached,mem=mem)
                        return [types.SimpleNamespace(request_id=rid,prompt_token_ids=tokens,num_cached_tokens=n if cached else 0,
                            outputs=[types.SimpleNamespace(token_ids=[42])],metrics=Metrics())]
                original_generate=LLM.generate;taps.install_generate(LLM)
                def control(worker,action,path=None):
                    if action=="begin":profile["path"]=path;return {"profile_active":True}
                    if action=="end":
                        events=[]
                        if profile["cached"]:
                            n=profile["n"];rid=profile["rid"]
                            events=[dict(name="py_kvcache.transfer",args=dict(req_id=rid,direction="storage_to_gpu",success=True,
                                num_bytes=n*57344,src_cache=n//16 if profile["mem"] else 0,src_file=0 if profile["mem"] else n//16,src_preload=0)),
                                dict(name="py_kvcache.cuda_staging",args=dict(req_id=rid,direction="storage_to_gpu"))]
                            if not profile["mem"]:events.append(dict(name="py_kvcache.file_read",args=dict(req_id=rid,num_bytes=n*57344)))
                        Path(path).write_text(json.dumps(dict(traceEvents=events)))
                    return dict(handlers=[dict(staging_bytes=128*1024**2,staging_budget=128*1024**2,pinned=True,
                        io_size=917504,storage_block_bytes=917504,aio=dict(accepted=0,completed=0,reaped=0,outstanding=0,pending=0,ready=0,unreaped=0,fatal=None))])
                def new_json(path,value):path.write_text(json.dumps(value))
                base=types.SimpleNamespace(MODEL_ID=R.MODEL_NAME,AUTHOR_ROOT=root,ENGINE=dict(kv_transfer_config=None),SAMPLING={},
                    configure_runtime_environment=lambda:None,validate_local_model=lambda *a:(root/"model",{}),
                    write_new_json=new_json,require=lambda ok,msg:R.require(ok,msg))
                (root/"model").mkdir(exist_ok=True)
                fake_vllm=types.ModuleType("vllm");fake_vllm.__file__=str(root/"vllm.py");fake_vllm.LLM=LLM;fake_vllm.SamplingParams=Sampling
                fake_torch=types.ModuleType("torch");fake_torch.cuda=types.SimpleNamespace(mem_get_info=lambda:(32*1024**3,32*1024**3))
                storage_module=types.ModuleType("experiment_storage");storage_module.authorized_path=lambda p:None;storage_module.preflight=lambda p,n:{}
                heldout=types.ModuleType("heldout_manifest");heldout.disk_requirement=lambda *a:dict(required_free_bytes=0,floor_bytes=0)
                concurrent=types.ModuleType("concurrent_pilot_contract");concurrent.acquisition_delta=lambda *a:{};concurrent.acquisition_io_depth=lambda *a:4
                namespace=dict(__name__="cpu_original_acquisition",__file__=str(path),argparse=argparse,dataclasses=dataclasses,hashlib=hashlib,
                    json=json,os=os,re=re,Path=Path,time=time,traceback=traceback,base=base,worker_control=control)
                exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec",dont_inherit=True),namespace)
                old_cwd=Path.cwd()
                with mock.patch.dict(sys.modules,{"vllm":fake_vllm,"torch":fake_torch,"experiment_storage":storage_module,
                    "heldout_manifest":heldout,"concurrent_pilot_contract":concurrent}),mock.patch.dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",CUDA_VISIBLE_DEVICES=R.GPU_UUID),mock.patch.object(sys,"argv",R.fixed_argv(mode,out,storage,root/"model",root/"plan")):
                    # CPU fake-backend replay observes the unmodified alias
                    # call, without requiring Windows administrator symlinks.
                    try:
                        with mock.patch.object(Path,"symlink_to") as alias,contextlib.redirect_stdout(io.StringIO()):
                            code=namespace["main"]()
                        alias.assert_called_once_with(root/"model",target_is_directory=True)
                    finally:os.chdir(old_cwd);taps.restore()
                self.assertEqual(code,0);self.assertIs(LLM.generate,original_generate)
                report=json.loads((out/"result.json").read_text());config=json.loads((out/"frozen-config.json").read_text())
                self.assertEqual(config["model_alias"],R.MODEL_NAME);self.assertEqual(config["sampling"]["max_tokens"],1)
                if mode!="cold":self.assertEqual(config["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["load_planner"],"off")
                self.assertEqual(len(taps.requests),{"cold":3,"populate":9,"paired":6}[mode])
                if mode=="populate":
                    self.assertEqual([r["ordinal"] for r in taps.requests if r["sentinel"]],[1,4,7])
                    self.assertEqual([r["prompt_token_ids"] for r in taps.requests if r["sentinel"]],[[30000],[30001],[30002]])
                self.assertEqual(calls.count("shutdown"),1)
                all_requests.extend(taps.requests);results.append(report)
            self.assertEqual(len(all_requests),18)
            self.assertEqual([r["kind"] for r in results[0]["rows"]],["f"]*3)
            self.assertEqual([r["kind"] for r in results[2]["rows"]],["g_ssd","g_mem"]*3)
            self.assertEqual([r["metrics"]["first_token_latency"] for r in results[2]["rows"]],[0.125]*6)


if __name__=="__main__":unittest.main(argv=[sys.argv[0],*REST],verbosity=2)
