"""Original source AST replay, explicit CPU actors and no framework imports."""
import argparse
import ast
import collections
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parent.parent/"g3_source_readonly")
parser.add_argument("--support-root",type=Path,default=Path(__file__).resolve().parent.parent/"g2_source_readonly")
parser.add_argument("--drain-source",type=Path)
parser.add_argument("--connector-source",type=Path)
parser.add_argument("--adapter-source",type=Path)
args,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
spec=importlib.util.spec_from_file_location("_g3_observer_test",Path(__file__).with_name("g3_native_owner_observer.py"))
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
ROOT=args.source_root.resolve();SUPPORT=args.support_root.resolve()


class Poison:
    def __getattribute__(self,name): raise AssertionError("off touched "+name)


class CV:
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def notify_all(self): pass


class Actor:
    def __init__(self): self.alive=True;self.joins=0;self.ident=threading.get_ident()+1
    def join(self): self.joins+=1;self.alive=False
    def is_alive(self): return self.alive


class Closeable:
    def __init__(self): self.closes=0
    def close(self): self.closes+=1


class Incoming:
    def __init__(self): self.puts=[]
    def put(self,item): self.puts.append(item)


def compile_excerpt(module,path,classes=(),functions=()):
    tree=ast.parse(path.read_bytes(),str(path))
    body=[deepcopy(n) for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=="__future__"]
    for name,methods in classes:
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        selected=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        if not selected: selected=[ast.Pass(lineno=cls.lineno,col_offset=4)]
        require_names={n.name for n in selected if isinstance(n,ast.FunctionDef)}
        assert require_names==set(methods),(name,methods,require_names)
        # Class bases are CPU fixture-only; original method AST/line numbers are
        # exact and compiled with original future flags and source filename.
        replay=deepcopy(cls);replay.bases=[];replay.keywords=[];replay.decorator_list=[];replay.body=selected
        body.append(replay)
    body.extend(deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),"exec",dont_inherit=True),module.__dict__)
    # CPython 3.12 optimizes names using the complete module import symbol table.
    # Execute the exact code objects from that original full-source compilation,
    # retaining excerpt-created defaults/closure cells and fixture-only globals.
    baseline=P.source_codes(path.read_bytes(),path)
    for value in tuple(module.__dict__.values()):
        values=tuple(value.__dict__.values()) if isinstance(value,type) and value.__module__==module.__name__ else (value,)
        for function in values:
            if type(function) in (staticmethod,classmethod): function=function.__func__
            if type(function) is types.FunctionType and function.__globals__ is module.__dict__:
                matches=baseline.get(function.__qualname__,())
                if len(matches)==1: function.__code__=matches[0]


class Fixture:
    def __init__(self,max_records=1024):
        self.observer=P.NativeOwnerObserver(enabled=True,run_id="cpu-g3",source_root=ROOT,support_root=SUPPORT,
            drain_module_path=args.drain_source,connector_path=args.connector_source,adapter_path=args.adapter_source,max_records=max_records)
        self.previous={};self.modules={}
        for name,path in self.observer.paths.items():
            self.previous[name]=sys.modules.get(name)
            module=types.ModuleType(name);module.__file__=str(path);module.__spec__=None;module.__loader__=None
            module.__dict__.update(threading=threading,time=time,collections=collections,Future=Future,
                Any=object,TransferSpec=tuple,CanonicalKVCaches=object,LoadStoreSpec=object,OffloadingHandler=object,TransferResult=object,
                profile_scope=lambda *a,**kw:CV(),add_event=lambda *a,**kw:None,now_ns=time.monotonic_ns,
                NativeDrainUnknown=RuntimeError,LoadDeclined=RuntimeError,STAGES=P.STAGES,get_ident=threading.get_ident)
            sys.modules[name]=module;self.modules[name]=module
        state=self.m("vllm.distributed.kv_transfer.kv_transfer_state")
        state.KVConnectorBaseType=object
        compile_excerpt(state,self.path(state),functions=("get_kv_transfer_group","ensure_kv_transfer_shutdown"))
        gpu=self.m("vllm.v1.worker.gpu_worker")
        compile_excerpt(gpu,self.path(gpu),classes=(("Worker",("shutdown",)),))
        gpu.get_kv_transfer_group=state.get_kv_transfer_group;gpu.ensure_kv_transfer_shutdown=state.ensure_kv_transfer_shutdown
        c=self.m("vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector")
        compile_excerpt(c,self.path(c),classes=(("OffloadingConnector",("shutdown",)),))
        cw=self.m("vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker")
        compile_excerpt(cw,self.path(cw),classes=(("OffloadingConnectorWorker",("shutdown","_register_handlers")),))
        ow=self.m("vllm.v1.kv_offload.worker.worker")
        compile_excerpt(ow,self.path(ow),classes=(("OffloadingWorker",("__init__","register_handler","shutdown","transfer_async","get_finished")),))
        # TransferResult is an original dataclass declaration, not a supplied
        # result mapping. No source runtime imports or backend initializers run.
        result=next(n for n in ast.parse(self.path(ow).read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=="TransferResult")
        ow.dataclass=dataclass;ow.TransferType=tuple
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(result)],type_ignores=[])),str(self.path(ow)),"exec",dont_inherit=True),ow.__dict__)
        base=self.m("vllm.v1.kv_offload.base")
        compile_excerpt(base,self.path(base),classes=(("GPULoadStoreSpec",("medium",)),))
        hm=self.m("py_kvcache.vllm");hm.TransferResult=ow.TransferResult;hm.GPULoadStoreSpec=base.GPULoadStoreSpec
        hm.futures_wait=lambda *a,**kw:(set(),set())
        compile_excerpt(hm,self.path(hm),classes=(("NoopSharedStorageOffloadingHandler",("shutdown","transfer_async","get_finished","_check_native_fatal","_future_to_transfer_result")),
            ("PyKvCacheOffloadingSpec",("get_handlers",)),("SharedStorageLoadStoreSpec",("__init__","medium"))),functions=("_make_transfer_result",))
        rm=self.m("py_kvcache.reactor")
        rm.split_block_ids_for_files=lambda *a,**kw:[];rm._TransferProfile=types.SimpleNamespace
        compile_excerpt(rm,self.path(rm),classes=(("TransferCoordinator",("shutdown","inspect_snapshot","submit_load","submit_store","raise_if_native_fatal")),
            ("IoReactor",("shutdown","inspect_snapshot","_capture_owner_snapshot","parent_admission_snapshot","raise_if_native_fatal"))))
        rm.IoReactor._prefix_capacity_snapshot=lambda self,**kw:dict(free_reclaimable_staging_bytes=0)
        hm.TransferCoordinator=rm.TransferCoordinator
        am=self.m("py_kvcache.linux_aio");am.os=types.SimpleNamespace(close=lambda fd:None)
        compile_excerpt(am,self.path(am),classes=(("LinuxAioRing",("snapshot","close")),))
        am.LinuxAioRing._flush_locked=lambda self:None
        ac=self.m("prefix_io_control.stage_accounting")
        node=next(n for n in ast.parse(self.path(ac).read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=="StageAccounting")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(self.path(ac)),"exec",dont_inherit=True),ac.__dict__)
        tr=self.m("py_kvcache.transfer")
        compile_excerpt(tr,self.path(tr),classes=(("ParsedKvLayout",()),))
        self.account=ac.StageAccounting();self.account.bind()
        for i,stage in enumerate(P.STAGES): self.account.accepted(stage,i,(i+1)*8);self.account.completed(stage,i,(i+1)*8)
        self.ring=am.LinuxAioRing.__new__(am.LinuxAioRing)
        ring=self.ring;ring._cv=CV();ring._worker=Actor();ring._selector=Closeable();ring._wake_fd=123
        ring._closed=False;ring._closing=False;ring._drained=True;ring._fatal=None;ring.capacity=64
        ring._ops={};ring._opened={};ring._pending=collections.deque();ring._ready=collections.deque();ring._done=collections.deque()
        ring._stats=dict(accepted=7,completed=7,reaped=7,submit_calls=2,partial_submits=0,retry_submits=0,
            max_outstanding=2,max_kernel_inflight=2,max_submit_ns=10)
        self.reactor=rm.IoReactor.__new__(rm.IoReactor);r=self.reactor
        r.ring=ring;r._worker=Actor();r._submit_lock=CV();r._parent_cv=r._submit_lock;r._closed=False
        r._incoming=Incoming();r._STOP=object();r.staging_pool=Closeable();r._prefix_progress=types.SimpleNamespace(run_id="cpu-g3")
        r._prefix_stage_accounting=self.account;r._prefix_dispatch_controller=None;r._prefix_p4_bridge=None
        r._parent_count_valid=True;r._accepted_parent_count=0;r._parent_peak=2;r._parent_retired_count=3
        r._max_accepted_parents=2;r._parent_waiters=0;r._parent_backpressure_waits=0
        r._native_drain_unknown=False;r._native_fatal_reason=None;r._native_fatal_drain_verified=False
        r._native_failure_sync_total=0;r._native_failure_syncs=dict(h2d=0,d2h=0);r._observation_failures=0
        for name in self.observer.D.OWNER_FIELDS:
            setattr(r,name,0 if name in ("_preload_inflight_total","_data_inflight","_open_inflight","_prefix_p4_controls_pending","_owner_snapshot_pending") else {})
        r._active=[];r._pending_copies=[];r._copy_ready=collections.deque();r._ready_fds_load=collections.deque();r._ready_fds_preload=collections.deque()
        self.layout=tr.ParsedKvLayout();self.layout.bytes_per_kernel_block=[8,8];self.layout.storage_block_size_factor=2
        self.coordinator=rm.TransferCoordinator.__new__(rm.TransferCoordinator);self.coordinator.reactor=r;self.coordinator.layout=self.layout
        self.handler=hm.NoopSharedStorageOffloadingHandler.__new__(hm.NoopSharedStorageOffloadingHandler)
        h=self.handler;h.coordinator=self.coordinator;h.is_shutdown=False;h._active={};h._finished={};h._submitted_preload_ids=set()
        self.worker=gpu.Worker.__new__(gpu.Worker);self.worker.vllm_config=types.SimpleNamespace()
        self.spec=hm.PyKvCacheOffloadingSpec.__new__(hm.PyKvCacheOffloadingSpec);self.spec._handler=h;self.spec.vllm_config=self.worker.vllm_config
        self.registry=ow.OffloadingWorker()
        self.registry.register_handler(base.GPULoadStoreSpec,hm.SharedStorageLoadStoreSpec,h)
        self.registry.register_handler(hm.SharedStorageLoadStoreSpec,base.GPULoadStoreSpec,h)
        self.cw=cw.OffloadingConnectorWorker.__new__(cw.OffloadingConnectorWorker);self.cw.worker=self.registry;self.cw.spec=self.spec
        self.connector=c.OffloadingConnector.__new__(c.OffloadingConnector);self.connector.connector_worker=self.cw;self.connector.connector_scheduler=None
        state._KV_CONNECTOR_AGENT=self.connector

    def m(self,name): return self.modules[name]
    def path(self,module): return self.observer.paths[module.__name__]
    def restore(self):
        for name,previous in self.previous.items():
            if previous is None: sys.modules.pop(name,None)
            else: sys.modules[name]=previous
        self.previous.clear();self.modules.clear()
    def empty_load(self,job=7):
        hm=self.m("py_kvcache.vllm");gpu=self.m("vllm.v1.kv_offload.base").GPULoadStoreSpec()
        gpu.block_ids=[4];gpu.group_sizes=[1];gpu.block_indices=[0]
        return job,(hm.SharedStorageLoadStoreSpec([]),gpu)
    def shutdown(self): return self.observer.observe_shutdown(self.worker,self.handler)


class Contracts(unittest.TestCase):
    def setUp(self): self.f=Fixture()
    def tearDown(self): self.f.restore()

    def test_source_bound_registered_chain_shape_and_original_code_globals(self):
        witness=self.f.observer.verify_owner(self.f.worker,self.f.handler)
        self.assertTrue(self.f.observer.valid,self.f.observer.reason)
        self.assertEqual(witness.registration_chain[4:6],("OffloadingWorker","NoopSharedStorageOffloadingHandler"))
        self.assertFalse(witness.source_bound_runtime_observed or witness.gpu_verified or witness.production_qualified)

    def test_original_transfer_and_result_batch_once_identity_without_extra_consumption(self):
        seen={};handler=self.f.handler;o=self.f.observer
        code1=type(handler).transfer_async.__code__;code2=type(handler).get_finished.__code__
        def profile(frame,event,value):
            if frame.f_code in (code1,code2) and event=="return": seen.setdefault(frame.f_code,[]).append(value)
        sys.setprofile(profile)
        try:
            returned=o.observe_transfer(self.f.worker,handler,*self.f.empty_load(),req_id="r1")
            results=o.observe_get_finished(self.f.worker,handler)
        finally: sys.setprofile(None)
        self.assertTrue(o.valid,o.reason)
        self.assertIs(returned,seen[code1][0]);self.assertIs(results,seen[code2][0])
        self.assertEqual((len(seen[code1]),len(seen[code2])),(1,1))
        self.assertEqual((o.transfers[0].job_id,o.results[0].job_id),(7,7))
        self.assertEqual(handler.get_finished(),[])  # Native prior batch was consumed once.

    def test_original_shutdown_join_close_and_snapshot_once_then_all_four_scalar_totals(self):
        seen=[];snapshot=type(self.f.coordinator).inspect_snapshot.__code__
        def profile(frame,event,value):
            if frame.f_code is snapshot and event=="call": seen.append(1)
        sys.setprofile(profile)
        try: self.assertIsNone(self.f.shutdown())
        finally: sys.setprofile(None)
        o=self.f.observer;self.assertTrue(o.valid,o.reason);self.assertEqual(seen,[1])
        self.assertEqual((self.f.reactor._worker.joins,self.f.ring._worker.joins,self.f.reactor.staging_pool.closes),(1,1,1))
        tail=o.last_tail;self.assertEqual(tuple(row[2] for row in tail.stages),(8,16,24,32))
        self.assertTrue(tail.all_four_stages_nonzero)
        self.assertFalse(tail.native_io_completion_observed or tail.gpu_verified or tail.production_qualified or tail.cost_qualified or tail.effect_verified or tail.release_credit)
        self.assertIsNone(tail.gpu_elapsed_ns);self.assertEqual(tail.frame_io_attribution,"UNKNOWN")

    def test_off_does_not_read_sources_worker_payload_snapshot_or_arguments(self):
        o=P.NativeOwnerObserver(enabled=False,origin=Poison(),source_root=Poison(),run_id=Poison(),max_records=Poison())
        calls=[];sentinel=object();p=Poison()
        class Original:
            def transfer_async(self,*a,**kw): calls.append((a,kw));return sentinel
            def get_finished(self): calls.append("result");return sentinel
            def shutdown(self): calls.append("shutdown");return sentinel
        h=Original()
        self.assertIs(o.observe_transfer(p,h,p,spec=p),sentinel)
        self.assertIs(o.observe_get_finished(p,h),sentinel);self.assertIs(o.observe_shutdown(p,h),sentinel)
        self.assertEqual(len(calls),3);self.assertIs(calls[0][0][0],p)

    def test_registered_handler_or_direction_or_spec_identity_drift_rejected(self):
        for mutate in (lambda:self.f.registry.handlers.clear(),
            lambda:self.f.registry.transfer_type_to_handler.__setitem__(("GPU","SHARED_STORAGE"),object()),
            lambda:setattr(self.f.spec,"_handler",object()),
            lambda:setattr(self.f.spec,"vllm_config",object())):
            saved=(set(self.f.registry.handlers),dict(self.f.registry.transfer_type_to_handler),self.f.spec._handler,self.f.spec.vllm_config)
            self.f.observer.valid=True;mutate();self.assertIsNone(self.f.observer.verify_owner(self.f.worker,self.f.handler))
            self.assertFalse(self.f.observer.valid)
            self.f.registry.handlers,self.f.registry.transfer_type_to_handler,self.f.spec._handler,self.f.spec.vllm_config=saved

    def test_same_filename_qualname_code_with_foreign_globals_rejected(self):
        original=type(self.f.handler).shutdown
        duplicate=types.FunctionType(original.__code__,dict(original.__globals__),original.__name__)
        duplicate.__qualname__=original.__qualname__
        type(self.f.handler).shutdown=duplicate
        try:
            self.assertIsNone(self.f.observer.verify_owner(self.f.worker,self.f.handler))
            self.assertFalse(self.f.observer.valid)
        finally: type(self.f.handler).shutdown=original

    def test_modified_code_same_globals_filename_qualname_rejected(self):
        original=type(self.f.handler).shutdown
        changed=original.__code__.replace(co_consts=original.__code__.co_consts+("drift",))
        duplicate=types.FunctionType(changed,original.__globals__,original.__name__);duplicate.__qualname__=original.__qualname__
        type(self.f.handler).shutdown=duplicate
        try: self.assertIsNone(self.f.observer.verify_owner(self.f.worker,self.f.handler));self.assertFalse(self.f.observer.valid)
        finally: type(self.f.handler).shutdown=original

    def test_loaded_module_identity_or_origin_drift_rejected(self):
        name="py_kvcache.vllm";module=sys.modules[name];sys.modules[name]=types.ModuleType(name)
        try: self.assertIsNone(self.f.observer.verify_owner(self.f.worker,self.f.handler));self.assertFalse(self.f.observer.valid)
        finally: sys.modules[name]=module

    def test_cpu_fixture_cannot_claim_runtime_or_ssd_truth(self):
        self.f.observer.origin="source_bound_runtime"
        self.assertIsNone(self.f.observer.verify_owner(self.f.worker,self.f.handler));self.assertFalse(self.f.observer.valid)
        with self.assertRaises(ValueError): P.OwnerWitness("cpu","cpu_fixture",(),(),True)
        with self.assertRaises(ValueError): P.NativeTailObservation("cpu","cpu_fixture",True,1,(),(),(),True,True,())

    def test_observer_preflight_fault_preserves_original_success_once(self):
        calls=[];sentinel=object()
        self.f.handler.transfer_async=lambda *a,**kw:calls.append(1) or sentinel
        self.assertIs(self.f.observer.observe_transfer(self.f.worker,self.f.handler,1,Poison()),sentinel)
        self.assertEqual(calls,[1]);self.assertFalse(self.f.observer.valid)

    def test_original_exception_and_control_signal_same_object(self):
        for error in (RuntimeError("native"),KeyboardInterrupt("original control")):
            calls=[]
            def fail(*a,**kw): calls.append(1);raise error
            self.f.handler.transfer_async=fail;self.f.observer.valid=True
            try: self.f.observer.observe_transfer(self.f.worker,self.f.handler,1,())
            except BaseException as caught: self.assertIs(caught,error)
            else: self.fail("original exception lost")
            self.assertEqual(calls,[1])

    def test_result_observer_capacity_failure_does_not_lose_original_batch(self):
        o=self.f.observer;o.max_records=1
        self.assertTrue(o.observe_transfer(self.f.worker,self.f.handler,*self.f.empty_load()))
        result=o.observe_get_finished(self.f.worker,self.f.handler)
        self.assertEqual(len(result),1);self.assertFalse(o.valid);self.assertIsNone(o.last_tail)

    def test_task_output_array_or_owner_cannot_be_retained_as_result_scalar(self):
        error_owner=types.SimpleNamespace()
        class Evil:
            def __init__(self,owner): self.owner=owner
            def __eq__(self,other): raise AssertionError("arbitrary equality called")
        o=self.f.observer
        o.observe_transfer(self.f.worker,self.f.handler,*self.f.empty_load())
        value=self.f.m("vllm.v1.kv_offload.worker.worker").TransferResult(7,True,0,transfer_type=(Evil(error_owner),"GPU"))
        with self.assertRaises(ValueError): o._copy_results([value])
        self.assertEqual(o.results,[])

    def test_failed_short_physical_io_does_not_qualify_even_empty_owner_queues(self):
        self.f.account.stats["ssd_read"]["transferred_bytes"]-=1
        self.assertIsNone(self.f.shutdown());self.assertFalse(self.f.observer.valid);self.assertIsNone(self.f.observer.last_tail)
        self.assertEqual(self.f.reactor._worker.joins,1)

    def test_missing_direct_parent_field_not_snapshot_default_zero(self):
        del self.f.reactor._parent_waiters
        self.assertIsNone(self.f.shutdown());self.assertFalse(self.f.observer.valid);self.assertIsNone(self.f.observer.last_tail)

    def test_ring_unreaped_or_outstanding_remains_unknown(self):
        self.f.ring._done.append((7,8))
        self.assertIsNone(self.f.shutdown());self.assertFalse(self.f.observer.valid);self.assertIsNone(self.f.observer.last_tail)

    def test_observation_failure_or_wrong_run_rejected(self):
        self.f.reactor._observation_failures=1
        self.assertIsNone(self.f.shutdown());self.assertFalse(self.f.observer.valid)

    def test_original_shutdown_failure_same_object_and_no_snapshot(self):
        calls=[];error=RuntimeError("native shutdown failed")
        def fail(): calls.append(1);raise error
        self.f.handler.shutdown=fail
        try: self.f.shutdown()
        except RuntimeError as caught: self.assertIs(caught,error)
        else: self.fail("native error lost")
        self.assertEqual(calls,[1]);self.assertIsNone(self.f.observer.last_tail)

    def test_source_drift_fault_before_original_shutdown_still_closes_once(self):
        # Do not edit frozen sources: inject failure at the CPU source gate.
        original=self.f.observer._check_sources
        self.f.observer._check_sources=lambda:(_ for _ in ()).throw(ValueError("SHA drift"))
        self.assertIsNone(self.f.shutdown());self.assertEqual(self.f.reactor._worker.joins,1)
        self.assertFalse(self.f.observer.valid);self.assertIsNone(self.f.observer.last_tail)
        self.f.observer._check_sources=original

    def test_actual_source_size_sha_gate_rejects_changed_bytes_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/"source.py";raw=b"value=1\n";path.write_bytes(raw)
            sha=hashlib.sha256(raw).hexdigest()
            self.assertEqual(P.checked(path,len(raw),sha)[1],raw)
            with self.assertRaisesRegex(ValueError,"bytes drift"): P.checked(path,len(raw)+1,sha)
            path.write_bytes(b"value=2\n")
            with self.assertRaisesRegex(ValueError,"SHA drift"): P.checked(path,len(raw),sha)

    def test_no_owner_future_or_exception_traceback_retained_after_success(self):
        o=self.f.observer;o.observe_transfer(self.f.worker,self.f.handler,*self.f.empty_load())
        o.observe_get_finished(self.f.worker,self.f.handler);self.f.shutdown()
        refs=[weakref.ref(x) for x in (self.f.worker,self.f.handler,self.f.reactor,self.f.coordinator,self.f.ring)]
        self.f.restore();self.f=None;gc.collect()
        self.assertTrue(all(ref() is None for ref in refs));self.assertIsNotNone(o.last_tail)
        self.f=Fixture()  # tearDown owns this separate fixture only.

    def test_no_owner_retention_from_original_exception_traceback(self):
        o=self.f.observer
        class OriginalFault(Exception): pass
        def fail(*args,**kw): raise OriginalFault("original")
        self.f.handler.transfer_async=fail
        try: o.observe_transfer(self.f.worker,self.f.handler,1,())
        except OriginalFault: pass
        reference=weakref.ref(self.f.handler)
        self.f.restore();self.f=None;gc.collect();self.assertIsNone(reference())
        self.f=Fixture()

    def test_original_source_and_old_validator_classes_are_unchanged(self):
        source={name:path.read_bytes() for name,path in self.f.observer.paths.items()}
        classes=dict(self.f.observer.D.NativeDrainCPUProvider.__dict__)
        self.f.shutdown()
        for name,raw in source.items(): self.assertEqual(self.f.observer.paths[name].read_bytes(),raw)
        self.assertEqual(dict(self.f.observer.D.NativeDrainCPUProvider.__dict__),classes)


if __name__=="__main__": unittest.main()
