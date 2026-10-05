"""CPU fixtures execute the locked original shutdown/snapshot ASTs, no runtime."""
import argparse
import ast
import collections
from copy import deepcopy
from dataclasses import FrozenInstanceError
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest
import weakref

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parent.parent/"g2_source_readonly")
args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
ROOT=args.source_root.resolve(strict=True)
spec=importlib.util.spec_from_file_location("g2_drain_candidate",Path(__file__).with_name("g2_native_drain_provider.py"))
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
VLLM="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/vllm.py"
NATIVE="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
AIO="third_party/work/py-kvcache-p4-02-cpu/py_kvcache/linux_aio.py"
ACCOUNT="third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/stage_accounting.py"


class Poison:
    def __getattribute__(self,name): raise AssertionError("off read poison: "+name)


class CV:
    def __enter__(self): return self
    def __exit__(self,*args): return None
    def notify_all(self): pass


class Worker:
    def __init__(self): self.alive=True;self.joins=0;self.ident=threading.get_ident()+1
    def join(self): self.joins+=1;self.alive=False
    def is_alive(self): return self.alive


class Closeable:
    def __init__(self): self.closes=0
    def close(self): self.closes+=1


class Incoming:
    def __init__(self): self.puts=[]
    def put(self,x): self.puts.append(x)


class OSFixture:
    def __init__(self): self.closed=[]
    def close(self,fd): self.closed.append(fd)


def original_class(relative,name,methods,namespace):
    source=ROOT/relative
    tree=ast.parse(source.read_text(),str(source))
    actual=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    selected=[deepcopy(n) for n in actual.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected}==set(methods)
    body=ast.ClassDef(name=name,bases=[],keywords=[],body=selected,decorator_list=[])
    code=ast.fix_missing_locations(ast.Module(body=[body],type_ignores=[]))
    exec(compile(code,str(source),"exec"),namespace)
    return namespace[name]


class Fixture:
    def __init__(self):
        self.os=OSFixture()
        ns=dict(threading=threading,time=__import__("time"),collections=collections,
            os=self.os,errno=__import__("errno"),NativeDrainUnknown=RuntimeError)
        Ring=original_class(AIO,"LinuxAioRing",("snapshot","close"),ns)
        Reactor=original_class(NATIVE,"IoReactor",("shutdown","inspect_snapshot","_capture_owner_snapshot",
            "parent_admission_snapshot"),ns)
        Coordinator=original_class(NATIVE,"TransferCoordinator",("shutdown","inspect_snapshot"),ns)
        Handler=original_class(VLLM,"NoopSharedStorageOffloadingHandler",("shutdown",),ns)
        Ring._flush_locked=lambda self:None
        Reactor.raise_if_native_fatal=lambda self:None
        Reactor._prefix_capacity_snapshot=lambda self,**kw:dict(free_reclaimable_staging_bytes=0)
        Handler._future_to_transfer_result=lambda *args:object()
        tree=ast.parse((ROOT/ACCOUNT).read_text())
        account=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="StageAccounting")
        scope=dict(get_ident=threading.get_ident,STAGES=P.STAGES)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[account],type_ignores=[])),str(ROOT/ACCOUNT),"exec"),scope)
        self.accounting=scope["StageAccounting"]()
        self.accounting.bind()
        for n,stage in enumerate(P.STAGES):
            self.accounting.accepted(stage,n,(n+1)*8)
            self.accounting.completed(stage,n,(n+1)*8)
        self.ring=Ring()
        self.ring._cv=CV();self.ring._closed=False;self.ring._closing=False;self.ring._drained=True;self.ring._fatal=None
        self.ring._worker=Worker();self.ring._selector=Closeable();self.ring._wake_fd=123
        self.ring._pending=collections.deque();self.ring._ready=collections.deque();self.ring._done=collections.deque()
        self.ring._ops={};self.ring._opened={};self.ring.capacity=64
        self.ring._stats=dict(accepted=7,completed=7,reaped=7,submit_calls=2,partial_submits=0,
            retry_submits=0,max_outstanding=2,max_kernel_inflight=2,max_submit_ns=10)
        self.reactor=Reactor()
        r=self.reactor
        r.ring=self.ring;r._worker=Worker();r._submit_lock=CV();r._parent_cv=r._submit_lock
        r._closed=False;r._incoming=Incoming();r._STOP=object();r.staging_pool=Closeable()
        r._prefix_progress=types.SimpleNamespace(run_id="cpu-drain")
        r._prefix_stage_accounting=self.accounting;r._prefix_dispatch_controller=None;r._prefix_p4_bridge=None
        r._parent_count_valid=True;r._accepted_parent_count=0;r._parent_peak=2;r._parent_retired_count=3
        r._max_accepted_parents=2;r._parent_waiters=0;r._parent_backpressure_waits=0
        r._native_drain_unknown=False;r._native_fatal_reason=None;r._native_fatal_drain_verified=False
        r._native_failure_sync_total=0;r._native_failure_syncs=dict(h2d=0,d2h=0)
        r._observation_failures=0
        for name in P.OWNER_FIELDS:
            setattr(r,name,0 if name in ("_preload_inflight_total","_data_inflight","_open_inflight",
                "_prefix_p4_controls_pending","_owner_snapshot_pending") else {})
        r._active=[];r._pending_copies=[];r._copy_ready=collections.deque()
        r._ready_fds_load=collections.deque();r._ready_fds_preload=collections.deque()
        self.coordinator=Coordinator();self.coordinator.reactor=r
        self.handler=Handler();h=self.handler
        h.coordinator=self.coordinator;h.is_shutdown=False;h._active={};h._finished={};h._submitted_preload_ids=set()
        self.provider=P.NativeDrainCPUProvider(enabled=True,run_id="cpu-drain",source_root=ROOT)

    def observe(self): return self.provider.observe_original_shutdown(self.handler)


class DrainContracts(unittest.TestCase):
    def setUp(self): self.f=Fixture()

    def invalid(self):
        self.f.observe()
        self.assertFalse(self.f.provider.valid)
        self.assertIsNone(self.f.provider.last_drain)
        self.assertIsNone(self.f.provider.last_receipt)

    def test_exact_original_shutdown_chain_joins_closes_then_copies_all_four_stages(self):
        self.assertIsNone(self.f.observe())
        p,r=self.f.provider,self.f.reactor
        self.assertTrue(p.valid)
        self.assertEqual((r._worker.joins,self.f.ring._worker.joins,r.staging_pool.closes,self.f.ring._selector.closes),(1,1,1,1))
        self.assertEqual(self.f.os.closed,[123])
        self.assertEqual(tuple(x.nbytes for x in p.last_drain.accepted),(8,16,24,32))
        self.assertEqual(p.last_drain.transferred_bytes,(8,16,24,32))
        self.assertEqual(p.last_drain.failed_ops,(0,0,0,0))
        self.assertFalse(p._tail_open)
        self.assertFalse(p.last_receipt.gpu_verified or p.last_receipt.production_qualified or p.last_receipt.effect_verified)
        self.assertEqual(p.last_receipt.origin,"cpu_fixture")

    def test_source_original_methods_are_not_replaced_or_wrapped(self):
        targets=(self.f.handler,self.f.coordinator,self.f.reactor,self.f.ring)
        before=[dict(type(x).__dict__) for x in targets]
        source={s:(ROOT/s).read_bytes() for s in P.SOURCE_REFS}
        self.f.observe()
        for owner,previous in zip(targets,before): self.assertEqual(dict(type(owner).__dict__),previous)
        for path,value in source.items(): self.assertEqual((ROOT/path).read_bytes(),value)
        self.assertFalse(any("shutdown" in vars(x) for x in targets))

    def test_off_does_not_read_source_adapter_snapshot_or_owner_fields(self):
        p=P.NativeDrainCPUProvider(enabled=False,origin=Poison(),run_id=Poison(),source_root=Poison(),
            adapter_path=Poison(),connector_path=Poison())
        sentinel=object();calls=[]
        class OnlyShutdown:
            def shutdown(self): calls.append(1);return sentinel
            def __getattr__(self,name): raise AssertionError("off owner read")
        self.assertIs(p.observe_original_shutdown(OnlyShutdown()),sentinel)
        self.assertEqual(calls,[1]);self.assertIsNone(p.last_receipt)

    def test_native_origin_rejected_before_source_or_gpu_reads(self):
        with self.assertRaisesRegex(ValueError,"rejects native"):
            P.NativeDrainCPUProvider(enabled=True,origin="native_gpu_recording",source_root=Poison())

    def test_gpu_authority_always_rejected(self):
        with self.assertRaisesRegex(ValueError,"no GPU launch"):
            self.f.provider.require_gpu_launch(approved=True,seconds=400)

    def test_original_error_object_wins_and_snapshot_not_read(self):
        error=RuntimeError("original failure")
        calls=[]
        def fail(): calls.append(1);raise error
        self.f.handler.shutdown=fail
        try: self.f.observe()
        except RuntimeError as caught: self.assertIs(caught,error)
        else: self.fail("original error lost")
        self.assertEqual(calls,[1]);self.assertFalse(self.f.provider.valid)
        self.assertIsNone(self.f.provider.last_drain)

    def test_original_return_identity_survives_unbound_method_and_observer_failure(self):
        sentinel=object();calls=[]
        self.f.handler.shutdown=lambda:calls.append(1) or sentinel
        self.assertIs(self.f.observe(),sentinel)
        self.assertEqual(calls,[1]);self.assertFalse(self.f.provider.valid)

    def test_original_baseexception_remains_same_object(self):
        error=KeyboardInterrupt("original control signal")
        def fail(): raise error
        self.f.handler.shutdown=fail
        with self.assertRaises(KeyboardInterrupt) as caught: self.f.observe()
        self.assertIs(caught.exception,error)

    def test_snapshot_error_after_success_does_not_rerun_shutdown(self):
        original=type(self.f.reactor)._capture_owner_snapshot
        error=RuntimeError("snapshot failure")
        def raising(*args,**kw): raise error
        # An altered instance boundary is rejected before native call, preserving original shutdown.
        self.f.reactor._capture_owner_snapshot=raising
        self.assertIsNone(self.f.observe())
        self.assertEqual(self.f.reactor._worker.joins,1)
        self.assertEqual(self.f.ring._worker.joins,1)
        self.assertFalse(self.f.provider.valid)
        self.assertIs(type(self.f.reactor)._capture_owner_snapshot,original)

    def test_provider_tail_cannot_be_fabricated_before_or_after_original(self):
        with self.assertRaisesRegex(ValueError,"shutdown tail"):
            self.f.provider.convert_post_shutdown({}, {})
        self.f.observe()
        with self.assertRaisesRegex(ValueError,"shutdown tail"):
            self.f.provider.convert_post_shutdown({}, {})

    def test_dead_worker_alone_future_done_and_empty_queue_cannot_qualify(self):
        self.f.reactor._worker.alive=False
        self.f.accounting.stats["h2d"]["transferred_bytes"]-=1
        self.invalid()
        self.assertEqual(self.f.reactor._worker.joins,1)

    def test_short_io_rejected_even_when_completed_requested_matches_acceptance(self):
        self.f.accounting.stats["ssd_read"]["transferred_bytes"]=4
        self.f.accounting.stats["ssd_read"]["failed_ops"]=1
        self.invalid()

    def test_failed_operation_cannot_be_hidden_by_full_transferred_bytes(self):
        self.f.accounting.stats["d2h"]["failed_ops"]=1
        self.invalid()

    def test_accepted_completed_ops_and_requested_bytes_must_exactly_match(self):
        for field in ("completed_ops","completed_requested_bytes","accepted_bytes","accepted_ops"):
            with self.subTest(field=field):
                self.f=Fixture();self.f.accounting.stats["h2d"][field]+=1;self.invalid()

    def test_actual_stage_inflight_and_records_must_be_empty(self):
        for field in ("inflight_ops","inflight_bytes","record"):
            with self.subTest(field=field):
                self.f=Fixture()
                if field=="record": self.f.accounting.records[("h2d","missing")]=8
                else: self.f.accounting.stats["h2d"][field]=1
                self.invalid()

    def test_accounting_valid_bound_error_and_missing_stage_fail_closed(self):
        for case in ("valid","bound","error","stage"):
            with self.subTest(case=case):
                self.f=Fixture()
                if case in ("valid","bound"): setattr(self.f.accounting,case,False)
                elif case=="error": self.f.accounting.error="unknown completion"
                else: self.f.accounting.stats.pop("ssd_write")
                self.invalid()

    def test_actual_boolean_and_integer_fields_never_accept_truthy_surrogates(self):
        for case in ("valid","ops","closed","drained","worker"):
            with self.subTest(case=case):
                self.f=Fixture()
                if case=="valid": self.f.accounting.valid=1
                elif case=="ops": self.f.accounting.stats["h2d"]["accepted_ops"]=True
                elif case=="closed": self.f.ring._closed=1
                elif case=="drained": self.f.ring._drained=1
                else: self.f.reactor._worker.is_alive=lambda:0
                self.invalid()

    def test_run_id_must_come_from_actual_original_progress_state(self):
        for value in (None,"foreign"):
            with self.subTest(run=value):
                self.f=Fixture()
                self.f.reactor._prefix_progress=None if value is None else types.SimpleNamespace(run_id=value)
                self.invalid()

    def test_all_actual_native_owner_collections_and_counts_required_zero(self):
        for field in P.OWNER_FIELDS:
            with self.subTest(field=field):
                self.f=Fixture()
                value=getattr(self.f.reactor,field)
                if type(value) is int: setattr(self.f.reactor,field,1)
                elif type(value) is dict: value["retained"]=object()
                else: value.append(object())
                self.invalid()

    def test_missing_owner_field_never_defaults_to_zero(self):
        del self.f.reactor._copy_ready
        self.invalid()

    def test_observation_failures_are_real_native_counter_not_future_state(self):
        self.f.reactor._observation_failures=1
        self.invalid()

    def test_missing_observation_counter_never_defaults_to_zero(self):
        del self.f.reactor._observation_failures
        self.invalid()

    def test_accepted_parent_count_and_waiters_must_be_zero_and_valid(self):
        for field,value in (("_accepted_parent_count",1),("_parent_waiters",1),("_parent_count_valid",False)):
            with self.subTest(field=field):
                self.f=Fixture();setattr(self.f.reactor,field,value);self.invalid()

    def test_native_unknown_or_fatal_shutdown_cannot_yield_receipt(self):
        for field,value in (("_native_drain_unknown",True),("_native_fatal_reason","native failed")):
            with self.subTest(field=field):
                self.f=Fixture();setattr(self.f.reactor,field,value)
                try: self.f.observe()
                except RuntimeError: pass
                self.assertFalse(self.f.provider.valid);self.assertIsNone(self.f.provider.last_drain)

    def test_aio_drain_fatal_or_unreaped_result_cannot_qualify(self):
        for case in ("drain","fatal","unreaped","pending","ready","outstanding","accepted","reaped"):
            with self.subTest(case=case):
                self.f=Fixture()
                if case=="drain": self.f.ring._drained=False
                elif case=="fatal": self.f.ring._fatal=RuntimeError("actual aio failure")
                elif case=="unreaped": self.f.ring._done.append((1,8))
                elif case=="pending": self.f.ring._pending.append(object())
                elif case=="ready": self.f.ring._ready.append(object())
                elif case=="outstanding": self.f.ring._ops[1]=object()
                else: self.f.ring._stats[case]+=1
                try: self.f.observe()
                except RuntimeError: pass
                self.assertFalse(self.f.provider.valid);self.assertIsNone(self.f.provider.last_drain)

    def test_alive_native_worker_after_join_refuses_snapshot_qualification(self):
        self.f.reactor._worker.join=lambda:None
        self.invalid()

    def test_missing_linux_aio_snapshot_field_not_synthesized(self):
        self.f.ring._stats.pop("reaped")
        self.invalid()

    def test_future_access_belongs_only_to_original_handler_not_provider(self):
        class Future:
            def __init__(self): self.calls=0
            def done(self): self.calls+=1;return True
        future=Future()
        self.f.handler._active[1]=(future,0,("cpu","cpu"))
        self.f.observe()
        self.assertEqual(future.calls,1)
        self.assertEqual(self.f.provider.last_receipt.handler_active,0)

    def test_immutable_receipt_retains_no_runtime_owner_or_exception(self):
        self.f.observe();receipt=self.f.provider.last_receipt
        with self.assertRaises(FrozenInstanceError): receipt.run_id="other"
        refs=[weakref.ref(x) for x in (self.f.handler,self.f.coordinator,self.f.reactor,self.f.ring,self.f.accounting)]
        provider=self.f.provider;self.f=None;gc.collect()
        self.assertTrue(all(r() is None for r in refs))
        self.assertIs(provider.last_receipt,receipt)
        self.assertTrue(all(type(row) is tuple for row in receipt.remaining_native_owners))

    def test_source_hash_drift_rejected_before_any_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            dst=Path(folder)
            for path in P.SOURCE_REFS:
                (dst/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dst/path)
            changed=dst/NATIVE;changed.write_bytes(changed.read_bytes()+b"\n#drift\n")
            with self.assertRaisesRegex(ValueError,"source byte count drift"):
                P.NativeDrainCPUProvider(enabled=True,run_id="cpu",source_root=dst)

    def test_verification_does_not_import_runtime_or_submit_new_work(self):
        self.f.observe()
        for name in ("torch","vllm","py_kvcache","pynvml","numpy"):
            self.assertNotIn(name,sys.modules)
        self.assertEqual(self.f.reactor._incoming.puts,[self.f.reactor._STOP])
        self.assertFalse(hasattr(P,"execute_model") or hasattr(P,"load_cost_table"))


    def test_observer_exception_string_or_control_signal_never_skips_original(self):
        class BadError(RuntimeError):
            def __str__(self): raise KeyboardInterrupt("bad observer repr")
        def broken_source_gate(): raise BadError()
        self.f.provider._check_sources=broken_source_gate
        self.assertIsNone(self.f.observe())
        self.assertEqual(self.f.reactor._worker.joins,1)
        self.assertFalse(self.f.provider.valid)

    def test_instance_invalidation_override_cannot_replace_original_exception(self):
        error=RuntimeError("original exception")
        def fail(): raise error
        def observer_invalid(*args): raise KeyboardInterrupt("observer invalidation")
        self.f.handler.shutdown=fail
        self.f.provider.invalidate=observer_invalid
        with self.assertRaises(RuntimeError) as caught: self.f.observe()
        self.assertIs(caught.exception,error)

    def test_frozen_adapter_consumes_exact_same_private_drain_type_without_gpu_qualification(self):
        F=self.f.provider.F
        adapter=F.FullStepFrameAdapter("cpu-drain","a"*64,P.NATIVE_SOURCE_SHA256,enabled=True,max_steps=2)
        for offset in range(2):
            ordinal=10+offset
            runner=types.SimpleNamespace(speculative_config=None,use_async_scheduling=False,is_pooling_model=False,
                _profile_step=ordinal+1,parallel_config=types.SimpleNamespace(pipeline_parallel_size=1,
                    data_parallel_size=1,tensor_parallel_size=1),input_batch=types.SimpleNamespace(num_reqs=1,
                    req_ids=["r"],num_computed_tokens_cpu=[offset],num_prompt_tokens=[1]))
            scheduler=types.SimpleNamespace(num_scheduled_tokens={"r":1},total_num_scheduled_tokens=1,
                scheduled_spec_decode_tokens={})
            output=types.SimpleNamespace(req_ids=["r"],sampled_token_ids=[[offset]],req_id_to_index={"r":0})
            adapter.begin(ordinal,100+offset*100)
            adapter.prepared(runner,scheduler);adapter.executed(returned_none=True)
            adapter.sampled(output,150+offset*100)
        self.f.observe()
        self.assertIs(type(self.f.provider.last_drain),F.DrainEvidence)
        result=adapter.close_run({"r":[0,1]},self.f.provider.last_drain)
        self.assertIsNotNone(result)
        self.assertEqual(result.outputs,(("r",(0,1)),))
        self.assertFalse(result.gpu_verified or result.production_qualified)


if __name__=="__main__": unittest.main(verbosity=2)
