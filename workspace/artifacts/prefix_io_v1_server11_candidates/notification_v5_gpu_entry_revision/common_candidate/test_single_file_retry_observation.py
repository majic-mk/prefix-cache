"""CPU replay of real source-extracted collect/reserve/preview/drain paths."""
from __future__ import annotations
import ast
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_retry_cpu_policy_support', HERE/'test_single_file_policy.py')
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
from prefix_io_control.p4_bridge import NativeP4Bridge
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_types import P4Config
from prefix_io_control.dispatch_budget import STAGES

SOURCE = HERE/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
parsed = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
native = next(x for x in parsed.body if isinstance(x, ast.ClassDef) and x.name=='IoReactor')
NAMES = {'_prefix_stage_decide','_prefix_ready_read_decision','_prefix_single_file_retry_key',
    '_drain_ready_preload_fds','_reserve_preload_slot','_reserve_foreground_slot',
    '_prefix_p4_note_reservation','_prefix_p4_collect','_prefix_p4_load_values',
    '_prefix_p4_existing_io_state','_prefix_capacity_components','_prefix_clean_reclaimable_bytes',
    '_prefix_invalidate_capacity','_has_real_load_pressure','_can_issue_speculative_preload'}
nodes = [x for x in parsed.body if isinstance(x, ast.ClassDef) and x.name in
         {'_ReadyFd','_PreloadInfo','_StageDispatch','_P4Deferred'}]
nodes += [x for x in native.body if isinstance(x, ast.FunctionDef) and x.name in NAMES]
CLOCK = [1000]
NSOURCE = {'dataclass':dataclass, 'time':NS(monotonic_ns=lambda:CLOCK[0]), 'threading':threading}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), str(SOURCE),'exec'),NSOURCE)
IO = 917504; HASH = b'x'*32

class Pool:
    def __init__(self): self.slot_count=self.free_count=16; self._closed=False; self.reserves=self.releases=0
    def try_reserve(self):
        self.free_count-=1; self.reserves+=1
        return NS(index=3)
    def release(self,index):
        assert index==3; self.free_count+=1; self.releases+=1

class Owner:
    def __init__(self, *, off=False, shadow=False, max_wait=500000):
        CLOCK[0]=1000
        receipt=support.fixture_receipt()
        self.live=(5,10,20,1,1,0,144)
        self.bridge=NativeP4Bridge(P4Policy('r',P4Config('interference',1000000,max_wait,100),single_file=receipt))
        self.bridge.bind();self.bridge.single_file_shadow=shadow
        self.bridge._single_file_runtime=NS(signature_prefix=receipt.signature[:4])
        self.bridge._single_file_state=lambda:self.live
        self._prefix_p4_bridge=None if off else self.bridge
        self._prefix_dispatch_controller=None
        self._prefix_stage_accounting=NS(valid=True,journal_valid=True,event_sequence=0,_owner=lambda:None,
            stats={s:{'inflight_ops':0,'inflight_bytes':0} for s in STAGES})
        self._active=[];self._inflight={};self._pending_copies=[];self._copy_ready=[]
        self._ready_fds_load=deque();self._preload_pending=deque()
        info=NSOURCE['_PreloadInfo'](HASH,'preload','request','cpu-fixture',0,1)
        self.ready=NSOURCE['_ReadyFd'](fd=7,job=None,file_index=-1,preload_hash=HASH,
            open_start_ns=100,preload_info=info,sequence=1)
        self._ready_fds_preload=deque([self.ready]);self._ready_sequence=1
        self._preload_waiters={};self._open_inflight=0;self._data_inflight=0;self.iodepth=4
        # A real opened, not-yet-read speculative preload retains this native count.
        self._preload_inflight_total=1;self._preload_inflight_hashes={HASH:1}
        self._shared_cached={};self._preload_slots={};self._preload_cached_total=0
        self._prefix_shared_cached_pinned_slots=0;self._prefix_shared_cached_observed_count=0
        self._prefix_capacity_valid=True;self._prefix_capacity_error=None
        self._staging_cache=None;self.staging_pool=Pool();self.file_store=NS(io_size=IO)
        self.layout=NS(storage_block_bytes=IO,bytes_per_kernel_block=(IO,))
        self._stop=False;self._native_drain_unknown=False;self._native_fatal_reason=None
        self._worker=NS(ident=threading.get_ident());self._submit_lock=threading.Lock()
        self._prefix_p4_load_frame=None;self._prefix_p4_load_sequence=0
        self.collects=0;self.submitted=[];self.parents=0
    def accepted_parent_count(self):return self.parents
    def is_mandatory(self,future):return future=='mandatory'
    def _has_preload_waiter(self,key):return bool(self._preload_waiters.get(key))
    def _evict_one_cache_slot(self,reason):raise AssertionError('unexpected eviction')
    def _evict_one_preload_slot(self):raise AssertionError('unexpected eviction')
    def _submit_read_from_ready(self,ready,slot_index,**kwargs):self.submitted.append((ready.sequence,slot_index))
    def _prefix_p4_collect(self):
        self.collects+=1
        return NSOURCE['_prefix_p4_collect'](self)
    def repeat(self):
        result=self._drain_ready_preload_fds()
        CLOCK[0]+=100
        return result

for name in NAMES-{'_prefix_p4_collect'}:setattr(Owner,name,NSOURCE[name])

class RetryObservationTests(unittest.TestCase):
    def prime(self):
        value=Owner();self.assertFalse(value.repeat())
        self.assertIsNotNone(value._prefix_single_file_retry)
        return value
    def test_real_collect_reserve_preview_queue_replay_reuses_once_sampled_values(self):
        owner=self.prime();cached=owner._prefix_single_file_retry
        for _ in range(154):self.assertFalse(owner.repeat())
        self.assertEqual(owner.collects,1)
        self.assertEqual(owner.bridge.single_file_previews,155)
        self.assertEqual(owner.bridge.single_file_blocked_attempts,155)
        self.assertEqual(owner.bridge.single_file_observation_reuses,154)
        self.assertEqual(owner.staging_pool.reserves,155)
        self.assertEqual(owner.staging_pool.releases,155)
        self.assertEqual(owner.staging_pool.free_count,16)
        self.assertEqual(owner._preload_inflight_total,1)
        self.assertIs(owner._ready_fds_preload[0],owner.ready)
        self.assertEqual(owner.ready.open_start_ns,100)
        self.assertIs(owner._prefix_single_file_retry[1],cached[1])
        self.assertIs(owner._prefix_single_file_retry[2],cached[2])
        self.assertEqual((cached[1].monotonic_ns,cached[1].native_state.captured_ns),(1000,1000))
    def test_new_mandatory_waiter_uses_original_foreground_issue(self):
        owner=self.prime();owner._preload_waiters[HASH]=[(NS(future='mandatory'),0)]
        self.assertTrue(owner.repeat())
        self.assertEqual(owner.submitted,[(1,3)])
        self.assertEqual(owner.bridge.single_file_observation_reuses,0)
        self.assertEqual(owner.bridge.last_preview.reason,'native_progress_override')
    def test_new_nonmandatory_waiter_disables_reuse(self):
        owner=self.prime();owner._preload_waiters[HASH]=[(NS(future='ordinary'),0)]
        owner.repeat();self.assertEqual(owner.collects,2)
        self.assertIsNone(owner._prefix_single_file_retry)
    def test_context_end_or_change_cannot_reuse(self):
        for state in (None,(5,10,20,1,1,0,145),(6,10,20,1,1,0,144)):
            owner=self.prime();owner.live=state;owner.repeat()
            self.assertEqual(owner.collects,2)
            self.assertEqual(owner.bridge.single_file_observation_reuses,0)
            if state is None or state[-1]==145:self.assertEqual(owner.submitted,[(1,3)])
    def test_journal_sequence_change_requires_real_new_collection(self):
        owner=self.prime();owner._prefix_stage_accounting.event_sequence+=2
        owner.repeat();self.assertEqual(owner.collects,2)
        self.assertEqual(owner.bridge.single_file_observation_reuses,0)
    def test_new_io_disables_cache_and_preserves_existing_policy_fallback(self):
        owner=self.prime();owner._prefix_stage_accounting.stats['ssd_read']['inflight_ops']=1
        owner._prefix_stage_accounting.stats['ssd_read']['inflight_bytes']=IO
        owner.repeat();self.assertEqual(owner.collects,2)
        self.assertEqual(owner.submitted,[(1,3)])
        self.assertIsNone(owner._prefix_single_file_retry)
    def test_capacity_change_recollects_without_refreshing_old_cache(self):
        owner=self.prime();old=owner._prefix_single_file_retry[1]
        owner.staging_pool.free_count-=1;owner.repeat()
        self.assertEqual(owner.collects,2);self.assertEqual(owner.bridge.single_file_observation_reuses,0)
        self.assertEqual(old.monotonic_ns,1000)
    def test_max_wait_issues_and_does_not_extend_original_arrival(self):
        owner=self.prime();CLOCK[0]=500100;self.assertTrue(owner.repeat())
        self.assertEqual(owner.bridge.last_preview.reason,'native_progress_override')
        self.assertEqual(owner.ready.open_start_ns,100)
    def test_original_sample_freshness_not_extended(self):
        owner=Owner(max_wait=2000000);owner.repeat()
        self.assertIsNotNone(owner._prefix_single_file_retry)
        CLOCK[0]=1000101
        owner.repeat();self.assertEqual(owner.bridge.single_file_observation_reuses,0)
        self.assertIsNone(owner._prefix_single_file_retry)
        self.assertEqual(owner.bridge.last_preview.reason,'outside_verified_single_file_condition')
    def test_new_parent_work_excludes_observation_reuse(self):
        owner=self.prime();owner.parents=1
        owner._active.append(NS(accepted_parent_sequence=2,accepted_parent_retired=False,
            is_store=True,failed=None,next_file_index=0,total_files=1,done_files=0,
            inflight_files=0,future_set=False,future='ordinary',profile=NS(start_ns=500)))
        owner.repeat();self.assertEqual(owner.collects,2)
        self.assertEqual(owner.bridge.single_file_observation_reuses,0)
        self.assertIsNone(owner._prefix_single_file_retry)
    def test_scheduler_publication_and_epoch_change_disable_reuse(self):
        for name,value in (('_prefix_p4_load_sequence',1),('_prefix_p4_load_frame',object())):
            owner=self.prime();setattr(owner,name,value);owner.repeat()
            self.assertEqual(owner.collects,2);self.assertIsNone(owner._prefix_single_file_retry)
        owner=self.prime();owner.bridge._epoch+=1;owner.repeat()
        self.assertEqual(owner.collects,2);self.assertEqual(owner.bridge.single_file_observation_reuses,0)
    def test_stop_continuation_and_geometry_do_not_use_cached_work(self):
        for condition in ('stop','continuation','geometry','stage'):
            owner=self.prime();kwargs=dict(ready=owner.ready,reserved_bytes=IO)
            stage='ssd_read';nbytes=IO
            if condition=='stop':owner._stop=True
            if condition=='continuation':kwargs['continuation']=True
            if condition=='geometry':nbytes=2*IO
            if condition=='stage':stage='h2d'
            owner._prefix_stage_decide(stage,nbytes,('test',1),**kwargs)
            self.assertEqual(owner.bridge.single_file_observation_reuses,0)
            self.assertIsNone(owner._prefix_single_file_retry)
    def test_ready_identity_and_reopen_age_change_recollect(self):
        for attr,value in (('sequence',2),('open_start_ns',200),('fd',8)):
            owner=self.prime();setattr(owner.ready,attr,value);owner.repeat()
            self.assertEqual(owner.collects,2);self.assertEqual(owner.bridge.single_file_observation_reuses,0)
    def test_shadow_and_off_never_create_or_use_repeat_cache(self):
        for kwargs in ({'shadow':True},{'off':True}):
            owner=Owner(**kwargs);self.assertTrue(owner.repeat())
            self.assertIsNone(getattr(owner,'_prefix_single_file_retry',None))
            self.assertEqual(owner.bridge.single_file_observation_reuses,0)
            self.assertEqual(owner.bridge.single_file_blocked_attempts,0)
    def test_cache_is_immutable_values_without_owners(self):
        owner=self.prime()
        def walk(value):
            if value is None or type(value) in (str,int,bool,bytes):return
            if type(value) in (tuple,frozenset):
                for item in value:walk(item)
                return
            self.assertTrue(is_dataclass(value));self.assertTrue(value.__dataclass_params__.frozen)
            for field in fields(value):walk(getattr(value,field.name))
        walk(owner._prefix_single_file_retry)
    def test_fault_and_unknown_accounting_never_reuse(self):
        for which in ('fault','valid','journal_valid'):
            owner=self.prime()
            if which=='fault':owner.bridge.fail('cpu fixture fault')
            else:setattr(owner._prefix_stage_accounting,which,False)
            owner.repeat();self.assertEqual(owner.bridge.single_file_observation_reuses,0)
            self.assertIsNone(owner._prefix_single_file_retry)

if __name__=='__main__':unittest.main()
