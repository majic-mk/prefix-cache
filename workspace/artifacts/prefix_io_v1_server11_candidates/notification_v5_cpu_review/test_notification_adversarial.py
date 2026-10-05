"""Independent CPU attacks against actual C5 Queue/collector source; no CUDA imports."""
from __future__ import annotations
import ast
import hashlib
from dataclasses import dataclass,replace
import importlib.util
import os
from pathlib import Path
import queue
import sys
import threading
import time
import unittest
import weakref
from types import SimpleNamespace as NS

HERE=Path(__file__).resolve().parent
CANDIDATE=Path(os.environ.get('SERVER11_C5_CANDIDATE',HERE.parent/'p4_single_file_candidate_v5_cpu')).resolve()
REACTOR=CANDIDATE/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
COL=load('_independent_c5_collector',CANDIDATE/'native_full_step_collector.py')
NAMES={'install_p4_single_file_wait','_prefix_wake_single_file_locked','notify_p4_single_file_wait',
 '_prefix_single_file_wait_deadline','_prefix_wait_single_file_retry','_intake','publish_p4_scheduled_load'}
def methods(clock=time):
 tree=ast.parse(REACTOR.read_text(encoding='utf-8-sig'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IoReactor')
 nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_SingleFileWake']
 nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
 ns=dict(dataclass=dataclass,queue=queue,threading=threading,weakref=weakref,time=clock)
 unit=ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(unit),str(REACTOR),'exec'),ns)
 return ns
METHODS=methods()

class RawEvent:
 def __init__(self):self.records=0
 def record(self):self.records+=1
 def query(self):return True
 def elapsed_time(self,other):return 1.
class Bridge:
 mode='interference';single_file_shadow=False
 def __init__(self):self.run_id='r';self.policy=NS(single_file=object());self.fault=None;self.fail_calls=0
 def fail(self,reason):self.fail_calls+=1;self.fault=reason
class Runner:pass
class HookQueue(queue.Queue):
 def __init__(self):super().__init__();self.before_get=None;self.block_gets=0
 def get(self,block=True,timeout=None):
  if block:
   self.block_gets+=1
   if self.before_get:self.before_get()
  return super().get(block,timeout)
class Owner:
 def __init__(self):
  self._prefix_p4_bridge=Bridge();self._incoming=HookQueue();self._submit_lock=threading.Lock()
  self.intakes=[];self.deadline=time.monotonic_ns()+80_000_000
 def _prefix_single_file_wait_deadline(self,cached):return self.deadline if self.deadline>time.monotonic_ns() else None
 def _intake(self,item):
  if type(item)is METHODS['_SingleFileWake']:return METHODS['_intake'](self,item)
  self.intakes.append(item)
for name in NAMES-{'_prefix_single_file_wait_deadline','_intake'}:setattr(Owner,name,METHODS[name])

def fixture(real_observer=False,owner=None):
 owner=Owner() if owner is None else owner
 capture=COL.FullStepCapture(run_id='r',origin='native_gpu_recording',selected_offsets=(16,),action=None)
 runner=Runner();capture.runner_ref=weakref.ref(runner)
 start,end=COL.EventProxy(RawEvent(),time.monotonic_ns),COL.EventProxy(RawEvent(),time.monotonic_ns)
 start.record();start.query();frame=NS(step_kind='decode',batch=1,active_decode=1,prefill_tokens=0,context_length=144)
 pending=dict(phase='prepared',ordinal=146,frame=frame)
 if real_observer:
  copied=HERE/'frozen_observer_sources'
  pinned={'g2_worker_observation.py':'096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b',
   'p4_runtime_scalar_connector.py':'347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb'}
  for name,digest in pinned.items():
   if hashlib.sha256((copied/name).read_bytes()).hexdigest()!=digest:raise AssertionError('frozen observer source changed')
  worker=load('_independent_frozen_worker',copied/'g2_worker_observation.py')
  scalar=load('_independent_frozen_scalar',copied/'p4_runtime_scalar_connector.py')
  observer=worker.WorkerObservationConnection();observer.enabled=True
  observer.scalar=scalar.RuntimeScalarConnection();observer.scalar.enabled=True
  observer.scalar._adapter=NS(_pending=pending,frames=[None]*16,invalidate=lambda reason:None)
  observer.events=worker.QueryOnlyEventObserver('r','cpu_fixture');observer.events.active=(146,start,end)
  capture.observer=observer
 else:
  capture.observer=NS(enabled=True,events=NS(active=(146,start,end)),
   scalar=NS(enabled=True,_adapter=NS(_pending=pending)),frames=[None]*16,detach=lambda:None)
 capture.attach_single_file_wait(owner);capture.after_prepare()
 state=capture.current_single_file_step();owner._prefix_single_file_retry=((state,),None,None)
 return owner,capture,end,runner

class QueueRaceTests(unittest.TestCase):
 def setUp(self):self.o,self.c,self.end,self.runner=fixture()
 def wait_thread(self):
  box={}
  def work():
   try:box['result']=self.o._prefix_wait_single_file_retry()
   except BaseException as exc:box['error']=exc
  thread=threading.Thread(target=work,daemon=True);thread.start();return thread,box
 def join(self,t):
  t.join(1);self.assertFalse(t.is_alive(),'lost wake or deadlock')
 def test_end_before_registration_never_blocks(self):
  self.end.record();self.assertFalse(self.o._prefix_wait_single_file_retry());self.assertEqual(self.o._incoming.block_gets,0)
 def test_end_after_final_check_before_queue_get_is_retained(self):
  self.o._incoming.before_get=lambda:self.end.record()
  self.assertTrue(self.o._prefix_wait_single_file_retry());self.assertEqual(self.end.raw.records,1)
  self.assertIsNone(self.c._wait_registration);self.assertIsNone(self.o._prefix_single_file_wait_armed)
 def test_end_after_blocking_get_wakes(self):
  waiting=threading.Event();self.o._incoming.before_get=waiting.set
  t,box=self.wait_thread();self.assertTrue(waiting.wait(.5));self.end.record();self.join(t)
  self.assertTrue(box['result']);self.assertTrue(self.o._incoming.empty())
 def test_end_during_registration_lock_no_loss_or_lock_cycle(self):
  reached,release=threading.Event(),threading.Event();original=self.c.current_single_file_step;once=[True]
  def state():
   if once[0]:
    once[0]=False;reached.set();self.assertTrue(release.wait(.5))
   return original()
  self.c.current_single_file_step=state
  t,box=self.wait_thread();self.assertTrue(reached.wait(.5))
  end_thread=threading.Thread(target=self.end.record,daemon=True);end_thread.start();release.set()
  self.join(t);self.join(end_thread);self.assertNotIn('error',box);self.assertEqual(self.end.raw.records,1)
 def test_existing_queue_work_and_fifo_remain_native(self):
  self.o._incoming.put('mandatory');self.o._incoming.put('STOP')
  self.assertTrue(self.o._prefix_wait_single_file_retry());self.assertEqual(self.o.intakes,['mandatory'])
  self.assertEqual(self.o._incoming.get_nowait(),'STOP')
 def test_old_run_step_duplicate_messages_do_not_clear_current_arm(self):
  current=('r',self.c.wait_nonce,146,2);old=('old',self.c.wait_nonce,145,1)
  self.o._prefix_single_file_wait_armed=current;self.o._prefix_single_file_wake_pending=current
  for token in (old,old):
   METHODS['_intake'](self.o,METHODS['_SingleFileWake'](token))
  self.assertEqual(self.o._prefix_single_file_wait_armed,current)
  self.assertEqual(self.o._prefix_single_file_wake_pending,current)
 def test_old_pending_wake_cannot_swallow_new_end_forever(self):
  old=('r',self.c.wait_nonce,145,1)
  self.o._prefix_single_file_wake_pending=old;self.o._incoming.put(METHODS['_SingleFileWake'](old))
  self.o._incoming.before_get=lambda:self.end.record()
  self.assertTrue(self.o._prefix_wait_single_file_retry())
  self.assertFalse(self.o._prefix_wait_single_file_retry())
  self.assertIsNone(self.o._prefix_single_file_wait_armed)
 def test_duplicate_notifications_are_coalesced(self):
  token=('r',self.c.wait_nonce,146,1);self.o._prefix_single_file_wait_armed=token
  for _ in range(20):self.o.notify_p4_single_file_wait(token)
  self.assertEqual(self.o._incoming.qsize(),1)
 def test_wrong_token_arm_rejected(self):
  state=self.c.current_single_file_step()
  for token in [('wrong',self.c.wait_nonce,146,1),('r','wrong',146,1),('r',self.c.wait_nonce,147,1),('r',self.c.wait_nonce,146,True)]:
   with self.subTest(token=token):self.assertFalse(self.c.arm_single_file_wait(token,state))
 def test_fault_and_detach_unblock_real_queue(self):
  for action in ('capture_fail','bridge_fail','detach'):
   with self.subTest(action=action):
    self.o,self.c,self.end,self.runner=fixture()
    waiting=threading.Event();self.o._incoming.before_get=waiting.set;t,box=self.wait_thread()
    self.assertTrue(waiting.wait(.5))
    if action=='capture_fail':self.c.fail('injected')
    elif action=='bridge_fail':self.o._prefix_p4_bridge.fail('injected')
    else:self.c.detach()
    self.join(t);self.assertTrue(box['result']);self.assertIsNone(self.c._wait_registration)
    if action=='bridge_fail':self.assertEqual(self.o._prefix_p4_bridge.fail_calls,1)
 def test_original_intake_exception_is_not_optional_observer_failure(self):
  self.o._incoming.put('bad-native-job')
  def bad(item):raise RuntimeError('original-intake-failed')
  self.o._intake=bad
  with self.assertRaisesRegex(RuntimeError,'original-intake-failed'):self.o._prefix_wait_single_file_retry()
  self.assertIsNone(self.o._prefix_single_file_wait_armed)
 def test_notification_failure_does_not_skip_original_event_record(self):
  def bad(token=None):raise RuntimeError('broken-notifier')
  self.o.notify_p4_single_file_wait=bad
  token=('r',self.c.wait_nonce,146,1);self.assertTrue(self.c.arm_single_file_wait(token,self.c.current_single_file_step()))
  self.end.record();self.assertEqual(self.end.raw.records,1)
 def test_original_deadline_timeout_ends_wait_without_end_event(self):
  self.o.deadline=time.monotonic_ns()+5_000_000
  self.assertTrue(self.o._prefix_wait_single_file_retry());self.assertIsNone(self.c._wait_registration)
  self.assertIsNone(self.o._prefix_single_file_wait_armed)
  self.assertEqual(self.end.raw.records,0)
 def test_off_uninstalled_does_not_touch_clock_queue_or_capture(self):
  owner=Owner()
  def bad(*a,**k):raise AssertionError('off entered optional waiting')
  owner._incoming.get=bad;owner._prefix_single_file_wait_deadline=bad
  self.assertFalse(owner._prefix_wait_single_file_retry())
 def test_detach_restores_only_its_own_bridge_wrapper(self):
  bridge=self.o._prefix_p4_bridge
  replacement=lambda reason:None;bridge.fail=replacement;self.c.detach()
  self.assertIs(bridge.fail,replacement)
 def test_end_record_precedes_queue_publication(self):
  original=self.o.notify_p4_single_file_wait;seen=[]
  def notify(token):
   seen.append((self.end.raw.records,self.end.record_after_ns));original(token)
  self.o.notify_p4_single_file_wait=notify
  self.o._incoming.before_get=self.end.record
  self.assertTrue(self.o._prefix_wait_single_file_retry())
  self.assertEqual(len(seen),1);self.assertEqual(seen[0][0],1);self.assertIsNotNone(seen[0][1])
 def test_original_record_error_survives_notification_failure(self):
  def bad_record():self.end.raw.records+=1;raise ValueError('native-record-failed')
  def bad_notify(token):raise RuntimeError('optional-notify-failed')
  self.end.raw.record=bad_record;self.o.notify_p4_single_file_wait=bad_notify
  token=('r',self.c.wait_nonce,146,1)
  self.assertTrue(self.c.arm_single_file_wait(token,self.c.current_single_file_step()))
  with self.assertRaisesRegex(ValueError,'native-record-failed'):self.end.record()
  self.assertEqual(self.end.raw.records,1);self.assertTrue(self.c._wait_failed)
  self.assertIsNone(self.c._wait_registration)
 def test_unarmed_end_and_fail_do_not_publish_empty_control_messages(self):
  self.end.record();self.c.fail('closed');self.o._prefix_p4_bridge.fail('closed')
  self.assertTrue(self.o._incoming.empty())
 def test_disarm_failure_does_not_consume_original_message(self):
  def fail_disarm(token):raise RuntimeError('optional-disarm-failed')
  self.c.disarm_single_file_wait=fail_disarm;self.o._incoming.put('native-job')
  self.assertTrue(self.o._prefix_wait_single_file_retry());self.assertEqual(self.o.intakes,['native-job'])
  self.assertIsNone(self.o._prefix_single_file_wait_armed)
 def test_disarm_failure_does_not_mask_original_intake_error(self):
  def fail_disarm(token):raise RuntimeError('optional-disarm-failed')
  def fail_intake(item):raise ValueError('native-intake-failed')
  self.c.disarm_single_file_wait=fail_disarm;self.o._intake=fail_intake;self.o._incoming.put('native-job')
  with self.assertRaisesRegex(ValueError,'native-intake-failed'):self.o._prefix_wait_single_file_retry()
  self.assertIsNone(self.o._prefix_single_file_wait_armed)
 def test_frozen_observer_fault_entrypoints_wake_without_end_record(self):
  for kind in ('invalidate','disable','observe','original_failed','detach','scalar_fail','scalar_observe','scalar_original_failed','scalar_detach'):
   with self.subTest(kind=kind):
    self.o,self.c,self.end,self.runner=fixture(real_observer=True);observer=self.c.observer
    def bad():raise ValueError('observer-fault')
    actions={'invalidate':lambda:observer.invalidate('fault'),'disable':observer.disable,
     'observe':lambda:observer.observe(bad),'original_failed':lambda:observer._original_failed('fault'),
     'detach':observer.detach,'scalar_fail':lambda:observer.scalar._fail('fault'),
     'scalar_observe':lambda:observer.scalar._observe(bad),
     'scalar_original_failed':lambda:observer.scalar._original_failed('fault'),
     'scalar_detach':observer.scalar.detach}
    waiting=threading.Event();self.o._incoming.before_get=waiting.set;t,box=self.wait_thread()
    self.assertTrue(waiting.wait(.5));actions[kind]();self.join(t)
    self.assertTrue(box['result']);self.assertIsNone(self.c.current_single_file_step())
    self.assertEqual(self.end.raw.records,0);self.assertTrue(self.o._incoming.empty())
    self.assertFalse(self.o._prefix_wait_single_file_retry())
 def test_frozen_observer_wrappers_restored_and_foreign_override_preserved(self):
  self.o,self.c,self.end,self.runner=fixture(real_observer=True);observer=self.c.observer
  self.assertIn('invalidate',vars(observer));self.assertIn('_fail',vars(observer.scalar))
  foreign=lambda reason:None;observer.invalidate=foreign;self.c.detach()
  self.assertIs(observer.invalidate,foreign)
  self.assertIs(observer.scalar._fail.__func__,type(observer.scalar)._fail)
  self.assertIs(observer.scalar.detach.__func__,type(observer.scalar).detach)
 def test_scheduled_publication_and_conflict_after_check_wake(self):
  load('_scheduled_fixture',CANDIDATE/'test_single_file_retry_observation.py')
  from prefix_io_control.p4_load_observation import SchedulerLoadUnavailable
  for conflict in (False,True):
   with self.subTest(conflict=conflict):
    self.o,self.c,self.end,self.runner=fixture();self.o._closed=False
    if conflict:
     self.o._prefix_p4_load_frame=SchedulerLoadUnavailable('r',1,time.monotonic_ns(),'previous')
     self.o._prefix_p4_load_sequence=1
    frame=SchedulerLoadUnavailable('r',1,time.monotonic_ns(),'changed')
    def publish():
     if conflict:
      with self.assertRaisesRegex(ValueError,'conflicting'):self.o.publish_p4_scheduled_load(frame)
     else:self.assertTrue(self.o.publish_p4_scheduled_load(frame))
    self.o._incoming.before_get=publish
    self.assertTrue(self.o._prefix_wait_single_file_retry());self.assertTrue(self.o._incoming.empty())
    self.assertEqual(self.end.raw.records,0)
    self.assertIs(self.o._prefix_p4_load_frame,None if conflict else frame)
 def test_wake_put_failure_is_bounded_and_disables_future_parking(self):
  token=('r',self.c.wait_nonce,146,1);self.o._prefix_single_file_wait_armed=token
  def bad_put(item):raise RuntimeError('queue-put-failed')
  self.o._incoming.put=bad_put;self.o.notify_p4_single_file_wait(token)
  self.assertTrue(self.o._prefix_single_file_wait_faulted)
  self.assertIsNone(self.o._prefix_single_file_wake_pending)
  self.assertIsNone(METHODS['_prefix_single_file_wait_deadline'](self.o,self.o._prefix_single_file_retry))

class SourceBoundaryTests(unittest.TestCase):
 def test_original_intake_tail_is_identical(self):
  previous=Path(os.environ.get('SERVER11_PREVIOUS_CANDIDATE_ROOT',CANDIDATE.with_name('p4_single_file_candidate_v4')))/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
  def intake(path):
   tree=ast.parse(path.read_text(encoding='utf-8-sig'))
   owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IoReactor')
   return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_intake')
  old,new=intake(previous),intake(REACTOR)
  self.assertEqual(len(new.body),len(old.body)+1)
  new.body=new.body[1:]
  self.assertEqual(ast.dump(old),ast.dump(new),'original STOP/mandatory/native intake changed')
 def test_original_decision_calls_never_use_released_capacity_mode(self):
  tree=ast.parse(REACTOR.read_text(encoding='utf-8-sig'))
  owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IoReactor')
  functions={n.name:n for n in owner.body if isinstance(n,ast.FunctionDef)}
  key=functions['_prefix_single_file_retry_key']
  self.assertEqual(key.args.kwonlyargs[0].arg,'released_slot')
  self.assertIs(key.args.kw_defaults[0].value,False)
  special=[]
  for name,fn in functions.items():
   for node in ast.walk(fn):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='_prefix_single_file_retry_key':
     if any(kw.arg=='released_slot' for kw in node.keywords):special.append(name)
  self.assertEqual(special,['_prefix_single_file_wait_deadline'])

class NativeQueueIntegrationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  # Reuse only retained-cache synthetic backend fixture, never its historical
  # C3->C4 assertions. Native intake/submission/mandatory/STOP remain actual C5 AST.
  cls.support=load('_independent_c5_native_queue',CANDIDATE/'test_retained_idle_wait.py')
  cls.support.EXTRACTED['IoReactor']._intake.__globals__['_SingleFileWake']=METHODS['_SingleFileWake']
 def owner(self,shared=True):
  owner=self.support.Owner(shared=shared)
  for name in NAMES-{'_intake','_prefix_single_file_wait_deadline'}:setattr(owner,name,METHODS[name].__get__(owner,type(owner)))
  owner._prefix_p4_bridge=Bridge();owner.deadline=time.monotonic_ns()+80_000_000
  owner._prefix_single_file_wait_deadline=lambda cached:None if owner._stop else owner.deadline
  # Qualified state/predicate integration is exercised separately. This fixture
  # isolates original routing and release semantics at a synthetic park boundary.
  _,capture,end,runner=fixture(owner=owner)
  return owner,capture,end,runner
 def test_real_submit_mandatory_fifo_and_stop_preserve_native_ownership(self):
  owner,capture,end,runner=self.owner();job=self.support.job()
  owner.submit_job(job);self.assertTrue(owner.request_mandatory([job.future]));owner.shutdown(wait=False)
  self.assertTrue(owner._prefix_wait_single_file_retry())
  self.assertIs(owner._active[0],job);self.assertEqual(owner.staging_pool.released,[])
  self.assertFalse(job.future.done())
  mandatory=owner._incoming.get_nowait();owner._intake(mandatory)
  self.assertIn(job.future,owner._prefix_progress.required)
  stop=owner._incoming.get_nowait();self.assertIs(stop,owner._STOP);owner._intake(stop)
  self.assertEqual(owner.staging_pool.released,[7]);self.assertFalse(job.future.done())
  self.assertFalse(owner._prefix_wait_single_file_retry());owner._pump_once()
  owner._cleanup_retained_cache_after_stop();self.assertEqual(owner.staging_pool.released,[7])
  self.assertTrue(job.future.done());self.assertEqual(owner._accepted_parent_count,0)
 def test_real_stop_unblocks_get_retained_shared_and_plain_release_once(self):
  for shared in (False,True):
   with self.subTest(shared=shared):
    owner,capture,end,runner=self.owner(shared);box={}
    def wait():box['result']=owner._prefix_wait_single_file_retry()
    thread=threading.Thread(target=wait,daemon=True);thread.start()
    self.assertTrue(owner._incoming.entered.wait(.5));owner.shutdown(wait=False);thread.join(1)
    self.assertFalse(thread.is_alive());self.assertTrue(box['result']);self.assertTrue(owner._stop)
    self.assertEqual(owner.staging_pool.released,[7]);self.assertEqual(end.raw.records,0)
    owner._cleanup_retained_cache_after_stop();self.assertEqual(owner.staging_pool.released,[7])

class OriginalPredicateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f=load('_independent_c5_retry_fixture',CANDIDATE/'test_single_file_retry_observation.py')
  cls.ns=methods(cls.f.NSOURCE['time'])
 def prime(self):
  o=self.f.Owner(max_wait=100_000_000)
  o.bridge.policy.config=replace(o.bridge.policy.config,sample_max_age_ns=100_000_000)
  self.assertFalse(o.repeat());return o
 def deadline(self,o):return self.ns['_prefix_single_file_wait_deadline'](o,o._prefix_single_file_retry)
 def test_original_released_capacity_matches_cached_reserve_once(self):
  o=self.prime();free=o.staging_pool.free_count;reserves=o.staging_pool.reserves;releases=o.staging_pool.releases
  self.assertIsNotNone(self.deadline(o))
  self.assertEqual((o.staging_pool.free_count,o.staging_pool.reserves,o.staging_pool.releases),(free,reserves,releases))
  o.staging_pool.free_count-=1;self.assertIsNone(self.deadline(o))
 def test_each_actual_progress_or_invalid_state_forbids_parking(self):
  changes=[
   lambda o:o._active.append(object()),lambda o:o._inflight.update(x=object()),
   lambda o:o._pending_copies.append(object()),lambda o:o._copy_ready.append(object()),
   lambda o:o._ready_fds_load.append(object()),lambda o:o._preload_pending.append(object()),
   lambda o:setattr(o,'_open_inflight',1),lambda o:setattr(o,'_data_inflight',1),
   lambda o:setattr(o,'parents',1),lambda o:setattr(o,'_stop',True),
   lambda o:setattr(o,'_native_drain_unknown',True),lambda o:setattr(o,'_native_fatal_reason','bad'),
   lambda o:setattr(o._prefix_stage_accounting,'valid',False),lambda o:setattr(o._prefix_stage_accounting,'journal_valid',False),
   lambda o:setattr(o,'_prefix_p4_load_sequence',1),lambda o:setattr(o,'_prefix_p4_load_frame',object()),
   lambda o:setattr(o.bridge,'fault','bad'),lambda o:setattr(o.bridge,'single_file_shadow',True),
   lambda o:o._preload_waiters.update({self.f.HASH:[(NS(future='mandatory'),0)]}),
   lambda o:o._ready_fds_preload.append(o.ready),
   lambda o:o._prefix_stage_accounting.stats['ssd_read'].__setitem__('inflight_bytes',917504),
   lambda o:setattr(o.staging_pool,'_closed',True)]
  for index,change in enumerate(changes):
   with self.subTest(index=index):
    o=self.prime();change(o);self.assertIsNone(self.deadline(o))
 def test_original_age_bound_never_reset_by_repeat_or_check(self):
  o=self.prime();key,snapshot,work=o._prefix_single_file_retry
  original=(o.ready.open_start_ns,snapshot.monotonic_ns,snapshot.native_state.captured_ns,key[0][2])
  expected=min(x+100_000_000 for x in original)
  self.assertEqual(self.deadline(o),expected)
  self.f.CLOCK[0]=expected-1;self.assertEqual(self.deadline(o),expected)
  self.f.CLOCK[0]=expected;self.assertIsNone(self.deadline(o))
  self.assertEqual((o.ready.open_start_ns,snapshot.monotonic_ns,snapshot.native_state.captured_ns,key[0][2]),original)
 def test_every_accounted_stage_inflight_ops_and_bytes_forbid_parking(self):
  for stage in self.f.STAGES:
   for metric in ('inflight_ops','inflight_bytes'):
    with self.subTest(stage=stage,metric=metric):
     owner=self.prime();owner._prefix_stage_accounting.stats[stage][metric]=1
     self.assertIsNone(self.deadline(owner))

if __name__=='__main__':
 unittest.main(verbosity=2)

