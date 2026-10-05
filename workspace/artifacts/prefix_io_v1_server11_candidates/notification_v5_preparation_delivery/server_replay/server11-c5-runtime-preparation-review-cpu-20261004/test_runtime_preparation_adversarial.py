"""Independent actual-callsite and real Queue attacks. Backend/events are synthetic."""
from __future__ import annotations
import ast
from copy import deepcopy
import gc
import importlib.util
import os
from pathlib import Path
import queue
import sys
import threading
import time
import types
import unittest
import weakref
from types import SimpleNamespace as NS

HERE=Path(__file__).resolve().parent
PREP=Path(os.environ.get('SERVER11_C5_PREPARATION',HERE.parent/'notification_v5_runtime_preparation')).resolve()
CANDIDATE=Path(os.environ.get('SERVER11_C5_CANDIDATE',HERE.parent/'p4_single_file_candidate_v5_cpu')).resolve()
RACE=Path(os.environ.get('SERVER11_NOTIFICATION_RACE_REVIEW',HERE.parent/'notification_v5_cpu_review')).resolve()
REACTOR=CANDIDATE/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
COLLECTOR=CANDIDATE/'native_full_step_collector.py'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module
A=load('_independent_preparation_adapter',PREP/'notification_runtime_adapter.py')
F=load('_independent_preparation_wait_fixture',CANDIDATE/'test_single_file_wait.py')
sys.modules[F.collector.FullStepCapture.__module__]=F.collector
W=load('_independent_preparation_worker',RACE/'frozen_observer_sources/g2_worker_observation.py')
S=load('_independent_preparation_scalar',RACE/'frozen_observer_sources/p4_runtime_scalar_connector.py')
# Reconstitute only the actual frozen dataclass AST in a source-labelled CPU
# module. This is explicitly a fixture, not a real imported native engine.
native=types.ModuleType('_independent_frozen_reactor_ast');native.__file__=str(REACTOR)
sys.modules[native.__name__]=native;native.__dict__.update(F.G);native.__dict__['__name__']=native.__name__
tree=ast.parse(REACTOR.read_text(encoding='utf-8-sig'))
wake_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_SingleFileWake')
exec(compile(ast.fix_missing_locations(ast.Module(body=[wake_node],type_ignores=[])),str(REACTOR),'exec'),native.__dict__)
F.Wake=native._SingleFileWake;F.G['_SingleFileWake']=F.Wake

def fixture(mode='on',do_attach=True):
 owner=F.WaitOwner(clock=time.monotonic_ns,max_wait=100_000_000,sample_age=100_000_000,
  off=(mode=='off'),shadow=(mode=='shadow'))
 owner._incoming=queue.Queue()
 capture,start,end=F.make_capture(owner,attach=False)
 old=capture.observer;observer=W.WorkerObservationConnection();observer.enabled=True
 scalar=S.RuntimeScalarConnection();scalar.enabled=True;scalar._adapter=old.scalar._adapter
 scalar._adapter.frames=[];scalar._adapter.invalidate=lambda reason:None
 observer.scalar=scalar;observer.events=W.QueryOnlyEventObserver('r','cpu_fixture');observer.events.active=old.events.active
 capture.observer=observer;bridge=owner._prefix_p4_bridge
 owner.bridge._single_file_capture=weakref.ref(capture) # Synthetic bridge metadata only.
 if not do_attach:return owner,capture,start,end,None
 audit=A.prepare_notification_capture(mode,owner,bridge,capture,run_id='r',request_id='r-p0-B',
  reactor_source=REACTOR,collector_source=COLLECTOR,wake_type=F.Wake)
 capture.after_prepare()
 if mode=='on':
  assert owner._drain_ready_preload_fds() is False
  assert owner._prefix_single_file_retry is not None
 return owner,capture,start,end,audit

def finish(capture,audit):
 capture.detach();audit.close();return audit.export()

class AdapterAttacks(unittest.TestCase):
 def test_end_between_final_current_check_and_get_is_valid_matching_wake(self):
  owner,capture,start,end,audit=fixture()
  original=capture.single_file_wait_current
  def checked_then_end(*args):
   result=original(*args);end.record();return result
  capture.single_file_wait_current=checked_then_end
  self.assertTrue(owner._prefix_wait_single_file_retry())
  document=finish(capture,audit)
  result=A.validate_notification_evidence(document,mode='on',run_id='r',request_id='r-p0-B')
  self.assertTrue(result['notification_exercised'])
  self.assertEqual(document['get_calls'],1);self.assertEqual(document['rows'][0]['outcome'],'matching_wake')
  self.assertIsNone(document['rows'][0]['capture_registration'])
 def test_none_registration_cannot_promote_native_message_old_wake_or_timeout(self):
  owner,capture,start,end,audit=fixture();original=capture.single_file_wait_current
  def checked_then_end(*args):
   result=original(*args);end.record();return result
  capture.single_file_wait_current=checked_then_end;owner._prefix_wait_single_file_retry()
  doc=finish(capture,audit)
  for outcome,returned in (('returned',None),('old_wake',['r','old',4,1]),('original_deadline_timeout',None)):
   candidate=deepcopy(doc);row=candidate['rows'][0];row['outcome']=outcome;row['returned_token']=returned
   row['ended_ns']=max(row['ended_ns'],row['original_deadline_ns'])
   with self.subTest(outcome=outcome):
    with self.assertRaises(ValueError):A.validate_notification_evidence(candidate,mode='on',run_id='r',request_id='r-p0-B')
 def test_detach_does_not_hand_clear_arm_and_drain_must_precede_export(self):
  owner,capture,start,end,audit=fixture();entered=threading.Event();resume=threading.Event();errors=[]
  original=audit._before_get
  def pause_before_get(*args):
   row=original(*args);entered.set()
   if not resume.wait(2):raise AssertionError('test producer failed to release get')
   return row
  audit._before_get=pause_before_get
  def producer():
   try:
    self.assertTrue(entered.wait(2));capture.detach()
    self.assertIsNotNone(owner._prefix_single_file_wait_armed)
    self.assertEqual(audit.rows[0]['outcome'],'pending')
    early=audit.export()
    with self.assertRaises(ValueError):A.validate_notification_evidence(early,mode='on',run_id='r',request_id='r-p0-B')
   except BaseException as exc:errors.append(exc)
   finally:resume.set()
  thread=threading.Thread(target=producer);thread.start()
  self.assertTrue(owner._prefix_wait_single_file_retry());thread.join(2)
  self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
  self.assertIsNone(owner._prefix_single_file_wait_armed);audit.close();doc=audit.export()
  self.assertTrue(doc['reactor_arm_clear']);self.assertEqual(doc['rows'][0]['outcome'],'matching_wake')
  A.validate_notification_evidence(doc,mode='on',run_id='r',request_id='r-p0-B')
 def test_source_qualified_native_class_and_actual_original_get_once(self):
  owner,capture,start,end,audit=fixture();original_get=queue.Queue.get;calls=[]
  # The wrapper has already captured the original unbound Queue.get; replacing
  # the class attribute now must neither replace it nor duplicate native work.
  def prohibited(*args,**kwargs):calls.append((args,kwargs));raise AssertionError('replacement called')
  queue.Queue.get=prohibited
  try:
   owner._incoming.put('once');self.assertEqual(owner._incoming.get(False),'once')
   self.assertEqual(owner._incoming.qsize(),0);self.assertEqual(calls,[])
  finally:queue.Queue.get=original_get;finish(capture,audit)
 def test_actual_launcher_measured_callsite_uses_shared_helper_and_owner_id(self):
  source=(PREP/'run_p4_single_file_experiment.py').read_text(encoding='utf-8');tree=ast.parse(source)
  def assigned(name):return [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
  install=next(n for n in assigned('active_capture') if isinstance(n.value,ast.Call))
  attach=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='attach_single_file_capture' for c in ast.walk(n)) and n.lineno>install.lineno)
  audit_node=next(n for n in assigned('notification_audit') if isinstance(n.value,ast.Call))
  for mode in ('off','shadow','on'):
   owner,capture,start,end,_=fixture(mode,do_attach=False);calls=[]
   def install_fixture(*args,**kwargs):
    calls.append(('install',kwargs['run_id']));capture.run_id=kwargs['run_id'];return capture
   bridge=owner._prefix_p4_bridge
   if bridge is not None:
    def bind(self,cap,**kwargs):
     calls.append(('original_bridge_attach',kwargs['shadow']));self._single_file_capture=weakref.ref(cap)
    bridge.attach_single_file_capture=types.MethodType(bind,bridge)
   env=dict(collector=NS(install=install_fixture),worker=object(),common=object(),root=CANDIDATE,
    refs={},collector_source_view=lambda refs:refs,LABEL='r',torch=NS(cuda=NS(Event=object)),
    action=lambda *args:None,condition='B',bridge=bridge,runtime_identity=object(),mode=mode,
    notification_adapter=lambda:A,handler=NS(coordinator=NS(reactor=owner)),rid='r-p0-B',
    REACTOR=REACTOR.relative_to(CANDIDATE),config={'collector_relative':'native_full_step_collector.py'},_SingleFileWake=F.Wake)
   nodes=[deepcopy(install),deepcopy(attach),deepcopy(audit_node)]
   exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(PREP/'run_p4_single_file_experiment.py'),'exec'),env)
   audit=env['notification_audit'];self.assertEqual(calls[0],('install','r'));self.assertEqual(audit.request_id,'r-p0-B')
   self.assertEqual(audit.installed,mode=='on')
   if mode!='off':self.assertEqual(calls[1],('original_bridge_attach',mode=='shadow'))
   finish(capture,audit)
 def test_actual_launcher_success_and_exception_cleanup_order(self):
  tree=ast.parse((PREP/'run_p4_single_file_experiment.py').read_bytes())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_window')
  calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)]
  def lines(name):return sorted(n.lineno for n in calls if (isinstance(n.func,ast.Name) and n.func.id==name) or (isinstance(n.func,ast.Attribute) and n.func.attr==name))
  prepare=lines('prepare_notification_capture')[0];frontend=lines('frontend_capture')[0]
  detach=next(x for x in lines('detach') if x>frontend);drain=next(x for x in lines('original_drain') if x>detach)
  close=next(x for x in lines('close') if x>detach);validate=next(x for x in lines('validate_notification_evidence') if x>detach)
  self.assertLess(prepare,frontend);self.assertLess(detach,drain);self.assertLess(drain,close);self.assertLess(close,validate)
  # Error cleanup retains the borrowed capture and lets the original owner
  # drain/shutdown before restoring the queue observer; no manual arm mutation.
  final=next(n for n in fn.body if isinstance(n,ast.Try)).finalbody
  final_calls=[n for item in final for n in ast.walk(item) if isinstance(n,ast.Call)]
  final_close=next(n.lineno for n in final_calls if isinstance(n.func,ast.Attribute) and n.func.attr=='close')
  shutdown=next(n.lineno for n in final_calls if isinstance(n.func,ast.Attribute) and n.func.attr=='shutdown')
  self.assertLess(shutdown,final_close)
  self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and n.attr=='_prefix_single_file_wait_armed' for n in ast.walk(fn)))
 def test_actual_verifier_identity_and_original_128_frame_callsite(self):
  frozen=load('_independent_preparation_frozen_contract_test',HERE/'test_frozen_entry_constraints.py')
  source=PREP/'verify_p4_single_file.py';tree=ast.parse(source.read_bytes())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze_runtime')
  nodes=[n for n in fn.body if (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
   and isinstance(n.value.func,ast.Attribute) and n.value.func.attr in ('validate_capture_identity','validate_notification_evidence'))
   or (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
    and n.value.func.attr=='validate_capture')]
  self.assertEqual(len(nodes),3)
  code=compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes),type_ignores=[])),str(source),'exec')
  owner,capture,start,end,audit=fixture('off');notification=finish(capture,audit)
  cap,tokens=frozen.synthetic_capture(run_id='r',native_request_id='native-id')
  output=dict(request_id='r-p0-B',native_request_id='native-id',output_token_ids=tokens)
  row=dict(request_id='r-p0-B',capture=cap,notification_evidence=notification)
  env=dict(notification_adapter=lambda:A,output=output,row=row,config={'label':'r'},mode='off',C=frozen.CONTRACT)
  exec(code,env);self.assertEqual(len(env['frames']),128)
  for kind in ('capture_run','frontend_request','native_request','missing_frame'):
   scope=dict(env);scope['row']=deepcopy(row);scope['output']=deepcopy(output)
   if kind=='capture_run':scope['row']['capture']['run_id']='r-p0-B'
   elif kind=='frontend_request':scope['output']['request_id']='wrong'
   elif kind=='native_request':scope['output']['native_request_id']='wrong'
   else:scope['row']['capture']['frames'].pop()
   with self.subTest(kind=kind):
    with self.assertRaises(ValueError):exec(code,scope)
 def test_inconsistent_performance_claim_and_source_pin_rejected(self):
  owner,capture,start,end,audit=fixture('off');doc=finish(capture,audit)
  for field in ('performance_claim','gpu_completion_proved','native_cost_qualified','source'):
   bad=deepcopy(doc)
   if field=='source':bad['source_refs']['collector']['sha256']='0'*64
   else:bad[field]=True
   with self.subTest(field=field):
    with self.assertRaises(ValueError):A.validate_notification_evidence(bad,mode='off',run_id='r',request_id='r-p0-B')
 def test_original_fifo_consumes_one_message_only(self):
  owner,capture,start,end,audit=fixture();owner._incoming.put('first');owner._incoming.put('second')
  self.assertTrue(owner._prefix_wait_single_file_retry());self.assertEqual(owner.seen,['first'])
  self.assertEqual(owner._incoming.get_nowait(),'second')
  document=finish(capture,audit)
  with self.assertRaises(ValueError):A.validate_notification_evidence(document,mode='on',run_id='r',request_id='r-p0-B')
 def test_optional_before_and_after_failures_preserve_queue_work_and_native_error(self):
  for when in ('before','after'):
   with self.subTest(when=when):
    owner,capture,start,end,audit=fixture()
    def fail(*args):raise RuntimeError('optional-observation-failure')
    setattr(audit,'_'+when+'_get',fail)
    owner._incoming.put('native-result');self.assertEqual(owner._incoming.get(False),'native-result')
    with self.assertRaisesRegex(ValueError,'timeout'):owner._incoming.get(timeout=-1)
    self.assertTrue(audit.failures);finish(capture,audit)
 def test_off_shadow_do_not_wrap_queue_or_install(self):
  for mode in ('off','shadow'):
   with self.subTest(mode=mode):
    owner,capture,start,end,audit=fixture(mode)
    self.assertNotIn('get',vars(owner._incoming));self.assertFalse(audit.installed)
    self.assertIsNone(getattr(owner,'_prefix_single_file_wait_capture',None))
    document=finish(capture,audit)
    result=A.validate_notification_evidence(document,mode=mode,run_id='r',request_id='r-p0-B')
    self.assertFalse(result['notification_exercised'])
 def test_detach_restores_only_own_queue_wrapper(self):
  owner,capture,start,end,audit=fixture();replacement=lambda *a,**k:'foreign'
  owner._incoming.get=replacement;document=finish(capture,audit)
  self.assertIs(owner._incoming.get,replacement);self.assertFalse(document['queue_get_restored'])
  with self.assertRaises(ValueError):A.validate_notification_evidence(document,mode='on',run_id='r',request_id='r-p0-B')
 def test_audit_does_not_keep_reactor_queue_or_capture_alive(self):
  owner,capture,start,end,audit=fixture();finish(capture,audit)
  owner_ref,queue_ref,capture_ref=weakref.ref(owner),weakref.ref(owner._incoming),weakref.ref(capture)
  owner.bridge._single_file_state=None
  del owner,capture,start,end;gc.collect()
  self.assertIsNone(owner_ref());self.assertIsNone(queue_ref());self.assertIsNone(capture_ref())
 def test_all_real_gpu_entrypoints_remain_unconditionally_blocked(self):
  for filename,names in (
   ('control_p4_single_file.py',('load_receipt','freeze_common','prepare','launch')),
   ('run_p4_single_file_experiment.py',('load_configuration','verify_guard','execute_window')),
   ('verify_p4_single_file.py',('verify_runtime',))):
   module=load('_independent_blocked_'+filename.replace('.','_'),PREP/filename)
   for name in names:
    with self.subTest(file=filename,method=name):
     # Feed all formal args synthetically; blocking must precede their inspection.
     import inspect
     signature=inspect.signature(getattr(module,name));kwargs={}
     for p in signature.parameters.values():
      if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL,p.VAR_KEYWORD):kwargs[p.name]=None
     with self.assertRaisesRegex(RuntimeError,A.BLOCKED):getattr(module,name)(**kwargs)

if __name__=='__main__':unittest.main(verbosity=2)
