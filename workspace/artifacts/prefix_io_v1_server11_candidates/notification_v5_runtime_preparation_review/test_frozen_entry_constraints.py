"""Actual frozen identity/128-frame validators with explicitly synthetic CPU data."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
import weakref

HERE=Path(__file__).resolve().parent
CANDIDATE=Path(os.environ.get('SERVER11_C5_CANDIDATE',HERE.parent/'p4_single_file_candidate_v5_cpu')).resolve()
NATIVE=Path(os.environ.get('SERVER11_NATIVE_V6_ROOT',HERE.parent/'native_cost_v6')).resolve()
REACTOR=CANDIDATE/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
RECEIPT=CANDIDATE/'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'
PINNED={REACTOR:'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
 RECEIPT:'11312ab7a6f5c7823e77b22cf17c93bd255438af7f886568e693f76fc5b9825d',
 NATIVE/'native_conditional_cost.py':'e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291'}
for path,digest in PINNED.items():
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise AssertionError('frozen source mismatch: '+str(path))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module
CONTRACT=load('_independent_original_native_v6_contract',NATIVE/'native_conditional_cost.py')
R=load('_independent_original_c5_receipt',RECEIPT)
tree=ast.parse(REACTOR.read_text(encoding='utf-8-sig'))
owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IoReactor')
fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='install_p4_single_file_wait')
env={'weakref':weakref};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(REACTOR),'exec'),env)

class Capture:pass
def owner_and_capture(owner_id,capture_id,*,off=False,shadow=False):
 owner=NS(_submit_lock=threading.Lock(),_prefix_p4_bridge=None if off else NS(
  single_file_shadow=shadow,mode='interference',run_id=owner_id,policy=NS(single_file=object())))
 capture=Capture();capture.run_id=capture_id
 return owner,capture

def synthetic_capture(run_id='owner-run',native_request_id='original-native-request'):
 """All GPU-looking witness values below are synthetic, never qualification."""
 frames=[];witnesses=[];tokens=list(range(128))
 for offset,token in enumerate(tokens):
  tick=1_000_000_000+offset*1000;ordinal=300+offset;context=128 if offset==0 else 128+offset
  prepared=dict(native_step_ordinal=ordinal,batch=1,active_decode=0 if offset==0 else 1,
   prefill_tokens=1 if offset==0 else 0,context_length=context,
   step_kind='prefill' if offset==0 else 'decode',context_basis='pre_computed_tokens',
   rows=[dict(request_id=native_request_id,pre_context=context,prompt_tokens=129,scheduled_tokens=1)],
   input_seq_lens_from_cpu_inputs=[129 if offset==0 else context+1])
  frames.append(dict(native_step_ordinal=ordinal,start_ns=tick+10,end_ns=tick+20,
   intended_timing_scope='full_decode_step',gpu_elapsed_ns=None,existing_io=None,new_io=None,
   prepared=prepared,outputs=[[native_request_id,[token]]]))
  witnesses.append(dict(native_step_ordinal=ordinal,start_record_before_ns=tick,
   start_record_after_ns=tick+2,start_completed_query_ns=tick+5,end_record_before_ns=tick+21,
   end_record_after_ns=tick+22,end_completed_query_ns=tick+25,
   event_elapsed_source='torch.cuda.Event.elapsed_time',gpu_elapsed_ns=20))
 return dict(scope='server11_full_step_native_capture_v1',run_id=run_id,origin='native_gpu_recording',
  valid=True,frames=frames,event_witnesses=witnesses,failures=[],pending_event_pairs=0,
  open_event_pair=False,no_added_synchronization=True,cross_clock_absolute_mapping=False,
  selected_offsets=[16]),tokens

class FrozenEntryTests(unittest.TestCase):
 def test_old_launcher_namespace_is_rejected_by_actual_c5_install(self):
  owner,capture=owner_and_capture('job-on','job-on-p0-B')
  with self.assertRaisesRegex(ValueError,'explicit active bridge'):env[fn.name](owner,capture)
  self.assertFalse(hasattr(owner,'_prefix_single_file_wait_capture'))
 def test_explicit_owner_capture_identity_enables_install_without_request_relabel(self):
  owner,capture=owner_and_capture('job-on','job-on');capture.request_id='job-on-p0-B'
  env[fn.name](owner,capture)
  self.assertIs(owner._prefix_single_file_wait_capture(),capture)
  self.assertEqual(capture.request_id,'job-on-p0-B')
 def test_off_and_shadow_cannot_install_waiting(self):
  for mode in ('off','shadow'):
   with self.subTest(mode=mode):
    owner,capture=owner_and_capture('job','job',**{mode:True})
    with self.assertRaises(ValueError):env[fn.name](owner,capture)
 def test_second_capture_on_same_owner_is_rejected(self):
  owner,capture=owner_and_capture('job','job');env[fn.name](owner,capture)
  second=Capture();second.run_id='job'
  with self.assertRaisesRegex(ValueError,'already installed'):env[fn.name](owner,second)
 def test_original_128_frame_validator_supports_distinct_run_and_request_ids(self):
  capture,tokens=synthetic_capture()
  rows=CONTRACT.validate_capture(capture,run_id='owner-run',request_id='original-native-request',
   output_ids=tokens,prompt_tokens=129,measured_offset=16,warmup_offsets=[1])
  self.assertEqual(len(rows),128);self.assertEqual(rows[16]['native_step_ordinal'],316)
 def test_identity_and_complete_frame_failures_still_rejected(self):
  for kind in ('run','prepared_request','output_request','missing_frame'):
   with self.subTest(kind=kind):
    capture,tokens=synthetic_capture()
    if kind=='run':capture['run_id']='different-run'
    elif kind=='prepared_request':capture['frames'][99]['prepared']['rows'][0]['request_id']='wrong'
    elif kind=='output_request':capture['frames'][99]['outputs'][0][0]='wrong'
    else:capture['frames'].pop()
    with self.assertRaises(ValueError):CONTRACT.validate_capture(capture,run_id='owner-run',
     request_id='original-native-request',output_ids=tokens,prompt_tokens=129,measured_offset=16,warmup_offsets=[1])
 def test_c5_copy_of_receipt_is_explicitly_still_c4_v6_only(self):
  self.assertIn('candidate-v4-',R._COMMON_OVERLAY);self.assertIn('native-cost-v6-',R._D6)
  self.assertEqual(R._NATIVE,R._COMMON_OVERLAY+'/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py')
  node=next(n for n in ast.parse(RECEIPT.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='load_verified_single_file')
  values=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant)]
  self.assertIn('server11-native-cost-six-window-06',values)
 def test_public_receipt_constructor_cannot_promote_cpu_fixture(self):
  with self.assertRaisesRegex(ValueError,'source-bound'):R.ExactSingleFileReceipt()

if __name__=='__main__':unittest.main(verbosity=2)
