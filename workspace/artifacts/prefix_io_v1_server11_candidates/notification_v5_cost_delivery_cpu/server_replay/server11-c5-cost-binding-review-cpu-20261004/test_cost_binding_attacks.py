"""Independent attacks on CPU preparation, with explicit host source roots."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
FACTORY=Path(os.environ['SERVER11_C5_COST_FACTORY_ROOT']).resolve(strict=True)
# Fixture construction copies source bytes, uses synthetic model/Event files,
# and is confined to this review directory. It is never native acquisition.
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module
F=load('_independent_cost_factory_fixture',FACTORY/'test_c5_cost_binding_preparation.py')
M=F.M

class BindingAttacks(unittest.TestCase):
 def setUp(self):
  cases=(HERE/'TEMP_CASES').resolve();cases.mkdir(exist_ok=True)
  with patch.dict(os.environ,{'C5_COST_TEST_TEMP_ROOT':str(cases)}):
   self.h=F.C5SourceBindingContracts(methodName='test_actual_c5_and_preparation_sources_bind_but_issue_no_receipt')
   self.h.setUp()
  self.assertTrue(self.h.root.is_relative_to(cases));self.addCleanup(self.h.doCleanups)
 def test_preparation_retains_required_common_reactor_and_overlay_collector(self):
  h=self.h;reactor=h.plan['reactor_source_ref'];collector=h.plan['collector_source_ref']
  self.assertIn(reactor,h.binding['runtime_common_refs'])
  self.assertIn(collector,h.binding['runtime_overlay_refs'])
  self.assertNotIn(reactor,h.binding['runtime_overlay_refs']);self.assertNotIn(collector,h.binding['runtime_common_refs'])
  result=h.prepare();self.assertFalse(result['native_execution_verified']);self.assertFalse(result['on_observation_cost_measured'])
 def test_reactor_only_overlay_cannot_prepare_compatible_runtime_refs(self):
  h=self.h;reactor=h.plan['reactor_source_ref'];h.binding['runtime_common_refs'].remove(reactor)
  h.binding['runtime_overlay_refs'].append(reactor)
  with self.assertRaises(ValueError):h.prepare()
 def test_collector_only_common_or_duplicate_is_rejected(self):
  h=self.h;collector=h.plan['collector_source_ref'];h.binding['runtime_common_refs'].append(collector)
  with self.assertRaises(ValueError):h.prepare()
  h.binding['runtime_overlay_refs'].remove(collector)
  with self.assertRaises(ValueError):h.prepare()
 def test_bool_and_float_do_not_masquerade_as_original_owner_parameters(self):
  for key,bad in (('bridge_is_none',1),('max_accepted_parents',8.0),('max_accepted_parents',True)):
   with self.subTest(key=key,bad=bad):
    saved=deepcopy(self.h.plan['common_owner_parameters']);self.h.plan['common_owner_parameters'][key]=bad
    with self.assertRaises(ValueError):self.h.prepare()
    self.h.plan['common_owner_parameters']=saved
 def test_bool_warmup_offset_and_extra_formula_keys_rejected(self):
  h=self.h;h.plan['formula_contract']['warmup_offsets']=[True]
  with self.assertRaises(ValueError):h.prepare()
  h.plan['formula_contract']['warmup_offsets']=[1];h.plan['formula_contract']['new_margin_ns']=100000000
  with self.assertRaises(ValueError):h.prepare()
 def test_expected_ref_is_external_input_and_cannot_be_replaced_by_binding_claim(self):
  h=self.h;h.prepare();reference=M.file_ref(h.root,'binding.json');wrong=dict(reference,sha256='0'*64)
  with self.assertRaises(ValueError):h.prepare(expected_binding_ref=wrong)
  with self.assertRaises(ValueError):h.prepare(expected_plan_ref=wrong)
  with self.assertRaises(ValueError):h.prepare(expected_source_lock_ref=wrong)
 def test_missing_observer_from_both_new_groups_still_rejected_by_preparation_lock(self):
  h=self.h;path='preparation/notification_runtime_adapter.py'
  h.lock['files']=[r for r in h.lock['files'] if r['path']!=path];h.plan['source_lock_files']=len(h.lock['files'])
  h.binding['runtime_overlay_refs']=[r for r in h.binding['runtime_overlay_refs'] if r['path']!=path]
  with self.assertRaises(ValueError):h.prepare()
 def test_new_source_file_requires_closure_update(self):
  h=self.h;(h.root/'candidate/unrecorded_runtime.py').write_text('# independent synthetic drift\n',encoding='utf-8')
  with self.assertRaises(ValueError):h.prepare()
 def test_repeated_scope_ref_and_common_overlay_alias_rejected(self):
  h=self.h;h.lock['files'].append(deepcopy(h.lock['files'][0]));h.plan['source_lock_files']=len(h.lock['files'])
  with self.assertRaises(ValueError):h.prepare()
 def test_data_document_edit_cannot_change_preparation_or_issue_cost(self):
  h=self.h;h.prepare()
  bref=M.file_ref(h.root,'binding.json');pref=M.file_ref(h.root,'plan.json');lref=M.file_ref(h.root,'closure.json')
  item=M.prepare_cost_binding(h.root,binding_ref=bref,expected_binding_ref=bref,expected_plan_ref=pref,
   expected_source_lock_ref=lref,expected_factory_ref=h.binding['factory_source_ref'])
  doc=item.document();doc['native_execution_verified']=True;doc['effective_cost_upper_ns']=1
  actual=item.document();self.assertIs(actual['native_execution_verified'],False);self.assertIsNone(actual['effective_cost_upper_ns'])
  self.assertIsNone(actual['effective_step_budget_ns']);self.assertIsNone(actual['valid_native_receipt'])
  self.assertNotIn('signature',actual)
  with self.assertRaises(RuntimeError):M.load_verified_single_file(h.root,bref,item)
  with self.assertRaises(RuntimeError):M.launch_gpu(item)
 def test_native_or_effective_cost_claim_in_binding_rejected(self):
  h=self.h;h.binding['effective_cost_upper_ns']=1
  with self.assertRaises(ValueError):h.prepare()
  h.binding.pop('effective_cost_upper_ns');h.binding['origin']='native_gpu_recording'
  with self.assertRaises(ValueError):h.prepare()
 def test_original_verifier_entry_is_never_called_by_synthetic_audit(self):
  # Spy solely on the optional original analyzer object: its native qualifier
  # must remain unused. Loading the exact source itself remains mandatory.
  h=self.h;plan,windows,journal,drain=F.F.fixture()
  envelope=dict(origin='synthetic_cpu_contract',plan=plan,windows=windows,journal=journal,drain=drain)
  # The API can reject a generic eight-byte math fixture as non-C5 geometry;
  # normalize only actual physical amounts to the frozen one-file quantum.
  ratio=917504//plan['transfer_quantum_bytes'];plan['transfer_quantum_bytes']=917504
  for window in windows:
   if window['independent_payload'] is not None:window['independent_payload']['physical_bytes']=917504
  for event in journal['events']:
   event['physical_bytes']*=ratio
   if event['result'] is not None:event['result']*=ratio
  for frame in journal['frames']:
   for name in ('accepted','completed','inflight'):
    for amount in frame[name]:amount['nbytes']*=ratio
  for name in ('accepted','completed'):
   for amount in drain[name]:amount['nbytes']*=ratio
  drain['transferred_bytes']=[n*ratio for n in drain['transferred_bytes']]
  original_loader=M._load_pinned_analyzer;calls=[]
  def observed_loader(root,ref):
   analyzer=original_loader(root,ref)
   def prohibited(*args,**kwargs):calls.append(True);raise AssertionError('native issuer called')
   analyzer.verify_native_cell=prohibited;return analyzer
  with patch.object(M,'_load_pinned_analyzer',observed_loader):
   result=M.audit_synthetic_formula(h.root,envelope=envelope,analyzer_ref=h.plan['verifier_source_ref'],original_estimator_ref=h.plan['original_estimator_ref'])
  self.assertEqual(calls,[]);self.assertFalse(result['native_execution_verified']);self.assertIsNone(result['effective_cost_upper_ns'])
  self.assertFalse(result['on_observation_cost_measured'])
 def test_cli_is_blocked_before_argument_or_nonexistent_path_handling(self):
  proc=subprocess.run([sys.executable,'-B','-I','-S',str(FACTORY/'c5_cost_binding_preparation.py'),
   '--launch','--root','/nonexistent-do-not-create'],capture_output=True,text=True)
  self.assertEqual(proc.returncode,2,proc.stderr);doc=json.loads(proc.stdout)
  self.assertEqual(doc['status'],M.BLOCKED);self.assertFalse(doc['gpu_started']);self.assertIsNone(doc['valid_native_receipt'])
  self.assertEqual(proc.stderr,'')

if __name__=='__main__':unittest.main(verbosity=2)
