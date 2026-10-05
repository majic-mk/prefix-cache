"""Independent original-math attacks: synthetic data, never native qualification."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
NATIVE=Path(os.environ['SERVER11_NATIVE_V6_ROOT']).resolve(strict=True)
PROJECT=Path(os.environ['SERVER11_AUTHOR_SOURCE_ROOT']).resolve(strict=True)
CANDIDATE=Path(os.environ['SERVER11_C5_CANDIDATE_ROOT']).resolve(strict=True)
ORIGINAL=PROJECT/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
os.environ['NATIVE_ORIGINAL_ESTIMATOR']=str(ORIGINAL)

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module

F=load('_independent_original_cost_fixture',NATIVE/'test_native_conditional_cost.py')
R=load('_independent_frozen_receipt_budget',CANDIDATE/'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py')

def analyze(plan,windows,journal,drain):
 return F.M.analyze_paired(plan,windows,journal=journal,drain=drain,original_source_path=ORIGINAL)

class OriginalBoundaries(unittest.TestCase):
 def test_exact_original_calibration_formula_and_a_only_budget(self):
  plan,windows,journal,drain=F.fixture();result=analyze(plan,windows,journal,drain)
  self.assertEqual(result['calibration_only_cost']['baseline_ns'],100)
  self.assertEqual(result['calibration_only_cost']['incremental_or_joint_ns'],15)
  self.assertEqual(result['calibration_only_cost']['uncertainty_ns'],5)
  self.assertEqual(result['calibration_predicted_upper_ns'],120)
  self.assertEqual(R._calibration_a_budget(plan,result),100)
  for key in ('native_execution_verified','conditional_cost_cell_qualified','production_qualified','strategy_effect_verified'):
   self.assertIs(result[key],False)
 def test_holdout_outlier_never_changes_cost_or_a_budget(self):
  plan,windows,journal,drain=F.fixture()
  windows[-2]['capture']['event_witnesses'][16]['gpu_elapsed_ns']=8000
  windows[-1]['capture']['event_witnesses'][16]['gpu_elapsed_ns']=131
  result=analyze(plan,windows,journal,drain)
  self.assertEqual(result['calibration_predicted_upper_ns'],120)
  self.assertEqual(R._calibration_a_budget(plan,result),100)
  self.assertFalse(result['heldout_covered']);self.assertEqual(result['heldout_errors'][0]['underprediction_ns'],11)
  self.assertNotEqual(result['all_pairs_original_estimator_audit_only']['uncertainty_ns'],result['calibration_only_cost']['uncertainty_ns'])
 def test_warmup_outliers_cannot_widen_threshold(self):
  plan,windows,journal,drain=F.fixture()
  for window in windows:window['capture']['event_witnesses'][1]['gpu_elapsed_ns']=8000
  result=analyze(plan,windows,journal,drain)
  self.assertEqual(result['calibration_predicted_upper_ns'],120);self.assertEqual(R._calibration_a_budget(plan,result),100)
 def test_integer_rounding_negative_delta_and_a_residual(self):
  plan,windows,journal,drain=F.fixture()
  # AB, BA, AB: calibration A={101,102}; B={100,102}.
  for window,value in zip(windows,(101,100,102,102,101,102)):
   window['capture']['event_witnesses'][16]['gpu_elapsed_ns']=value
  result=analyze(plan,windows,journal,drain)
  self.assertEqual(result['calibration_only_cost']['baseline_ns'],102)
  self.assertEqual(result['calibration_only_cost']['incremental_or_joint_ns'],0)
  self.assertEqual(result['calibration_predicted_upper_ns'],102);self.assertEqual(R._calibration_a_budget(plan,result),102)
 def test_holdout_exact_boundary_and_one_nanosecond_underprediction(self):
  for duration,covered in ((120,True),(121,False)):
   plan,windows,journal,drain=F.fixture();windows[-1]['capture']['event_witnesses'][16]['gpu_elapsed_ns']=duration
   result=analyze(plan,windows,journal,drain)
   self.assertIs(result['heldout_covered'],covered);self.assertEqual(result['calibration_predicted_upper_ns'],120)
 def test_leaked_holdout_and_reordered_windows_rejected(self):
  for kind in ('prefix','trace','workload','order'):
   plan,windows,journal,drain=F.fixture()
   if kind=='order':windows[0],windows[1]=windows[1],windows[0]
   else:
    field={'prefix':'prefix_family_sha256','trace':'trace_sha256','workload':'workload_sha256'}[kind]
    plan['entries'][2][field]=plan['entries'][0][field]
   with self.subTest(kind=kind):
    with self.assertRaises(ValueError):analyze(plan,windows,journal,drain)
 def test_origin_claim_or_constructor_cannot_issue_native_receipt(self):
  with self.assertRaises(ValueError):F.M.validate_guard(dict(origin='native_gpu_recording',exit=0),gpu_uuid='GPU-X',job_id='new',wrapper_path='wrapper.py')
  with self.assertRaises(ValueError):R.ExactSingleFileReceipt()

if __name__=='__main__':unittest.main(verbosity=2)
