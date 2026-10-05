"""CPU contract/AST rejection tests; no native receipt or positive GPU fixture."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_repeatability_CPU_tests_module',HERE/'verify_repeatability.py')
V=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=V
spec.loader.exec_module(V)
PROTOCOL=json.loads((HERE/'PROTOCOL.json').read_bytes())
PROJECT=HERE.parents[2]
# The actual server project uses the immutable production path. The local
# source-only delivery uses its explicit candidate directory. Prefer the real
# production path whenever present; never fall back after a source-pin failure.
PRODUCTION_OLD_SOURCE=PROJECT/V.V_REF['path']
LOCAL_DELIVERY_OLD_SOURCE=PROJECT/'artifacts/prefix_io_v1_server12_candidates/normal_native_cpu/verify_p4_single_file.py'
OLD_SOURCE=PRODUCTION_OLD_SOURCE if PRODUCTION_OLD_SOURCE.exists() else LOCAL_DELIVERY_OLD_SOURCE
PRIOR_RAW_REF=PROTOCOL['immutable_prior_off_counterexample']['raw_result_ref']
PRODUCTION_PRIOR_RAW=PROJECT/PRIOR_RAW_REF['path']
LOCAL_DELIVERY_PRIOR_RAW=PROJECT/'artifacts/prefix_io_v1_server12_candidates/normal_native_cpu/OFF01_REAL_RAW_RESULT.json'
PRIOR_RAW_SOURCE=PRODUCTION_PRIOR_RAW if PRODUCTION_PRIOR_RAW.exists() else LOCAL_DELIVERY_PRIOR_RAW


def actual_prior_raw():
    data=PRIOR_RAW_SOURCE.read_bytes()
    if len(data)!=PRIOR_RAW_REF['bytes'] or sha256(data).hexdigest()!=PRIOR_RAW_REF['sha256']:
        raise ValueError('actual immutable prior raw source pin differs')
    return json.loads(data)


class ProtocolCPURejections(unittest.TestCase):
    def test_actual_protocol_byte_pin(self):
        raw=(HERE/'PROTOCOL.json').read_bytes()
        self.assertEqual(len(raw),V.PROTOCOL_REF['bytes'])
        self.assertEqual(sha256(raw).hexdigest(),V.PROTOCOL_REF['sha256'])
        self.assertIs(V.validate_protocol_document(PROTOCOL),PROTOCOL)

    def test_reject_protocol_boolean_duration(self):
        value=deepcopy(PROTOCOL);value['cost_upper_ns']=True
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_changed_budget(self):
        value=deepcopy(PROTOCOL);value['step_budget_ns']+=1
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_changed_upper(self):
        value=deepcopy(PROTOCOL);value['cost_upper_ns']+=1
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_extra_attempt(self):
        value=deepcopy(PROTOCOL);value['repetitions']=4
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_changed_seed(self):
        value=deepcopy(PROTOCOL);value['seed']=1830
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_changed_workload(self):
        value=deepcopy(PROTOCOL);value['prompt_first_token']=18100
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_promotion(self):
        value=deepcopy(PROTOCOL);value['decision_rule']['permits_next_mode']='shadow'
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_protocol_boolean_index(self):
        value=deepcopy(PROTOCOL);value['jobs'][0]['diagnostic_index']=False
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_reject_slot_label_change(self):
        value=deepcopy(PROTOCOL);value['jobs'][0]['label']='server12-c5-native-normal-off01'
        with self.assertRaises(ValueError):V.validate_protocol_document(value)

    def test_original_source_actual_pin(self):
        raw=OLD_SOURCE.read_bytes()
        self.assertEqual(len(raw),V.V_REF['bytes'])
        self.assertEqual(sha256(raw).hexdigest(),V.V_REF['sha256'])

    def test_actual_prior_event_witness_not_raw_frame_placeholder(self):
        raw=actual_prior_raw()
        self.assertIsNone(raw['windows'][0]['capture']['frames'][16]['gpu_elapsed_ns'])
        self.assertEqual(V.prior_selected_gpu_ns(raw),16893473)

    def test_prior_event_witness_ordinal_change_rejected(self):
        raw=actual_prior_raw()
        raw['windows'][0]['capture']['event_witnesses'][16]['native_step_ordinal']+=1
        with self.assertRaises(ValueError):V.prior_selected_gpu_ns(raw)

    def test_prior_event_witness_boolean_elapsed_rejected(self):
        raw=actual_prior_raw()
        raw['windows'][0]['capture']['event_witnesses'][16]['gpu_elapsed_ns']=True
        with self.assertRaises(ValueError):V.prior_selected_gpu_ns(raw)

    def test_analysis_AST_unchanged(self):
        original=ast.parse(OLD_SOURCE.read_bytes())
        before=ast.dump(original,include_attributes=False)
        adapted,proof=V.adapted_validator_tree(original)
        self.assertEqual(ast.dump(original,include_attributes=False),before)
        self.assertEqual(len(proof['metadata_AST_changes']),7)
        self.assertEqual(len(proof['unchanged_scientific_functions']),len(V.ANALYSIS_FUNCTIONS))
        self.assertTrue(all(row['identical_AST'] is True for row in proof['unchanged_scientific_functions']))
        compile(adapted,'CPU_metadata_AST_only','exec')

    def test_missing_actual_original_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):V.validator(Path(temp))

    def test_changed_actual_source_rejected_before_compile(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/V.V_REF['path'];path.parent.mkdir(parents=True)
            path.write_bytes(b'# CPU source drift rejection only\n')
            with self.assertRaises(ValueError):V.validator(root)

    def test_missing_protocol_has_no_native_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):V.prepare_protocol(Path(temp))

    def test_boolean_index_rejected_before_native_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):V.verify_one(Path(temp),False)

    def test_fourth_slot_rejected_before_native_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):V.verify_one(Path(temp),3)

    def test_arithmetic_bool_rejected(self):
        with self.assertRaises(ValueError):V.classify_durations([1,True,3],3)

    def test_arithmetic_only_fixed_comparison(self):
        self.assertEqual(V.classify_durations([10,10,10],10),'ALL_THREE_RECORDED_STEPS_COVERED_ONLY')
        self.assertEqual(V.classify_durations([9,11,10],10),'OBSERVED_THRESHOLD_CROSSING_VARIABILITY')
        self.assertEqual(V.classify_durations([11,12,13],10),'ALL_THREE_RECORDED_STEPS_EXCEEDED')
        # These are small CPU arithmetic values, not fabricated GPU records.

    def test_missing_slots_never_qualify(self):
        records=[dict(diagnostic_index=i,native_execution_verified=False,status='NO_ACTUAL_COMPLETION_GUARD')
            for i in range(3)]
        result=V.summarize_records(records,PROTOCOL)
        self.assertEqual(result['classification'],'INCOMPLETE_OR_INVALID_NO_REPEATABILITY_CONCLUSION')
        self.assertIsNone(result['descriptive_statistics'])
        self.assertIsNone(result['permits_next_mode'])
        self.assertIs(result['qualification_passed'],False)
        self.assertIs(result['prior_and_new_revisions_pooled'],False)
        self.assertEqual(result['original_prior_off_counterexample_retained']['selected_gpu_elapsed_ns'],16893473)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolCPURejections))
    raise SystemExit(not result.wasSuccessful())
