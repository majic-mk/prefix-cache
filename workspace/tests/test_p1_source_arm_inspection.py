"""CPU fixtures for an explanation audit, not GPU/quality evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).parents[1]/'artifacts/decoupled-v2/server31780-20260922/layer3_matrix/inspect_arms.py'
spec=importlib.util.spec_from_file_location('p1_arm_inspection', PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SourceArmInspectionTests(unittest.TestCase):
    def fixture(self):
        result={}
        for i,arm in enumerate(('dense','historical_1','historical_2','historical_3','historical_4','S0')):
            sid='source-'+str(i);h='hash-'+str(i);support=list(range(10+i,87+i))
            result[arm]=dict(operation=dict(action_key=['P1-E','g','q',arm],stage='P1-E',
                request=dict(token_ids=list(range(600)),segments=[dict(segment_id='C',positions=list(range(10,522)))]),
                first_reuse_layer=3,repair_ratio=.15),audit=dict(cleanup={'passed':True},
                source_id=sid,source_generation=0,committed_boundaries={'C':3},
                answer=dict(whole_request_origin='selective_reuse',token_ids=[1,2],qa_evidence={'answer_f1':1.}),
                consumption=dict(source_ids={'C':sid}),
                integrity=dict(expected_artifact_digest=h,source_digest_before=h,source_digest_after=h,
                    destination_digest=h,per_request_full_digest_verified=True),
                observation=dict(compared_ids=[sid],winner_source_id=sid,completed_depth=2,
                    scores=[[sid,.01*i]],current_k_digest='current',source_signatures=[[sid,h,'state']]),
                repair_support={'C':{str(j):support for j in range(3,33)}}))
        result['dense']['audit'].update(source_id=None,committed_boundaries={})
        result['dense']['audit']['answer']['whole_request_origin']='exact_dense_full_prefill'
        return result

    def test_distinct_tensors_and_masks_can_have_identical_answers(self):
        r=module.inspect_request(self.fixture())
        self.assertEqual(r['distinct_historical_KV_digests'],4)
        self.assertEqual(r['distinct_historical_repair_masks'],4)
        self.assertTrue(r['all_six_generated_token_ids_identical'])
        self.assertFalse(r['actual_quality_oracle_identifiable'])

    def test_missing_arm_is_not_success_or_dense(self):
        f=self.fixture();del f['S0']
        with self.assertRaisesRegex(ValueError,'complete six-arm'):module.inspect_request(f)

    def test_corruption_identity_and_fallback_rejected(self):
        for field in ('source','digest','fallback','depth','current'):
            f=self.fixture();a=f['historical_1']['audit']
            if field=='source':a['consumption']['source_ids']['C']='different'
            if field=='digest':a['integrity']['destination_digest']='corrupt'
            if field=='fallback':a['answer']['whole_request_origin']='exact_dense_full_prefill'
            if field=='depth':a['observation']['completed_depth']=8
            if field=='current':a['observation']['current_k_digest']='different'
            with self.assertRaises(ValueError,msg=field):module.inspect_request(f)

    def test_digest_alias_is_reported_not_silently_distinct(self):
        f=self.fixture();a=f['historical_2']['audit'];h='hash-1'
        a['integrity'].update(expected_artifact_digest=h,source_digest_before=h,source_digest_after=h,destination_digest=h)
        a['observation']['source_signatures'][0][1]=h
        self.assertEqual(module.inspect_request(f)['distinct_historical_KV_digests'],3)

    def test_mask_outside_target_or_changed_between_layers_rejected(self):
        for bad in ([9]+list(range(12,88)),list(range(12,89))):
            f=self.fixture();f['historical_1']['audit']['repair_support']['C']['3']=bad
            with self.assertRaises(ValueError):module.inspect_request(f)

    def test_all_zero_f1_is_not_unique_selector_accuracy(self):
        f=self.fixture()
        for row in f.values():row['audit']['answer']['qa_evidence']['answer_f1']=0.
        r=module.inspect_request(f)
        self.assertTrue(r['historical_F1_tied'])
        self.assertFalse(r['actual_quality_oracle_identifiable'])


if __name__=='__main__':unittest.main()
