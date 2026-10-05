"""CPU evidence checks; no native model or CUDA qualification."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import torch

from probekv.p0_decode_evidence_v2 import DecodeInputRecorderV2, validate_decode_trace
from probekv.p0_evidence_v2 import P0EvidenceWriter, compare_p0_logit_actions
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest
from tests import test_native_v2_prefix_publication as native_fixtures


class DecodeEvidenceTests(unittest.TestCase):
    def request(self, teacher=(9, 8)):
        return dict(request_id='request', token_ids=[11, 12, 13], max_new_tokens=3,
                    teacher_token_ids=list(teacher), capture_logits=True)

    def trace(self, request=None, predictions=(1, 0, 1)):
        request = request or self.request()
        recorder = DecodeInputRecorderV2(request, cached_prefix_tokens=0)
        for i, token in enumerate(predictions):
            recorder.append(token, fed_token=request['teacher_token_ids'][i-1] if i else None)
        return recorder.finish(logit_rows=3)

    def test_T20_actual_teacher_inputs_are_distinct_from_predictions(self):
        trace = self.trace()
        self.assertEqual(trace['fed_token_ids'], [9, 8])
        self.assertEqual(trace['predicted_token_ids'], [1, 0, 1])
        self.assertEqual(trace['logit_absolute_positions'], [2, 3, 4])
        validate_decode_trace(trace, request=self.request(), answer={'token_ids':[1, 0, 1]}, logit_rows=3)

    def test_incomplete_teacher_or_wrong_native_feed_cannot_complete(self):
        r = DecodeInputRecorderV2(self.request(), cached_prefix_tokens=0)
        r.append(1)
        with self.assertRaises(ValueError): r.append(0, fed_token=1)
        with self.assertRaises(ValueError): r.finish(logit_rows=1)

    def test_greedy_early_eos_is_valid_but_teacher_early_stop_is_not(self):
        request = self.request(); del request['teacher_token_ids']
        r = DecodeInputRecorderV2(request, cached_prefix_tokens=0); r.append(1)
        trace = r.finish(logit_rows=1)
        self.assertEqual(trace['mode'], 'greedy')
        with self.assertRaises(ValueError): r.append(2, fed_token=1)

    def test_invalid_bound_prefix_positions_and_hash_fail_closed(self):
        for change in ({'max_new_tokens':True}, {'teacher_token_ids':[1]}, {'token_ids':[False]}):
            with self.assertRaises(ValueError): DecodeInputRecorderV2({**self.request(), **change}, cached_prefix_tokens=0)
        with self.assertRaises(ValueError): DecodeInputRecorderV2(self.request(), cached_prefix_tokens=4)
        for key,value in (('logit_absolute_positions',[1,2,3]), ('trace_sha256','0'*64)):
            trace = self.trace(); trace[key] = value
            with self.assertRaises(ValueError):
                validate_decode_trace(trace, request=self.request(), answer={'token_ids':[1,0,1]}, logit_rows=3)

    def test_native_finish_records_actual_completed_decode_steps_cpu_fixture(self):
        f = native_fixtures.NativeV2PrefixPublicationTests()
        c, _ = f.fixture()
        c.request = self.request()
        c.finish_timing_landmarks = {}
        c.native.finish_prefill = Mock()
        c.native.append_for_decode = Mock(side_effect=lambda t:t)
        c.native.finish_decode_step = Mock()
        c.sampling = NS(selected_token_indices=torch.tensor([2]))
        c.sampling_signature = {'max_new_tokens':3}
        class Outer:
            def __call__(self, **kwargs): return torch.zeros(1,4)
            def compute_logits(self, *args): return torch.tensor([[0.,1.]])
        c.adapter.outer = Outer()
        c.adapter.kv = None
        c.adapter.check_deadline = Mock()
        c.adapter.prepare = lambda t:(torch.tensor([t]),torch.tensor([3]),None,c.sampling)
        c.adapter.llm = NS(get_tokenizer=lambda:NS(eos_token_id=1))
        c.adapter.warm_history = []
        c.engine.session.resolve_completed_layer_timings = Mock()
        c.engine.session.layer_audit = []
        c.engine.overlap_trace = lambda:[]
        result = c.finish_from_prefill_hidden(torch.zeros(3,4), Mock())
        self.assertEqual([x.args[0] for x in c.native.append_for_decode.call_args_list], [9,8])
        self.assertEqual(c.native.finish_decode_step.call_count, 2)
        self.assertEqual(result['decode_input_trace_v2']['fed_token_ids'], [9,8])
        self.assertEqual(len(c.logit_trace), 3)
        self.assertIsNone(result['quality_passed'])


class AlignedLogitFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.binding = {k:'cpu-fixture-only' for k in ('code_commit','runtime_digest','patch_sha256',
                       'model_signature','tokenizer_hash','gpu_uuid','instance_id')}

    def arm(self, name, *, teacher=(9,8), predictions=(1,0,1), binding=None,
            completed=True, cleanup=True, origin='cpu_fixture', logits=None):
        f = DecodeEvidenceTests(); request = f.request(teacher)
        trace = f.trace(request, predictions)
        manifest = {'binding':binding or self.binding, 'jobs':[
            dict(action_id='a', request=request, request_sha256=digest_json(request))]}
        w = P0EvidenceWriter(self.root/name, binding=manifest['binding'], manifest=manifest)
        w.write_action('a', audit=dict(status='COMPLETED' if completed else 'FAILED', request_id='request',
            cleanup={'passed':cleanup}, answer={'token_ids':list(predictions),
                'generation_mode':'teacher_forced_logit_diagnostic', 'decode_input_trace_v2':trace}),
            logits=logits if logits is not None else [torch.ones(2) for _ in range(3)], origin=origin)
        return self.root/name/'a'

    def compare(self, a, b, **kwargs):
        return compare_p0_logit_actions(a,b,
            record_hashes=(file_digest(a/'record.json'),file_digest(b/'record.json')),
            manifest_hashes=(file_digest(a.parent/'manifest.json'),file_digest(b.parent/'manifest.json')),
            relative_l2_limit=kwargs.get('limit',0.), minimum_positions=3)

    def mutate(self, path, fn):
        data=json.loads(path.read_text());fn(data);path.write_text(json.dumps(data))

    def test_T20_aligned_teacher_raw_pass_does_not_qualify_exact_or_mixed_recipe(self):
        result = self.compare(self.arm('ref'), self.arm('candidate', predictions=(0,0,0)))
        self.assertTrue(result['numeric_passed'])
        self.assertTrue(result['conditioning_alignment_verified'])
        self.assertFalse(result['predicted_token_ids_identical'])
        self.assertFalse(result['native_runtime_qualified'])
        self.assertFalse(result['recipe_alignment_verified'])
        self.assertFalse(result['P1_execution_allowed'])

    def test_same_numeric_arrays_with_different_teacher_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'conditioning'):
            self.compare(self.arm('ref'), self.arm('other', teacher=(9,7)))

    def test_different_model_patch_or_device_rejected(self):
        a = self.arm('ref')
        for field in ('model_signature','runtime_digest','gpu_uuid'):
            b = self.arm(field, binding={**self.binding, field:'different'})
            with self.assertRaises(ValueError):self.compare(a,b)

    def test_failed_action_or_cleanup_is_not_numerical_success(self):
        a = self.arm('ref')
        for name,kw in (('failed',dict(completed=False)), ('cleanup',dict(cleanup=False))):
            with self.assertRaises(ValueError): self.compare(a,self.arm(name,**kw))

    def test_missing_trace_or_tampered_answer_rejected(self):
        a,b = self.arm('ref'), self.arm('candidate')
        self.mutate(b/'request.json', lambda v:v['answer'].pop('decode_input_trace_v2'))
        self.mutate(b/'record.json', lambda v:v['request'].update(sha256=file_digest(b/'request.json')))
        with self.assertRaisesRegex(ValueError,'trace missing'):self.compare(a,b)

    def test_changed_raw_manifest_hash_and_redirected_paths_rejected(self):
        a,b = self.arm('ref'), self.arm('candidate')
        self.mutate(b/'record.json', lambda v:v['logits'].update(file='../other.npy'))
        with self.assertRaises(ValueError):self.compare(a,b)
        with self.assertRaises(ValueError):
            compare_p0_logit_actions(a,b,record_hashes=('bad','bad'),manifest_hashes=('bad','bad'),
                                    relative_l2_limit=0.,minimum_positions=3)

    def test_T21_even_cuda_label_cannot_prove_explicit_mixed_reference(self):
        # This deliberately demonstrates that a label alone cannot open a gate.
        result=self.compare(self.arm('ref',origin='real_cuda_execution'),self.arm('candidate',origin='real_cuda_execution'))
        self.assertFalse(result['native_runtime_qualified'])
        self.assertIn('exact_or_explicit_mixed_reference',result['pending_recipe_checks'])

    def test_logit_file_count_must_match_actual_completed_trace(self):
        a,b=self.arm('ref'),self.arm('candidate',logits=[torch.ones(2)]*2)
        with self.assertRaises(ValueError): self.compare(a,b)

    def test_raw_numeric_failure_preserved_after_alignment(self):
        result=self.compare(self.arm('ref'),self.arm('candidate',logits=[torch.zeros(2)]*3))
        self.assertFalse(result['numeric_passed'])
        self.assertTrue(result['conditioning_alignment_verified'])


if __name__ == '__main__': unittest.main()
