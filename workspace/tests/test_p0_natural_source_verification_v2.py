"""Real CPU file/ledger fixtures do not count as real-model execution."""
from copy import deepcopy
import unittest

import test_p0_launch_recipe_v2 as fixtures
from probekv.p0_natural_input_v2 import verify_natural_p0_birth
from probekv.v8_schema10_execution import digest_json


class NaturalSourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.LaunchRecipeTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.source = f.source((3,4), 'ordinary-birth')
        request = dict(request_id='ordinary-birth', token_ids=list(f.tokens),max_new_tokens=1,
                       segments=[dict(segment_id='T',positions=[3,4],token_ids=f.tokens[3:5])])
        self.job = dict(operation='source_request',request=request,request_sha256=digest_json(request),
                        sources_by_segment={'T':[]})

    def verify(self):
        return verify_natural_p0_birth(self.fixture.store, self.job,
            source_id=self.source['source_id'],target_id='T',expected_num_layers=32)

    def test_full_context_and_files_verified_without_lru_or_gpu_claim(self):
        store=self.fixture.store
        before=deepcopy(store._catalog)
        result=self.verify()
        self.assertEqual(store._catalog,before)
        self.assertFalse(result['actual_GPU_execution_verified'])
        self.assertFalse(result['P1_execution_allowed'])
        self.assertEqual(result['artifact_digest'],self.source['artifact_digest'])

    def test_same_target_but_different_prefix_refused(self):
        self.job['request']['token_ids'][0]=77
        self.job['request_sha256']=digest_json(self.job['request'])
        with self.assertRaisesRegex(ValueError,'historical context'):self.verify()

    def test_wrong_birth_id_and_target_position_refused(self):
        self.job['request']['request_id']='other-birth'
        self.job['request_sha256']=digest_json(self.job['request'])
        with self.assertRaisesRegex(ValueError,'birth/target'):self.verify()

    def test_teacher_reference_cannot_supply_source(self):
        self.job['request'].update(capture_logits=True,teacher_token_ids=[])
        self.job['request_sha256']=digest_json(self.job['request'])
        with self.assertRaisesRegex(ValueError,'ordinary cold-miss'):self.verify()

    def test_corrupt_backing_fails(self):
        store=self.fixture.store;row=store._catalog['rows'][self.source['source_id']]
        path=store._path(row['kv_file'])
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'file digest'):self.verify()

    def test_active_request_and_missing_source_refused(self):
        store=self.fixture.store;snapshot=store.begin_request('active')
        with self.assertRaisesRegex(ValueError,'quiescent'):self.verify()
        store.end_request(snapshot)
        self.source['source_id']='unknown'
        with self.assertRaisesRegex(ValueError,'published'):self.verify()

    def test_short_host_budget_refused(self):
        store=self.fixture.store
        # Constructor already tests configuration persistence; only exercise
        # the verifier's bounded allocation guard in this isolated fixture.
        from unittest.mock import patch
        with patch.object(store,'_guard'), patch.dict(store.config,staging_bytes=1):
            with self.assertRaisesRegex(MemoryError,'budget'):self.verify()


if __name__=='__main__':unittest.main()
