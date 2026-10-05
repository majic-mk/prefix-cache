from copy import deepcopy
import hashlib,json,tempfile,unittest
from pathlib import Path
from probekv.p1_input_consumer_v2 import resolve_input_graph,load_frozen_input_graph
from probekv.v8_schema10_execution import digest_json


class InputConsumerTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(model_signature='a'*64,input_model_revision='model@'+'c'*40,
            tokenizer_file_sha256='b'*64,runtime_tokenizer_identity='d'*64)
        rows=[]
        for i in range(5):
            q=dict(request_id='q'+str(i),token_ids=[999+i]+list(range(512))+[999],
                segments=[dict(segment_id='C',token_ids=list(range(512)),positions=list(range(1,513)))],
                content_group='g',partition_role='fit',development_partition_digest='p',origin_example_id='q'+str(i))
            rows.append(dict(role='historical_birth' if i<4 else 'future_consumer',
                request=q,request_sha256=digest_json(q),pseudotime=format(i,'064x')))
        s0q=deepcopy(rows[0]['request']);s0q['request_id']='S0'
        s0=dict(request=s0q,recipe_sha256=digest_json(s0q),question_included=False,answer_included=False,other_documents_included=False)
        group=dict(group_id='g',model_signature=self.identity['input_model_revision'],tokenizer_sha256='b'*64,
            partition_role='fit',partition_digest='p',chronology='corpus_pseudotime',target_token_count=512,
            requests=rows,independent_S0=s0)
        builds=[]
        for i in range(5):
            builds.append(dict(build_id='b'+str(i),group_id='g',role='historical_exact' if i<4 else 'independent_S0',
                request_sha256=rows[i]['request_sha256'] if i<4 else s0['recipe_sha256'],
                request_ref=dict(file='group-00.json',**(dict(index=i) if i<4 else dict(key='independent_S0.request'))),
                maximum_seconds=180,artifact_id=None))
        actions=[]
        for arm in ('dense','historical_1','historical_2','historical_3','historical_4','S0'):
            actions.append(dict(stage='P1-E',group_id='g',request_id='q4',request_sha256=rows[4]['request_sha256'],arm=arm,
                repair=None if arm=='dense' else 'legacy_normalized_kv',repair_ratio=None if arm=='dense' else .15,
                first_reuse_layer=None if arm=='dense' else 9,maximum_seconds=180,execution_allowed=False))
        self.docs={'group-00.json':group,'partition_extension.json':dict(digest='p',rows=[dict(group_id='g',component='g',role='fit')]),
            'input_freeze.json':dict(partition_digest='p',old_partition_modified=False,groups=1,targets=1,P1_E_actions=6,P1_E_builds=5,P1_M_actions=0,P1_M_builds=0),
            'planned_P1_E_actions.json':dict(actions=actions,builds=builds),
            'planned_P1_M_actions.json':dict(main_actions=[],builds=[])}
        self.rehash()

    def rehash(self):
        for name,field,hashfield in (('planned_P1_E_actions.json','actions','consumer_actions_sha256'),('planned_P1_M_actions.json','main_actions','actions_sha256')):
            p=self.docs[name];p[hashfield]=digest_json(p[field]);p['builds_sha256']=digest_json(p['builds'])

    def run_graph(self):return resolve_input_graph(self.docs,**self.identity)

    def test_resolves_all_builds_no_gpu_or_p1_authority(self):
        before=deepcopy(self.docs);r=self.run_graph()
        self.assertEqual(self.docs,before)
        self.assertEqual(len(r['source_builds']),5);self.assertEqual(len(r['consumer_actions']),6)
        self.assertEqual(r['maximum_build_seconds'],900);self.assertEqual(r['maximum_consumer_seconds'],1080)
        self.assertEqual(r['consumer_actions'][1]['depends_on'],['b0'])
        for key in ('GPU_execution_allowed','P1_execution_allowed','source_artifacts_verified','paper_evidence'):
            self.assertFalse(r[key])

    def test_future_birth_and_changed_target_refused(self):
        self.docs['group-00.json']['requests'][0]['pseudotime']='f'*64
        with self.assertRaisesRegex(ValueError,'future Source'):self.run_graph()

    def test_request_digest_recomputed_after_token_change(self):
        self.docs['group-00.json']['requests'][0]['request']['token_ids'][3]=5000
        with self.assertRaisesRegex(ValueError,'digest'):self.run_graph()

    def test_s0_question_is_forbidden_even_if_hash_updated(self):
        s0=self.docs['group-00.json']['independent_S0'];s0['request']['question']='leak'
        s0['recipe_sha256']=digest_json(s0['request'])
        with self.assertRaisesRegex(ValueError,'leakage'):self.run_graph()

    def test_missing_and_duplicate_builds_fail(self):
        self.docs['planned_P1_E_actions.json']['builds'].pop();self.rehash()
        with self.assertRaisesRegex(ValueError,'build is missing'):self.run_graph()

    def test_complete_arm_matrix_not_just_summary(self):
        p=self.docs['planned_P1_E_actions.json'];p['actions'].pop()
        self.docs['input_freeze.json']['P1_E_actions']=5;self.rehash()
        with self.assertRaisesRegex(ValueError,'arm matrix'):self.run_graph()

    def test_consumer_policy_and_fabricated_artifact_rejected(self):
        p=self.docs['planned_P1_E_actions.json'];p['actions'][1]['repair_ratio']=.20;self.rehash()
        with self.assertRaisesRegex(ValueError,'consumer policy'):self.run_graph()
        p['actions'][1]['repair_ratio']=.15;p['builds'][0]['artifact_id']='fake';self.rehash()
        with self.assertRaisesRegex(ValueError,'fabricated'):self.run_graph()

    def test_model_alias_not_interchangeable_with_runtime_hash(self):
        self.identity['input_model_revision']='a'*64
        with self.assertRaisesRegex(ValueError,'model name'):self.run_graph()

    def test_mixed_parent_pair_dependency(self):
        group=self.docs['group-00.json'];q=deepcopy(group['requests'][1]['request'])
        q['segments'].insert(0,dict(segment_id='U',positions=[0],token_ids=[q['token_ids'][0]]))
        parent=deepcopy(q);parent['request_id']='parent';parent['segments']=parent['segments'][:1]
        recipe=dict(group_id='g',parent_request=parent,paired_birth_request=q,
            paired_input_token_sha256=digest_json(q['token_ids']),target_positions_sha256=digest_json(q['segments'][1]['positions']),
            partition_digest='p',partition_role='fit',target_expected_generations=dict(E=0,M1=1),
            parent_generation=0,parent_origin='EXACT_CONTEXT',parent_pseudotime='0'*64,birth_pseudotime=format(1,'064x'),consumer_pseudotimes=[format(4,'064x')],
            exact_birth=dict(target_id='C',upstream_mode='FULL',target_mode='FULL'),
            mixed_birth=dict(target_id='C',target_mode='FULL',completed_depth=2,first_selective_reuse_layer=3,
                repair_metric='legacy_normalized_kv',repair_ratio=.15,mask_digest=None))
        self.docs['paired-birth-00.json']=recipe
        m=self.docs['planned_P1_M_actions.json']
        m['builds']=[dict(group_id='g',role=role,recipe_file='paired-birth-00.json',recipe_sha256=digest_json(recipe),
            maximum_seconds=180,artifact_id=None,execution_allowed=False) for role in ('G0_parent','exact_E','mixed_M1')]
        for arm in ('dense','E','M1'):
            m['main_actions'].append(dict(stage='P1-M',group_id='g',request_id='q4',request_sha256=group['requests'][4]['request_sha256'],
                arm=arm,first_reuse_layer=None if arm=='dense' else 9,repair_metric=None if arm=='dense' else 'legacy_normalized_kv',
                repair_ratio=None if arm=='dense' else .15,maximum_seconds=180,execution_allowed=False))
        self.docs['input_freeze.json'].update(P1_M_builds=3,P1_M_actions=3);self.rehash()
        result=self.run_graph()
        self.assertEqual(result['source_builds'][-1]['depends_on'],['M:g:G0_parent'])
        recipe['parent_request']['segments'][0]['token_ids']=[-1]
        for b in m['builds']:b['recipe_sha256']=digest_json(recipe)
        self.rehash()
        with self.assertRaises(ValueError):self.run_graph()

    def test_checksum_loader_rejects_tampering_and_bad_external_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);index={}
            for name,value in self.docs.items():
                raw=json.dumps(value).encode();(root/name).write_bytes(raw);index[name]=hashlib.sha256(raw).hexdigest()
            raw=json.dumps(index).encode();(root/'files.sha256.json').write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            self.assertEqual(len(load_frozen_input_graph(root,index_sha256=sha,**self.identity)['source_builds']),5)
            with self.assertRaisesRegex(ValueError,'external'):
                load_frozen_input_graph(root,index_sha256='0'*64,**self.identity)
            (root/'group-00.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_frozen_input_graph(root,index_sha256=sha,**self.identity)


if __name__=='__main__':unittest.main()
