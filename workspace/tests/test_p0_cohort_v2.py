"""Pure CPU cohort/S0 contracts; no real dataset or GPU qualification."""
import copy,hashlib,unittest
from probekv.canonical_segment import canonicalize_token_ids
from probekv.data import deterministic_group_split
from probekv.p0_cohort_v2 import independent_s0_request,qualify_group,paired_mixed_birth_recipe
from probekv.rag_data import RAGDocument,RAGExample,segment_text,render_preceding_context


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.encode=lambda s:[ord(c) for c in s]
        did=next(str(i) for i in range(1000) if deterministic_group_split('HotPotQA:'+str(i),20260726)=='calibration')
        self.doc=RAGDocument(did,'target','x'*1024,False,1)
        self.examples={};rows=[]
        for i in range(5):
            origin='origin'+str(i)
            prefix=RAGDocument('p'+str(i),'prefix'+str(i),'context'+str(i),False,0)
            example=RAGExample('HotPotQA',origin,'question',('answer',),(prefix,self.doc))
            self.examples[origin]=example;event=origin+':'+did
            rows.append(dict(origin_id=origin,event_id=event,pseudotime=hashlib.sha256(('20260726:'+event).encode()).hexdigest(),
                             prefix_sha256=hashlib.sha256(render_preceding_context((prefix,)).encode()).hexdigest()))
        rows.sort(key=lambda r:r['pseudotime'])
        self.tokens=tuple(self.encode(segment_text(self.doc)))
        self.target=canonicalize_token_ids(self.tokens,tokenizer_signature='tok',document_revision='raw')[0]
        self.assertEqual(self.target.token_count,512)
        self.group=dict(group_id='HotPotQA:'+did,document_id=did,sources=rows[:4],targets=rows[4:],
            target_tokens=list(self.target.token_ids),token_start=self.target.token_start,token_end=self.target.token_end,
            canonicalizer_signature=self.target.canonicalizer_signature)

    def qualify(self, group=None):
        return qualify_group(group or self.group,self.examples,self.encode,tokenizer_sha256='tok',raw_sha256='raw',
            model_signature='model',partition_digest='partition',partition_role='validation')

    def test_whole_document_S0_excludes_question_answer_and_other_context(self):
        r=self.qualify();s=r['independent_S0'];q=s['request']
        self.assertNotIn('question',q);self.assertNotIn('answers',q)
        self.assertIn(self.tokens,tuple(tuple(q['token_ids'][i:i+len(self.tokens)]) for i in range(len(q['token_ids'])-len(self.tokens)+1)))
        self.assertFalse(s['artifact_built']);self.assertFalse(r['P1_execution_allowed'])

    def test_exact_tokens_and_four_distinct_histories_required(self):
        g=copy.deepcopy(self.group);g['target_tokens'][0]+=1
        with self.assertRaises(ValueError):self.qualify(g)
        g=copy.deepcopy(self.group);g['sources'][1]=g['sources'][0]
        with self.assertRaises(ValueError):self.qualify(g)

    def test_future_history_and_forged_prefix_rejected(self):
        g=copy.deepcopy(self.group);g['sources'][0],g['targets'][0]=g['targets'][0],g['sources'][0]
        with self.assertRaisesRegex(ValueError,'future'):self.qualify(g)
        g=copy.deepcopy(self.group);g['sources'][0]['prefix_sha256']='forged'
        with self.assertRaisesRegex(ValueError,'prefix'):self.qualify(g)

    def test_locked_split_cannot_be_relabelled(self):
        g=copy.deepcopy(self.group)
        g['group_id']=next('HotPotQA:'+str(i) for i in range(1000) if deterministic_group_split('HotPotQA:'+str(i),20260726)=='test')
        with self.assertRaisesRegex(ValueError,'relabel'):self.qualify(g)

    def test_long_original_context_rejected_not_truncated(self):
        ex=self.examples[self.group['targets'][0]['origin_id']]
        self.examples[ex.example_id]=RAGExample(ex.dataset,ex.example_id,'q'*4096,ex.answers,ex.documents)
        with self.assertRaisesRegex(ValueError,'truncation'):self.qualify()

    def test_missing_original_and_changed_pseudotime_are_not_qualification(self):
        g=copy.deepcopy(self.group);g['sources'][0]['pseudotime']='0'*64
        with self.assertRaisesRegex(ValueError,'pseudotime'):self.qualify(g)
        self.examples.pop(self.group['targets'][0]['origin_id'])
        with self.assertRaises(KeyError):self.qualify()

    def test_epoch_order_cannot_differ_from_pseudotime(self):
        g=copy.deepcopy(self.group);g['sources'].reverse()
        with self.assertRaisesRegex(ValueError,'epochs'):self.qualify(g)

    def test_group_identity_and_partition_required(self):
        g=copy.deepcopy(self.group);g['document_id']='forged'
        with self.assertRaisesRegex(ValueError,'identity'):self.qualify(g)


class PairedRecipeTests(unittest.TestCase):
    def setUp(self):
        self.birth=dict(origin_example_id='b',development_partition_digest='p',partition_role='fit',
            token_ids=list(range(900)),segments=[dict(segment_id='C',token_ids=list(range(300,812)),
            positions=list(range(300,812)),content_key='c')],mandatory_suffix_positions=list(range(812,900)))
        self.parent=copy.deepcopy(self.birth);self.parent['origin_example_id']='a'
        self.upstream=dict(token_ids=list(range(50,178)),content_key='u')

    def recipe(self):
        return paired_mixed_birth_recipe(self.birth,self.parent,self.upstream,
            parent_pseudotime='1',birth_pseudotime='2',consumer_pseudotimes=['3'])

    def test_recipe_preserves_prompt_and_does_not_invent_mask_or_artifacts(self):
        before=copy.deepcopy(self.birth);r=self.recipe()
        self.assertEqual(before,self.birth)
        self.assertEqual(r['paired_birth_request']['token_ids'],self.birth['token_ids'])
        self.assertEqual(r['target_expected_generations'],dict(E=0,M1=1))
        self.assertIsNone(r['mixed_birth']['mask_digest'])
        self.assertFalse(r['execution_allowed'])

    def test_target_or_future_cannot_be_upstream(self):
        self.upstream['token_ids']=list(range(500,628))
        with self.assertRaisesRegex(ValueError,'missing'):self.recipe()

    def test_parent_must_have_same_tokens_and_partition(self):
        self.parent['token_ids'][70]=-1
        with self.assertRaisesRegex(ValueError,'missing'):self.recipe()
        self.parent['token_ids'][70]=70;self.parent['partition_role']='validation'
        with self.assertRaisesRegex(ValueError,'partition'):self.recipe()

    def test_forged_target_positions_and_tokens_reject(self):
        self.birth['segments'][0]['positions'][1]+=1
        with self.assertRaisesRegex(ValueError,'512'):self.recipe()
        self.birth['segments'][0]['positions'][1]-=1
        self.birth['segments'][0]['token_ids'][0]=-1
        with self.assertRaisesRegex(ValueError,'512'):self.recipe()


if __name__=='__main__':unittest.main()
