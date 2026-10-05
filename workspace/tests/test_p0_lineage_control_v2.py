"""CPU-only contracts: no real-model or CUDA qualification claims."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from probekv.p0_lineage_control_v2 import validate_lineage_job, _commit_recipe, _observe_and_prepare_parents
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager, HBMReservationKind
from probekv.source_comparison_v2 import ComparisonSessionV2
from probekv.native_publication_v2 import publish_compared_targets_v2
from tests import test_native_comparison_publication_v2 as publication_fixtures
from tests.test_native_source_capture_v2 import commit
from tests.test_native_source_capture_v2 import fixture
from tests.test_native_source_capture_v2 import CPUFakePinnedModel
from probekv.model_adapters import PinnedCacheBlendResumableAdapter
from probekv.resumable_prefill import ProbeKVResumablePrefillSession
from probekv.native_source_capture_v2 import NativeTargetCaptureV2
from probekv.source_manifest_v2 import RequestManifestRegistry, ParentSourceMetadata
from probekv.source_provenance_v2 import SourceOrigin


def job():
    tokens = list(range(24))
    return dict(operation='controlled_lineage_birth', input_origin='controlled_provenance_diagnostic',
        economic_admission_evaluated=False, production_reuse_commit_observed=False,
        request=dict(token_ids=tokens, max_new_tokens=1, segments=[
            dict(segment_id='U', positions=list(range(1,11)), token_ids=tokens[1:11]),
            dict(segment_id='T', positions=list(range(11,21)), token_ids=tokens[11:21])]),
        upstream_segment_id='U', target_id='T', sources_by_segment={'U':[dict(source_id='source')], 'T':[]},
        expected_parent_generation=0, expected_target_generation=1,
        comparison_profile=dict(provenance_policy='ALLOW_MIXED_G1', completed_depth=2))


class LineageContractsTests(unittest.TestCase):
    def test_parent_prepare_version_change_requires_fresh_next_observation(self):
        @dataclass(frozen=True)
        class Receipt:
            segment_id: str
            context_generation: int
            eligible_ids: tuple
            available_ids: tuple
            compared_ids: tuple
            winner_source_id: str
        context=NS(generation=2);order=[]
        def compare(c,sid):
            order.append(('observe',sid,c.generation))
            return Receipt(sid,c.generation,(sid,),(sid,),(sid,),sid)
        def prepare(receipt):
            self.assertEqual(receipt.context_generation,context.generation)
            order.append(('prepare',receipt.segment_id,context.generation))
            context.generation+=1
        context.prepare_selected_v2=prepare
        audit={'comparison_receipts':{}}
        receipts=_observe_and_prepare_parents(context,NS(parent_metadata=lambda *a:NS(generation=0)),
            object(),NS(compare_native=compare),('3','5'),{'3':('3',),'5':('5',)},
            {'3':0,'5':0},audit)
        self.assertEqual(order,[('observe','3',2),('prepare','3',2),('observe','5',3),('prepare','5',3)])
        self.assertEqual(receipts['3'].context_generation,2)
        self.assertEqual(receipts['5'].context_generation,3)

    def interleaved_job(self):
        j=job();j.pop('target_id');j.pop('expected_target_generation')
        j.pop('upstream_segment_id');j.pop('expected_parent_generation')
        tokens=list(range(62));j['request']['token_ids']=tokens
        j['request']['segments']=[dict(segment_id=str(i),positions=list(range(1+(i-1)*10,1+i*10)),
            token_ids=tokens[1+(i-1)*10:1+i*10]) for i in range(1,7)]
        j.update(reuse_segment_ids=['3','5'],expected_parent_generations={'3':0,'5':0},
            target_ids=['1','2','4','6'],expected_target_generations={'1':0,'2':0,'4':1,'6':1},
            sources_by_segment={str(i):[dict(source_id='s'+str(i))] if i in (3,5) else [] for i in range(1,7)})
        return j

    def test_six_segment_recipe_generation_is_causal_max_not_count(self):
        j=self.interleaved_job();validate_lineage_job(j,total_layers=32)
        j['expected_parent_generations']['5']=1;j['expected_target_generations']['6']=2
        validate_lineage_job(j,total_layers=32)
        j['expected_target_generations']['4']=2
        with self.assertRaises(ValueError):validate_lineage_job(j,total_layers=32)

    def test_six_segment_actual_CPU_capture_publishes_exact_and_mixed(self):
        for second_generation in (0,1):
            tokens=tuple(range(14));model=CPUFakePinnedModel()
            adapter=PinnedCacheBlendResumableAdapter(model,NS(adapter_name='cpu_fake',num_layers=3))
            session=ProbeKVResumablePrefillSession(adapter,'model',tokens,None,[]);session.begin_prefill()
            targets={'1':(1,2),'2':(3,4),'4':(7,8),'6':(11,12)}
            registry=RequestManifestRegistry(max_bytes=2000000,max_manifest_bytes=1000000)
            tracker=NativeTargetCaptureV2(request_id='interleaved',token_ids=tokens,model_signature='model',
                authorization_domain='domain',num_layers=3,targets=targets,selection_depths=(1,2),
                host_budget_bytes=100000,kv_heads=1,head_dim=2,registry=registry)
            session.provenance_capture=tracker;session.advance_to_layer(1)
            for sid,positions,g in (('3',(5,6),0),('5',(9,10),second_generation)):
                tracker.bind_parent(ParentSourceMetadata('source'+sid,sid*64,'model','domain',
                    SourceOrigin.EXACT if g==0 else SourceOrigin.MIXED,g))
                session.register_source_handle(sid,'source'+sid,object())
                session.commit_segment_reuse(segment_id=sid,source_id='source'+sid,boundary=2,
                    segment_positions=positions,repair_positions=(positions[0],))
            session.finish_prefill();out=tracker.finalize(request_completed=True,actual_token_ids=tokens)
            expected={'1':0,'2':0,'4':1,'6':second_generation+1}
            for sid,g in expected.items():
                self.assertEqual(tracker.ledger.target_proof(targets[sid]).generation,g)
                self.assertEqual(sid in out,g<=1)
            self.assertEqual(model.real_block_calls,3)
            self.assertEqual(registry.storage_audit()['manifest_count'],1)

    def test_multi_parent_precheck_rejects_before_any_commit(self):
        c=self.context();c.prepared['V']=object();c.supports['V']={3:(11,)}
        c.segments['V']={'positions':tuple(range(11,21))}
        with self.assertRaises(ValueError):_commit_recipe(c,('U','V'))
        c.engine.commit_ready_segment.assert_not_called()

    def multi_job(self, generation=0):
        j=job();j.pop('target_id');j.pop('expected_target_generation')
        j['expected_parent_generation']=generation
        tokens=j['request']['token_ids']
        j['request']['segments'][1]=dict(segment_id='B',positions=list(range(11,16)),token_ids=tokens[11:16])
        j['request']['segments'].append(dict(segment_id='C',positions=list(range(16,21)),token_ids=tokens[16:21]))
        j['target_ids']=['B','C'];j['expected_target_generations']={'B':generation+1,'C':generation+1}
        j['sources_by_segment']={'U':j['sources_by_segment']['U'],'B':[],'C':[]}
        return j

    def test_T03_T05_multi_target_recipe_no_per_segment_increment(self):
        for generation in (0,1):
            j=self.multi_job(generation);validate_lineage_job(j,total_layers=32)
            j['expected_target_generations']['C']=generation+2
            with self.assertRaises(ValueError):validate_lineage_job(j,total_layers=32)

    def test_T07_before_reused_segment_requires_explicit_zero_generation(self):
        j=self.multi_job();j['upstream_segment_id']='C';j['target_ids']=['U','B']
        j['sources_by_segment']={'C':[dict(source_id='source')],'U':[],'B':[]}
        j['expected_target_generations']={'U':0,'B':0}
        validate_lineage_job(j,total_layers=32)
        j['expected_target_generations']['B']=1
        with self.assertRaises(ValueError):validate_lineage_job(j,total_layers=32)

    def test_multi_target_ambiguous_duplicate_or_overlapping_recipe_rejects(self):
        for mutate in (lambda j:j.update(target_id='B'),
                       lambda j:j.update(target_ids=['B','B']),
                       lambda j:j['expected_target_generations'].update(C=True),
                       lambda j:j['request']['segments'][2].update(positions=list(range(11,16)),token_ids=list(range(11,16)))):
            j=self.multi_job();mutate(j)
            with self.assertRaises(ValueError):validate_lineage_job(j)

    def test_T03_T05_actual_CPU_hooks_keep_both_full_targets_same_generation(self):
        for generation in (0,1):
            m,s,t,registry=fixture(targets={'B':(2,3),'C':(4,)})
            s.advance_to_layer(1);commit(s,t,generation=generation)
            s.finish_prefill();out=t.finalize(request_completed=True,actual_token_ids=tuple(range(6)))
            self.assertEqual(m.real_block_calls,3)
            for target in ('B','C'):
                self.assertEqual(t.ledger.target_proof(t.targets[target]).generation,generation+1)
                self.assertEqual(target in out,generation==0)
            self.assertEqual(registry.storage_audit()['manifest_count'],1)

    def test_T07_actual_CPU_hooks_suffix_reuse_preserves_earlier_exact_target(self):
        m,s,t,registry=fixture(targets={'T':(0,1)})
        s.advance_to_layer(1);commit(s,t,positions=(3,4),repair=(3,))
        s.finish_prefill();out=t.finalize(request_completed=True,actual_token_ids=tuple(range(6)))
        self.assertEqual(out['T']['generation'],0)
        self.assertEqual(out['T']['origin'],'EXACT_CONTEXT')
        self.assertEqual(m.real_block_calls,3)

    def test_G0_G1_and_G1_G2_recipes_validate(self):
        for g in (0,1):
            j=job();j.update(expected_parent_generation=g,expected_target_generation=g+1)
            validate_lineage_job(j,total_layers=32)

    def test_teacher_reference_and_extra_forward_flags_reject(self):
        for changes in ({'teacher_token_ids':[]},{'capture_logits':True},{'max_new_tokens':32},
                        {'native_dense_continuation':True},{'capture_original_full_prefill':True},
                        {'publish_exact_prefix_shadow':True},{'correctness_repair_ratio':1.}):
            j=job();j['request'].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_lineage_job(j)

    def test_no_production_or_generation_relabeling(self):
        for changes in ({'economic_admission_evaluated':True},{'production_reuse_commit_observed':True},
                        {'input_origin':'frozen_development'},{'expected_parent_generation':2},
                        {'expected_target_generation':0},{'operation':'source_request'}):
            j=job();j.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_lineage_job(j)

    def test_exact_policy_illegal_depth_bad_geometry_reject(self):
        j=job();j['comparison_profile']['provenance_policy']='EXACT_ONLY'
        with self.assertRaises(ValueError):validate_lineage_job(j)
        j=job();j['comparison_profile']['completed_depth']=32
        with self.assertRaises(ValueError):validate_lineage_job(j,total_layers=32)
        j=job();j['request']['segments'][1]['token_ids'][0]=99
        with self.assertRaises(ValueError):validate_lineage_job(j)

    def context(self):
        hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=1000,safety_bytes=0)
        reservation=hbm.reserve_batch(owner_request_id='q',rows=(('U',100,HBMReservationKind.WINNER_PREFETCH),))[0]
        return NS(finished=False,closed=False,committed={},prepared={'U':object()},
            source_consumption_v2=NS(assert_can_commit=Mock()),current_completed_depth=2,
            supports={'U':{3:(1,2)}},segments={'U':{'positions':tuple(range(1,11))}},
            engine=NS(commit_ready_segment=Mock()),adapter=NS(hbm=hbm),
            replica_reservations={'U':reservation},generation=3)

    def test_control_commit_checks_ownership_and_promotes_without_fake_costs(self):
        c=self.context();_commit_recipe(c,'U')
        c.source_consumption_v2.assert_can_commit.assert_called_once_with('U')
        self.assertEqual(c.committed,{'U':3})
        self.assertEqual(c.replica_reservations['U'].kind,HBMReservationKind.COMMITTED_EXECUTION)
        self.assertEqual(c.engine.commit_ready_segment.call_args.kwargs['repair_positions'],(1,2))
        with self.assertRaises(ValueError):_commit_recipe(c,'U')

    def test_invalid_ownership_and_mask_do_not_execute(self):
        c=self.context();c.source_consumption_v2.assert_can_commit.side_effect=ValueError('lease stale')
        with self.assertRaises(ValueError):_commit_recipe(c,'U')
        c.engine.commit_ready_segment.assert_not_called()
        c=self.context();c.supports['U'][3]=(1,)
        with self.assertRaises(ValueError):_commit_recipe(c,'U')
        c.engine.commit_ready_segment.assert_not_called()

    def test_real_CPU_capture_and_issued_receipt_publish_G1_but_reject_G2(self):
        # Actual CPU block hooks, ledger, durable store, authentic miss receipt.
        # Parent source is a typed CPU fixture, not GPU consumption evidence.
        for g in (0,1):
            f=publication_fixtures.NativeComparisonPublicationTests();f.setUp()
            try:
                c,model,store,snapshot,issuer=f.prepared()
                receipt=issuer.compare_native(c,'C')
                commit(c.engine.session,c.source_capture_v2,generation=g)
                f.fixture.finish(c)
                proof=c.source_capture_v2.ledger.target_proof((2,3,4))
                self.assertEqual(proof.generation,g+1)
                report=publish_compared_targets_v2(c,issuer,{'C':receipt})
                self.assertEqual(model.real_block_calls,3)
                self.assertFalse(store.lookup(snapshot,(2,3,4)))
                store.end_request(snapshot)
                reader=store.begin_request('later')
                rows=store.lookup(reader,(2,3,4))
                if g==0:
                    self.assertTrue(report['targets']['C']['publication_performed'])
                    self.assertEqual(rows[0]['generation'],1)
                    with store.leased_target(reader,rows[0]['source_id']) as tensors:
                        self.assertEqual(len(tensors),3)
                    self.assertEqual(store.parent_metadata(reader,rows[0]['source_id']).generation,1)
                else:
                    self.assertFalse(rows)
                    self.assertFalse(report['targets']['C']['publication_performed'])
                    self.assertIn('only complete known G0/G1',c.source_capture_v2.rejections['C'])
                store.end_request(reader)
            finally:
                f.doCleanups()
