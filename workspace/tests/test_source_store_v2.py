"""Real local file/CPU tensor transactions, not GPU or QA qualification."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from probekv.segment_capture_v2 import SegmentCapture
from probekv.source_manifest_v2 import (ParentSourceMetadata, RequestManifestRegistry,
    TargetOccurrence, request_input_digest)
from probekv.source_provenance_v2 import (KVImport, PublicationScope,
    RequestExecutionLedger, SourceOrigin)
from probekv.source_store_v2 import TargetSourceStoreV2, parse_target_catalog_v2
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import tensor_digest


class TargetCatalogMetadataTests(unittest.TestCase):
    def envelope(self):
        payload = dict(kind='target_source_catalog_v2', epoch=0, use_epoch=0,
            content_opportunities={}, rows={}, config=dict(model_signature='model',
                authorization_domain='domain', tokenizer_hash='tokenizer', policy='ALLOW_MIXED_G1',
                max_bytes=2000000, staging_bytes=1000000, max_variants=4,
                probation_opportunities=2, purpose='isolated_P0_diagnostic'))
        return dict(payload=payload, digest=digest_json(payload))

    def test_actual_empty_and_published_catalog_files_share_public_parser(self):
        fixture = TargetSourceStoreTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        store = fixture.store()
        for populate in (False, True):
            if populate:
                fixture.publish(store, fixture.candidate('birth'))
            raw = json.loads((store.root / 'catalog.json').read_text(encoding='utf-8'))
            parsed = parse_target_catalog_v2(raw)
            self.assertEqual(parsed, store._catalog)
            self.assertEqual(len(parsed['rows']), int(populate))
            parsed['config']['model_signature'] = 'detached-copy'
            self.assertEqual(raw['payload']['config']['model_signature'], 'model')

    def test_flat_extra_fields_and_stale_inner_digest_are_rejected(self):
        envelope = self.envelope()
        with self.assertRaises(ValueError): parse_target_catalog_v2(envelope['payload'])
        envelope['extra'] = True
        with self.assertRaises(ValueError): parse_target_catalog_v2(envelope)
        envelope = self.envelope(); envelope['payload']['epoch'] = 1
        with self.assertRaisesRegex(ValueError, 'digest'): parse_target_catalog_v2(envelope)

    def test_rehashed_invalid_config_and_counters_are_rejected(self):
        for key, value in (('max_bytes', True), ('staging_bytes', 0), ('max_variants', 8),
                           ('policy', 'UNKNOWN'), ('probation_opportunities', -1),
                           ('purpose', 'production'), ('authorization_domain', '')):
            envelope = self.envelope(); envelope['payload']['config'][key] = value
            envelope['digest'] = digest_json(envelope['payload'])
            with self.subTest(key=key), self.assertRaises(ValueError): parse_target_catalog_v2(envelope)
        for key, value in (('epoch', True), ('use_epoch', -1), ('content_opportunities', {'x': False}),
                           ('rows', []), ('rows', {'one': {'source_id': 'different'}})):
            envelope = self.envelope(); envelope['payload'][key] = value
            envelope['digest'] = digest_json(envelope['payload'])
            with self.subTest(key=key), self.assertRaises(ValueError): parse_target_catalog_v2(envelope)


class TargetSourceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = RequestManifestRegistry(max_bytes=2000000,
            max_manifest_bytes=200000, persistent_root=self.root/'manifests')
        self.tokens = (10, 11, 12, 13)
        self.target_tokens = (12, 13)
        self.stores = []
        self.addCleanup(self.close_stores)

    def close_stores(self):
        for store in reversed(self.stores):
            store.close()

    def store(self, name='pool', **override):
        arguments = dict(registry=self.registry, model_signature='model',
            authorization_domain='domain', tokenizer_hash='tokenizer',
            policy='ALLOW_MIXED_G1', max_bytes=2000000, staging_bytes=1000000,
            max_variants=4, probation_opportunities=0)
        arguments.update(override)
        result = TargetSourceStoreV2(self.root/name, **arguments)
        self.stores.append(result)
        return result

    def candidate(self, request, *, base=0, mixed=False, parent=None, target=(2,3)):
        positions = (0, 1, 2, 3)
        ledger = RequestExecutionLedger(request_id=request,
            input_digest=request_input_digest(self.tokens, positions), model_signature='model',
            token_count=4, num_layers=3, exact_input_proof='cpu-native-input')
        if mixed and parent is None:
            parent = ParentSourceMetadata('parent', 'a'*64, 'model', 'domain', SourceOrigin.EXACT, 0)
        capture = SegmentCapture(ledger, target, byte_budget=10000, selection_depths=(1,2))
        for layer in (1,2,3):
            current = (1,2,3) if mixed else positions
            ledger.record_layer(layer_1based=layer, qkv_rows=positions,
                attention_rows=current, output_mlp_rows=current,
                effective_current_kv_rows=current,
                imported_kv=(KVImport(0,parent.source_id,parent.generation),) if mixed else (),
                completion_reference='cpu-layer-'+str(layer))
            k = (torch.arange(16).reshape(4,2,2)+base+layer).to(torch.bfloat16)
            v = (k.float()+17).to(torch.bfloat16)
            capture.record_layer(layer,k,v,row_positions=positions)
        occurrence = TargetOccurrence('target',target[0],target[-1]+1)
        manifest = self.registry.register_actual_execution(ledger=ledger,token_ids=self.tokens,
            absolute_positions=positions,authorization_domain='domain',occurrences=(occurrence,),
            parent_sources=(parent,) if mixed else (),request_completed=True,input_reference='input:'+request)
        ref = self.registry.capture_reference(manifest,'target',model_signature='model',
            authorization_domain='domain',target_token_ids=tuple(self.tokens[i] for i in target),
            target_positions=target)
        return capture.finalize(ref,request_completed=True)

    def plan(self, store, snapshot, candidate, *, tokens=None):
        tokens = self.target_tokens if tokens is None else tokens
        count = len(store.lookup(snapshot,tokens))
        scope = (PublicationScope('content_miss',0,0,0,0) if not count else
                 PublicationScope('complete_scope_mismatch',count,count,count,count,2.0,1.0))
        return store.plan_publication(snapshot,candidate,token_ids=tokens,scope=scope)

    def publish(self, store, candidate, *, tokens=None):
        tokens = self.target_tokens if tokens is None else tokens
        snapshot = store.begin_request(candidate['proof'].request_id)
        plan = self.plan(store,snapshot,candidate,tokens=tokens)
        result = store.commit_publication(snapshot,plan,candidate,token_ids=tokens)
        store.end_request(snapshot)
        return result

    def read_rows(self, store, request='reader', *, tokens=None):
        snapshot = store.begin_request(request)
        rows = store.lookup(snapshot,self.target_tokens if tokens is None else tokens)
        store.end_request(snapshot)
        return rows

    def test_exact_mixed_share_capacity_at_k1_k2_k4(self):
        for capacity in (1,2,4):
            store = self.store('k'+str(capacity), max_variants=capacity)
            published = []
            for i in range(capacity+1):
                candidate = self.candidate('k%d-birth%d'%(capacity,i),base=20*i,mixed=bool(i%2))
                result = self.publish(store,candidate)
                self.assertEqual(result['status'],'PUBLISHED')
                published.append(result['source_id'])
            rows = self.read_rows(store)
            self.assertEqual(len(rows),capacity)
            self.assertNotIn(published[0],{r['source_id'] for r in rows})
            self.assertEqual(store.storage_audit()['shared_max_variants'],capacity)

    def test_current_request_cannot_see_own_new_source_or_future_epoch(self):
        store = self.store()
        candidate = self.candidate('birth')
        snapshot = store.begin_request('birth')
        plan = self.plan(store,snapshot,candidate)
        result = store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(store.lookup(snapshot,self.target_tokens),())
        with self.assertRaises(ValueError):
            store.read_selection(snapshot,result['source_id'],1)
        with self.assertRaises(ValueError):
            store.lookup(replace(snapshot,epoch=result['publication_epoch']),self.target_tokens)
        store.end_request(snapshot)
        self.assertEqual(self.read_rows(store,request='birth'),())
        self.assertEqual(len(self.read_rows(store,request='future-request')),1)

    def test_second_active_request_and_closed_snapshot_rejected(self):
        store = self.store()
        snapshot = store.begin_request('one')
        with self.assertRaises(ValueError): store.begin_request('two')
        store.end_request(snapshot)
        with self.assertRaises(ValueError): store.lookup(snapshot,self.target_tokens)

    def test_persisted_target_independent_of_birth_tensors_and_input(self):
        store = self.store()
        candidate = self.candidate('birth')
        expected = candidate['layers'][0][0].clone()
        owner = weakref.ref(candidate['layers'][0][0])
        result = self.publish(store,candidate)
        candidate['layers'][0][0].zero_()
        del candidate
        gc.collect()
        self.assertIsNone(owner())
        snapshot = store.begin_request('reader')
        with store.leased_target(snapshot,result['source_id']) as layers:
            self.assertTrue(torch.equal(layers[0][0],expected))
            self.assertEqual(tuple(layers[0][0].shape),(2,2,2))
        store.end_request(snapshot)
        self.assertEqual(store.storage_audit()['parent_owned_kv_bytes'],0)
        self.assertEqual(store.storage_audit()['prefix_shadow_bytes'],0)

    def test_exact_only_rejects_mixed_without_relaxing_legacy_semantics(self):
        store = self.store(policy='EXACT_ONLY')
        candidate = self.candidate('mixed',mixed=True)
        snapshot = store.begin_request('mixed')
        with self.assertRaises(ValueError): self.plan(store,snapshot,candidate)
        self.assertEqual(store.storage_audit()['source_count'],0)
        store.end_request(snapshot)

    def test_forged_proof_selection_and_parent_view_rejected_before_disk(self):
        store = self.store()
        candidate = self.candidate('birth')
        snapshot = store.begin_request('birth')
        bad_proof = dict(candidate,proof=replace(candidate['proof'],ledger_digest='f'*64))
        bad_states = dict(candidate,selection_states={})
        altered = {d:k.clone() for d,k in candidate['selection_states'].items()}
        altered[1].add_(1)
        wrong_states = dict(candidate,selection_states=altered)
        aliased = dict(candidate,selection_states={1:candidate['layers'][1][0],2:candidate['layers'][2][0]})
        for bad in (bad_proof,bad_states,wrong_states,aliased):
            with self.subTest(kind=id(bad)),self.assertRaises(ValueError):
                self.plan(store,snapshot,bad)
        self.assertEqual(list(store.root.glob('*.kv')),[])
        store.end_request(snapshot)

    def test_candidate_full_digest_changed_after_plan_is_rejected(self):
        store = self.store()
        candidate = self.candidate('birth')
        snapshot = store.begin_request('birth')
        plan = self.plan(store,snapshot,candidate)
        candidate['layers'][0][1].add_(1)
        result = store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(store.storage_audit()['source_count'],0)
        store.end_request(snapshot)

    def test_full_digest_duplicate_preserves_existing_provenance_without_write(self):
        store = self.store(max_variants=1)
        first = self.publish(store,self.candidate('exact'))
        before = len(list(store.root.glob('*.kv')))
        result = self.publish(store,self.candidate('mixed',mixed=True))
        self.assertEqual(result['status'],'DUPLICATE')
        self.assertEqual(result['source_id'],first['source_id'])
        self.assertEqual(result['existing_origin'],SourceOrigin.EXACT.value)
        self.assertFalse(result['provenance_changed'])
        self.assertEqual(result['rewritten_bytes'],0)
        self.assertEqual(len(list(store.root.glob('*.kv'))),before)

    def test_equal_shallow_selection_states_do_not_deduplicate_different_full_kv(self):
        store = self.store()
        exact = self.candidate('exact')
        mixed = self.candidate('mixed',mixed=True)
        # A controlled collision: all selection K matches, final-layer V does
        # not. Source identity must include the full artifact, not d1/d2 only.
        mixed['layers'][2][1].add_(1)
        mixed['artifact_digest'] = tensor_digest(t for pair in mixed['layers'] for t in pair)
        mixed['source_artifact_id'] = digest_json(dict(
            manifest=asdict(mixed['manifest_reference']),proof=asdict(mixed['proof']),
            artifact_digest=mixed['artifact_digest']))
        for depth in (1,2):
            self.assertTrue(torch.equal(exact['selection_states'][depth],mixed['selection_states'][depth]))
        first = self.publish(store,exact)
        second = self.publish(store,mixed)
        self.assertEqual(second['status'],'PUBLISHED')
        self.assertNotEqual(first['source_id'],second['source_id'])
        self.assertEqual(len(self.read_rows(store)),2)

    def test_lru_refresh_only_actual_leased_use_not_lookup_or_comparison(self):
        store = self.store(max_variants=2)
        first = self.publish(store,self.candidate('one'))['source_id']
        second = self.publish(store,self.candidate('two',base=20))['source_id']
        snapshot = store.begin_request('reader')
        before = {r['source_id']:r['last_request_use_epoch'] for r in store.lookup(snapshot,self.target_tokens)}
        store.read_selection(snapshot,second,1)
        self.assertEqual({r['source_id']:r['last_request_use_epoch'] for r in store.lookup(snapshot,self.target_tokens)},before)
        with self.assertRaises(ValueError): store.mark_request_use(snapshot,first)
        with store.leased_target(snapshot,first): store.mark_request_use(snapshot,first)
        store.end_request(snapshot)
        result = self.publish(store,self.candidate('three',base=40))
        self.assertEqual(result['evicted_source_id'],second)
        self.assertIn(first,{r['source_id'] for r in self.read_rows(store)})

    def test_bounded_probation_grace_then_ordinary_lru(self):
        store = self.store(max_variants=1,probation_opportunities=2)
        first = self.publish(store,self.candidate('one'))['source_id']
        candidate = self.candidate('two',base=20)
        snapshot = store.begin_request('two')
        store.record_lookup_opportunity(snapshot,self.target_tokens)
        store.record_lookup_opportunity(snapshot,self.target_tokens)
        with self.assertRaises(MemoryError): self.plan(store,snapshot,candidate)
        store.end_request(snapshot)
        reader = store.begin_request('second-opportunity')
        store.record_lookup_opportunity(reader,self.target_tokens)
        store.end_request(reader)
        result = self.publish(store,candidate)
        self.assertEqual(result['evicted_source_id'],first)

    def test_victim_protection_change_requires_replan_not_silent_substitution(self):
        store = self.store(max_variants=2)
        first = self.publish(store,self.candidate('one'))['source_id']
        second = self.publish(store,self.candidate('two',base=20))['source_id']
        candidate = self.candidate('three',base=40)
        snapshot = store.begin_request('three')
        plan = self.plan(store,snapshot,candidate)
        self.assertEqual(plan.victim_id,first)
        with store.leased_target(snapshot,first):
            result = store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
            self.assertEqual(result['status'],'REJECTED')
            self.assertEqual({r['source_id'] for r in store.lookup(snapshot,self.target_tokens)},{first,second})
            replacement = self.plan(store,snapshot,candidate)
            self.assertEqual(replacement.victim_id,second)
        # Releasing the lease also changes the protection generation.
        again = store.commit_publication(snapshot,replacement,candidate,token_ids=self.target_tokens)
        self.assertEqual(again['status'],'REJECTED')
        store.end_request(snapshot)

    def test_failed_staging_and_catalog_preserve_preselected_victim(self):
        for fail_method in ('_stage','_save_catalog'):
            with self.subTest(method=fail_method):
                store = self.store(fail_method,max_variants=1)
                first = self.publish(store,self.candidate(fail_method+'-one'))['source_id']
                candidate = self.candidate(fail_method+'-two',base=20)
                snapshot = store.begin_request(candidate['proof'].request_id)
                plan = self.plan(store,snapshot,candidate)
                with patch.object(store,fail_method,side_effect=OSError('simulated disk failure')):
                    result = store.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
                self.assertEqual(result['status'],'REJECTED')
                self.assertEqual([r['source_id'] for r in store.lookup(snapshot,self.target_tokens)],[first])
                self.assertEqual(len(list(store.root.glob('*.kv'))),1)
                store.end_request(snapshot)

    def test_recovery_preserves_placements_lru_grace_and_metadata(self):
        store = self.store(probation_opportunities=2)
        sid = self.publish(store,self.candidate('birth',mixed=True))['source_id']
        snapshot = store.begin_request('reader')
        store.record_lookup_opportunity(snapshot,self.target_tokens)
        with store.leased_target(snapshot,sid): store.mark_request_use(snapshot,sid)
        expected = store.lookup(snapshot,self.target_tokens)
        store.end_request(snapshot)
        store.close()
        recovered = self.store(probation_opportunities=2)
        self.assertEqual(self.read_rows(recovered),expected)

    def test_corrupt_file_or_missing_selection_fails_recovery(self):
        for corruption in ('kv','selection'):
            store = self.store(corruption)
            self.publish(store,self.candidate(corruption))
            row = self.read_rows(store)[0]
            path = store.root/row['kv_file' if corruption=='kv' else 'selection_file']
            store.close()
            if corruption=='kv': path.write_bytes(b'corrupt')
            else: path.unlink()
            with self.assertRaises((ValueError,RuntimeError,EOFError)):
                self.store(corruption)

    def test_bad_catalog_provenance_with_rehashed_envelope_rejected(self):
        store = self.store()
        sid = self.publish(store,self.candidate('birth',mixed=True))['source_id']
        store.close()
        path = store.root/'catalog.json'
        document = json.loads(path.read_text(encoding='utf-8'))
        document['payload']['rows'][sid]['generation']=0
        document['payload']['rows'][sid]['origin']=SourceOrigin.EXACT.value
        document['digest']=digest_json(document['payload'])
        path.write_text(json.dumps(document),encoding='utf-8')
        with self.assertRaises(ValueError): self.store()

    def test_child_remains_readable_after_parent_eviction_no_parent_lease(self):
        store = self.store(max_variants=2)
        parent_sid = self.publish(store,self.candidate('parent'))['source_id']
        snapshot = store.begin_request('inspect-parent')
        parent = store.parent_metadata(snapshot,parent_sid)
        store.end_request(snapshot)
        child_sid = self.publish(store,self.candidate('child',base=20,mixed=True,parent=parent))['source_id']
        result = self.publish(store,self.candidate('replacement',base=40))
        self.assertEqual(result['evicted_source_id'],parent_sid)
        snapshot = store.begin_request('consume-child')
        with store.leased_target(snapshot,child_sid) as layers:
            self.assertEqual(len(layers),3)
        self.assertEqual(store.parent_metadata(snapshot,child_sid).generation,1)
        store.end_request(snapshot)
        self.assertEqual(store.storage_audit()['parent_lease_count'],0)

    def test_missing_selection_depth_never_reads_full_kv_as_substitute(self):
        store = self.store()
        sid = self.publish(store,self.candidate('birth'))['source_id']
        snapshot = store.begin_request('reader')
        with patch('probekv.source_store_v2.LayerFile',side_effect=AssertionError('full KV read')):
            with self.assertRaises(KeyError): store.read_selection(snapshot,sid,9)
        store.end_request(snapshot)

    def test_exception_inside_read_releases_lease_and_resource_lifecycle(self):
        store = self.store()
        sid = self.publish(store,self.candidate('birth'))['source_id']
        snapshot = store.begin_request('reader')
        with self.assertRaises(RuntimeError):
            with store.leased_target(snapshot,sid):
                with self.assertRaises(RuntimeError): store.end_request(snapshot)
                with self.assertRaises(RuntimeError): store.close()
                raise RuntimeError('consumer failed')
        self.assertFalse(any(store._lease_counts.values()))
        store.end_request(snapshot)

    def test_layer_open_failure_does_not_leak_lease(self):
        store = self.store()
        sid = self.publish(store,self.candidate('birth'))['source_id']
        snapshot = store.begin_request('reader')
        with patch('probekv.source_store_v2.LayerFile',side_effect=ValueError('bad header')):
            with self.assertRaises(ValueError):
                with store.leased_target(snapshot,sid): pass
        self.assertFalse(any(store._lease_counts.values()))
        store.end_request(snapshot)

    def test_manifest_physical_storage_is_included_once_in_pool_budget(self):
        store = self.store()
        self.publish(store,self.candidate('birth'))
        self.assertEqual(store.storage_audit()['shared_manifest_bytes'],
                         self.registry.storage_audit()['total_persistent_bytes'])

    def test_validation_staging_and_global_persistent_budgets_fail_closed(self):
        tiny = self.store('tiny-staging',staging_bytes=1000)
        candidate = self.candidate('birth')
        snapshot = tiny.begin_request('birth')
        plan = self.plan(tiny,snapshot,candidate)
        result = tiny.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(tiny.storage_audit()['source_count'],0)
        tiny.end_request(snapshot)
        with self.assertRaises(MemoryError):
            self.store('below-manifest-cost',max_bytes=1000)
        scarce = self.store('scarce',
            max_bytes=self.registry.storage_audit()['total_persistent_bytes']+1000)
        snapshot = scarce.begin_request('birth')
        plan = self.plan(scarce,snapshot,candidate)
        result = scarce.commit_publication(snapshot,plan,candidate,token_ids=self.target_tokens)
        self.assertEqual(result['status'],'REJECTED')
        self.assertEqual(scarce.storage_audit()['source_count'],0)
        scarce.end_request(snapshot)

    def test_incomplete_scope_non_growth_and_foreign_birth_rejected(self):
        store = self.store()
        self.publish(store,self.candidate('one'))
        candidate = self.candidate('two',base=20)
        snapshot = store.begin_request('two')
        for scope in (PublicationScope('content_miss',0,0,0,0),
                      PublicationScope('complete_scope_mismatch',1,1,1,0,2,1),
                      PublicationScope('economic_rejection',1,1,1,1,2,1)):
            with self.assertRaises(ValueError):
                store.plan_publication(snapshot,candidate,token_ids=self.target_tokens,scope=scope)
        with self.assertRaises(ValueError): self.plan(store,snapshot,self.candidate('foreign',base=40))
        store.end_request(snapshot)


if __name__ == '__main__':
    unittest.main()
