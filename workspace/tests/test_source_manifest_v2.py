import gc
import json
import unittest
import weakref
from unittest.mock import patch
from dataclasses import replace

from probekv.source_manifest_v2 import (ParentSourceMetadata, RequestManifestRegistry,
    TargetOccurrence, request_input_digest)
from probekv.source_provenance_v2 import KVImport, RequestExecutionLedger, SourceOrigin


class SharedManifestTests(unittest.TestCase):
    def test_repeated_reference_reuses_only_unchanged_byte_validation(self):
        r=self.registry();key,_=self.register(r)
        with patch.object(r,'_validate_payload',wraps=r._validate_payload) as validate:
            first=self.reference(r,key)
            self.assertEqual(self.reference(r,key),first)
            self.assertEqual(validate.call_count,1)
            detached=r.export_manifest(key,model_signature='model@revision',authorization_domain='domain')
            detached['token_ids'][0]=999
            self.assertEqual(self.reference(r,key),first)
            r._payloads[key]=r._encode(detached)
            with self.assertRaises(ValueError):self.reference(r,key)
            self.assertEqual(validate.call_count,2)

    def test_cached_reference_never_bypasses_revocation_or_model(self):
        r=self.registry();key,_=self.register(r);self.reference(r,key)
        with self.assertRaises(ValueError):
            r.export_manifest(key,model_signature='another-model',authorization_domain='domain')
        r.revoke_authorization_domain('domain')
        with self.assertRaises(ValueError):self.reference(r,key)

    def registry(self, budget=1000000):
        return RequestManifestRegistry(max_bytes=budget, max_manifest_bytes=budget)

    def ledger(self, *, request_id='birth', mixed=False, unknown=False, generation=0):
        tokens = (10, 11, 12, 13)
        positions = (0, 1, 2, 3)
        ledger = RequestExecutionLedger(request_id=request_id,
            input_digest=request_input_digest(tokens, positions), model_signature='model@revision',
            token_count=4, num_layers=2, exact_input_proof=None if unknown else 'verified-native-input')
        for layer in (1, 2):
            rows = (1, 2, 3) if mixed else (0, 1, 2, 3)
            ledger.record_layer(layer_1based=layer, qkv_rows=rows, attention_rows=rows,
                output_mlp_rows=rows, imported_kv=(KVImport(0, 'parent', generation),) if mixed else (),
                completion_reference='completed-native-event-' + str(layer))
        return ledger, tokens, positions

    def parent(self, generation=0):
        return ParentSourceMetadata('parent', 'a'*64, 'model@revision', 'domain',
            SourceOrigin.EXACT if generation == 0 else SourceOrigin.MIXED, generation)

    def register(self, registry, *, mixed=False, request_id='birth'):
        ledger, tokens, positions = self.ledger(mixed=mixed, request_id=request_id)
        key = registry.register_actual_execution(ledger=ledger, token_ids=tokens,
            absolute_positions=positions, authorization_domain='domain',
            occurrences=(TargetOccurrence('target', 2, 4),),
            parent_sources=(self.parent(),) if mixed else (), request_completed=True,
            input_reference='controlled-input:birth')
        return key, ledger

    def reference(self, registry, key):
        return registry.capture_reference(key, 'target', model_signature='model@revision',
            authorization_domain='domain', target_token_ids=(12,13), target_positions=(2,3))

    def test_exact_and_mixed_identity_differ_without_changing_tokens(self):
        r = self.registry()
        exact, _ = self.register(r, request_id='exact')
        mixed, _ = self.register(r, mixed=True, request_id='mixed')
        self.assertNotEqual(exact, mixed)
        self.reference(r, exact); self.reference(r, mixed)
        payload = r.export_manifest(mixed, model_signature='model@revision', authorization_domain='domain')
        self.assertEqual(payload['occurrences'][0]['proof']['generation'], 1)
        self.assertEqual(payload['occurrences'][0]['proof']['origin'], SourceOrigin.MIXED.value)

    def test_same_birth_is_idempotent_but_execution_change_rejected(self):
        r = self.registry()
        one, _ = self.register(r)
        size = r.storage_audit()['shared_manifest_json_bytes']
        two, _ = self.register(r)
        self.assertEqual(one, two)
        self.assertEqual(r.storage_audit()['shared_manifest_json_bytes'], size)
        with self.assertRaises(ValueError):
            self.register(r, mixed=True)
        self.assertEqual(r.storage_audit()['manifest_count'], 1)

    def test_references_are_lightweight_and_do_not_hold_parents_or_ledger(self):
        r = self.registry()
        key, ledger = self.register(r, mixed=True)
        weak = weakref.ref(ledger)
        refs = [self.reference(r, key) for _ in range(10)]
        del ledger
        gc.collect()
        self.assertIsNone(weak())
        self.assertTrue(all(ref.manifest_id == key for ref in refs))
        self.assertEqual(r.storage_audit()['manifest_count'], 1)
        self.assertEqual(r.storage_audit()['parent_owned_KV_bytes'], 0)
        self.assertFalse(r.durable)

    def test_ten_distinct_segments_share_one_birth_manifest(self):
        r = self.registry()
        tokens, positions = tuple(range(100,110)), tuple(range(10))
        ledger = RequestExecutionLedger(request_id='ten-segments',
            input_digest=request_input_digest(tokens,positions),model_signature='model@revision',
            token_count=10,num_layers=1,exact_input_proof='input')
        ledger.record_layer(layer_1based=1,qkv_rows=positions,attention_rows=positions,
                            output_mlp_rows=positions,completion_reference='done')
        key = r.register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
            authorization_domain='domain',occurrences=tuple(TargetOccurrence(str(i),i,i+1) for i in range(10)),
            request_completed=True,input_reference='controlled:ten')
        before = r.storage_audit()
        refs = [r.capture_reference(key,str(i),model_signature='model@revision',authorization_domain='domain',
                target_token_ids=(tokens[i],),target_positions=(i,)) for i in range(10)]
        self.assertEqual({ref.manifest_id for ref in refs},{key})
        self.assertEqual([ref.prefix_end for ref in refs],list(range(10)))
        self.assertEqual(r.storage_audit(),before)

    def test_unknown_and_tampered_reference_rejected(self):
        r = self.registry()
        key, _ = self.register(r)
        ref = self.reference(r, key)
        for field, value in (('manifest_digest','b'*64), ('prefix_end',1),
                             ('authorization_domain','other'), ('target_occurrence','missing')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_capture_reference(replace(ref, **{field:value}), model_signature='model@revision',
                    authorization_domain='domain', target_token_ids=(12,13), target_positions=(2,3))
        with self.assertRaises(ValueError):
            self.reference(r, 'unknown')

    def test_actual_token_positions_model_and_permission_checked(self):
        r = self.registry()
        key, _ = self.register(r)
        good = dict(model_signature='model@revision', authorization_domain='domain',
                    target_token_ids=(12,13), target_positions=(2,3))
        for field, value in (('model_signature','other'), ('authorization_domain','other'),
                            ('target_token_ids',(11,12)), ('target_positions',(1,2))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.capture_reference(key, 'target', **dict(good, **{field:value}))
        r.revoke_authorization_domain('domain')
        with self.assertRaises(ValueError):
            self.reference(r, key)

    def test_modified_return_value_cannot_mutate_registry_and_internal_tamper_fails(self):
        r = self.registry()
        key, _ = self.register(r)
        payload = r.export_manifest(key, model_signature='model@revision', authorization_domain='domain')
        payload['token_ids'][2] = 99
        self.reference(r, key)
        r._payloads[key] = json.dumps(payload).encode()
        with self.assertRaises(ValueError):
            self.reference(r, key)

    def test_no_capacity_partial_publication(self):
        r = self.registry(budget=10)
        with self.assertRaises(MemoryError):
            self.register(r)
        self.assertEqual(r.storage_audit()['manifest_count'], 0)
        self.assertEqual(r._by_birth, {})

    def test_missing_unknown_and_inconsistent_parent_rejected(self):
        for parents in ((), (replace(self.parent(), generation=1, origin=SourceOrigin.MIXED),),
                        (replace(self.parent(), authorization_domain='other'),), (object(),)):
            r = self.registry()
            ledger, tokens, positions = self.ledger(mixed=True)
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                r.register_actual_execution(ledger=ledger, token_ids=tokens, absolute_positions=positions,
                    authorization_domain='domain', occurrences=(TargetOccurrence('target',2,4),),
                    parent_sources=parents, request_completed=True, input_reference='input')
        for generation in (None, True, -1, 2):
            with self.assertRaises(ValueError):
                self.parent(generation)

    def test_unverified_input_unknown_rank_and_incomplete_request_rejected(self):
        r = self.registry()
        for unknown in (True, False):
            ledger, tokens, positions = self.ledger(unknown=unknown)
            with self.subTest(unknown=unknown), self.assertRaises(ValueError):
                r.register_actual_execution(ledger=ledger, token_ids=tokens, absolute_positions=positions,
                    authorization_domain='domain', occurrences=(TargetOccurrence('target',2,4),),
                    request_completed=unknown, input_reference='input')
        ledger, tokens, positions = self.ledger()
        with self.assertRaises(ValueError):
            r.register_actual_execution(ledger=ledger, token_ids=(10,11,12,99), absolute_positions=positions,
                authorization_domain='domain', occurrences=(TargetOccurrence('target',2,4),),
                request_completed=True, input_reference='input')

    def test_occurrence_ranges_and_partial_target_cannot_fake_full_capture(self):
        r = self.registry()
        ledger, tokens, positions = self.ledger(mixed=True)
        key = r.register_actual_execution(ledger=ledger, token_ids=tokens, absolute_positions=positions,
            authorization_domain='domain', occurrences=(TargetOccurrence('partial',0,1),TargetOccurrence('full',1,4)),
            parent_sources=(self.parent(),), request_completed=True, input_reference='input')
        with self.assertRaises(ValueError):
            r.capture_reference(key,'partial',model_signature='model@revision',authorization_domain='domain',
                                target_token_ids=(10,),target_positions=(0,))
        with self.assertRaises(ValueError):
            r.register_actual_execution(ledger=ledger, token_ids=tokens, absolute_positions=positions,
                authorization_domain='domain', occurrences=(TargetOccurrence('a',0,3),TargetOccurrence('b',2,4)),
                parent_sources=(self.parent(),), request_completed=True,input_reference='input')

    def test_g2_can_be_logged_but_not_given_main_capture_reference(self):
        r = self.registry()
        ledger,tokens,positions = self.ledger(mixed=True,generation=1)
        key = r.register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
            authorization_domain='domain',occurrences=(TargetOccurrence('target',2,4),),
            parent_sources=(self.parent(1),),request_completed=True,input_reference='input')
        with self.assertRaises(ValueError):
            self.reference(r,key)

    def test_exact_prefix_string_is_not_a_qualified_proof(self):
        ledger,tokens,positions = self.ledger()
        # Rebuild real event sequence with an arbitrary "proof" reference.
        ledger = RequestExecutionLedger(request_id='prefix', input_digest=request_input_digest(tokens,positions),
            model_signature='model@revision',token_count=4,num_layers=1,exact_input_proof='input')
        ledger.record_layer(layer_1based=1,qkv_rows=(1,2,3),attention_rows=(1,2,3),output_mlp_rows=(1,2,3),
            imported_kv=(KVImport(0,'parent',0,'arbitrary-string'),),completion_reference='done')
        with self.assertRaises(ValueError):
            self.registry().register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
                authorization_domain='domain',occurrences=(TargetOccurrence('target',2,4),),
                parent_sources=(self.parent(),),request_completed=True,input_reference='input')

    def test_unknown_later_suffix_does_not_taint_earlier_target(self):
        r = self.registry()
        tokens, positions = (10,11,12,13), (0,1,2,3)
        ledger = RequestExecutionLedger(request_id='suffix', input_digest=request_input_digest(tokens,positions),
            model_signature='model@revision',token_count=4,num_layers=1,exact_input_proof='input')
        ledger.record_layer(layer_1based=1,qkv_rows=(0,1,2),attention_rows=(0,1,2),output_mlp_rows=(0,1,2),
            imported_kv=(KVImport(3,'unknown-suffix-parent',None),),completion_reference='done')
        key = r.register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
            authorization_domain='domain',occurrences=(TargetOccurrence('early',0,2),),
            request_completed=True,input_reference='input')
        ref = r.capture_reference(key,'early',model_signature='model@revision',authorization_domain='domain',
            target_token_ids=(10,11),target_positions=(0,1))
        self.assertEqual(ref.prefix_end,0)
        payload=r.export_manifest(key,model_signature='model@revision',authorization_domain='domain')
        self.assertEqual(payload['unverified_noncausal_parent_ids'],['unknown-suffix-parent'])
        with self.assertRaises(ValueError):
            r.register_actual_execution(ledger=ledger,token_ids=tokens,absolute_positions=positions,
                authorization_domain='domain',occurrences=(TargetOccurrence('all',0,4),),
                request_completed=True,input_reference='input')


if __name__ == '__main__':
    unittest.main()
