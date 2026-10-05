"""CPU metadata tests only; no real-model/GPU execution is certified here."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from probekv.source_manifest_v2 import (ParentSourceMetadata, RequestManifestRegistry,
    TargetOccurrence, request_input_digest)
from probekv.source_provenance_v2 import KVImport, RequestExecutionLedger, SourceOrigin
from probekv.v8_schema10_execution import digest_json


class ManifestPersistenceTests(unittest.TestCase):
    def test_cached_validation_still_detects_disk_tamper(self):
        r=self.registry();key=self.register(r);self.reference(r,key)
        path=self.root/'manifests'/(key+'.json')
        original=path.read_bytes()
        path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'changed after'):
            self.reference(r,key)

    def test_reopen_revalidates_and_cached_access_checks_catalog(self):
        r=self.registry();key=self.register(r);self.reference(r,key)
        with patch.object(RequestManifestRegistry,'_validate_payload',autospec=True,
                          side_effect=RequestManifestRegistry._validate_payload) as validate:
            reopened=self.registry()
            self.assertGreater(validate.call_count,0)
            self.reference(reopened,key)
        path=self.root/'registry.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'catalog changed'):
            self.reference(r,key)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def registry(self, **kw):
        return RequestManifestRegistry(**dict(max_bytes=1000000,
            max_manifest_bytes=100000, persistent_root=self.root, **kw))

    def args(self, *, request='birth', mixed=False):
        tokens, positions = (10, 11, 12, 13), (0, 1, 2, 3)
        ledger = RequestExecutionLedger(request_id=request,
            input_digest=request_input_digest(tokens, positions), model_signature='model',
            token_count=4, num_layers=2, exact_input_proof='verified-runtime-embedding')
        for layer in (1, 2):
            rows = (1, 2, 3) if mixed else positions
            ledger.record_layer(layer_1based=layer, qkv_rows=rows,
                attention_rows=rows, output_mlp_rows=rows,
                imported_kv=(KVImport(0, 'parent', 0),) if mixed else (),
                completion_reference='cpu-test-layer-' + str(layer))
        parents = (ParentSourceMetadata('parent', 'a'*64, 'model', 'domain',
                                        SourceOrigin.EXACT, 0),) if mixed else ()
        return dict(ledger=ledger, token_ids=tokens, absolute_positions=positions,
            authorization_domain='domain', occurrences=(TargetOccurrence('target', 2, 4),),
            parent_sources=parents, request_completed=True, input_reference='controlled:'+request)

    def register(self, registry, **kw):
        return registry.register_actual_execution(**self.args(**kw))

    def export(self, registry, key):
        return registry.validate_manifest(key, model_signature='model', authorization_domain='domain')

    def reference(self, registry, key):
        return registry.capture_reference(key, 'target', model_signature='model',
            authorization_domain='domain', target_token_ids=(12, 13), target_positions=(2, 3))

    def overwrite_rehashed_payload(self, key, mutate):
        """Even self-consistent file/catalog hashes cannot certify fake ranks."""
        original = self.root/'manifests'/(key+'.json')
        payload = json.loads(original.read_bytes())
        mutate(payload)
        new_key = 'request-manifest-v2-' + digest_json(payload)
        (self.root/'manifests'/(new_key+'.json')).write_bytes(RequestManifestRegistry._encode(payload))
        envelope = json.loads((self.root/'registry.json').read_bytes())
        envelope['payload']['manifest_ids'] = [new_key]
        envelope['digest'] = digest_json(envelope['payload'])
        (self.root/'registry.json').write_bytes(RequestManifestRegistry._encode(envelope))
        return new_key

    def test_restore_replays_exact_and_mixed_and_keeps_one_file_per_birth(self):
        registry = self.registry()
        exact = self.register(registry)
        mixed = self.register(registry, request='mixed', mixed=True)
        self.assertEqual(self.register(registry), exact)
        self.assertEqual(len(list((self.root/'manifests').glob('*.json'))), 2)
        restored = self.registry()
        self.assertEqual(self.reference(registry, exact), self.reference(restored, exact))
        self.assertEqual(self.export(restored, mixed)['occurrences'][0]['proof']['generation'], 1)
        self.assertEqual(restored.storage_audit()['parent_owned_KV_bytes'], 0)
        self.assertTrue(restored.durable)

    def test_ten_target_references_do_not_duplicate_persistent_request(self):
        registry = self.registry()
        args = self.args()
        args['occurrences'] = tuple(TargetOccurrence(str(i), i, i+1) for i in range(4))
        key = registry.register_actual_execution(**args)
        before = registry.storage_audit()
        for _ in range(10):
            for i in range(4):
                registry.capture_reference(key, str(i), model_signature='model',
                    authorization_domain='domain', target_token_ids=(10+i,), target_positions=(i,))
        self.assertEqual(before, registry.storage_audit())
        self.assertEqual(len(list((self.root/'manifests').glob('*.json'))), 1)

    def test_parent_metadata_survives_without_parent_files_or_kv(self):
        registry = self.registry()
        key = self.register(registry, mixed=True)
        del registry
        restored = self.registry()
        payload = self.export(restored, key)
        self.assertEqual(payload['parent_sources'][0]['artifact_digest'], 'a'*64)
        self.assertFalse(payload['parent_references_own_KV'])
        self.assertEqual(restored.storage_audit()['parent_owned_lease_count'], 0)

    def test_authorization_revocation_is_persistent(self):
        registry = self.registry()
        key = self.register(registry)
        registry.revoke_authorization_domain('domain')
        restored = self.registry()
        with self.assertRaisesRegex(ValueError, 'revoked'):
            self.reference(restored, key)
        with self.assertRaisesRegex(ValueError, 'revoked'):
            self.register(restored, request='new')
        self.assertEqual(restored.storage_audit()['manifest_count'], 1)

    def test_revocation_write_failure_is_not_acknowledged(self):
        registry = self.registry()
        key = self.register(registry)
        with patch.object(registry, '_atomic_write', side_effect=OSError('disk failed')):
            with self.assertRaises(OSError):
                registry.revoke_authorization_domain('domain')
        self.reference(registry, key)
        self.reference(self.registry(), key)

    def test_stale_registry_cannot_ignore_or_overwrite_new_revocation(self):
        first = self.registry()
        key = self.register(first)
        second = self.registry()
        first.revoke_authorization_domain('domain')
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            self.reference(second, key)
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            self.register(second, request='new')
        with self.assertRaisesRegex(ValueError, 'revoked'):
            self.reference(self.registry(), key)

    def test_catalog_commit_failure_keeps_payload_invisible_and_retryable(self):
        registry = self.registry()
        with patch.object(registry, '_write_catalog', side_effect=OSError('catalog failed')):
            with self.assertRaises(OSError):
                self.register(registry)
        self.assertEqual(registry.storage_audit()['manifest_count'], 0)
        self.assertGreater(registry.storage_audit()['unpublished_orphan_bytes'], 0)
        restored = self.registry()
        self.assertEqual(restored.storage_audit()['manifest_count'], 0)
        key = self.register(restored)
        self.reference(self.registry(), key)
        self.assertEqual(restored.storage_audit()['unpublished_orphan_bytes'], 0)

    def test_payload_write_failure_has_no_visible_half_registration(self):
        registry = self.registry()
        with patch.object(registry, '_atomic_write', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                self.register(registry)
        self.assertEqual(self.registry().storage_audit()['manifest_count'], 0)

    def test_registration_replays_ledger_before_first_disk_publication(self):
        registry = self.registry()
        arguments = self.args(mixed=True)
        ledger = arguments['ledger']
        ledger._layers[1] = replace(ledger._layers[1], output_ranks=(0,0,0,0))
        with self.assertRaisesRegex(ValueError, 'recomputed ledger'):
            registry.register_actual_execution(**arguments)
        self.assertEqual(registry.storage_audit()['manifest_count'], 0)
        self.assertEqual(list((self.root/'manifests').iterdir()), [])

    def test_missing_catalog_cannot_infer_previous_permission_or_visibility(self):
        registry = self.registry()
        self.register(registry)
        (self.root/'registry.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing catalog'):
            self.registry()

    def test_corrupt_file_fails_live_read_and_restore(self):
        registry = self.registry()
        key = self.register(registry)
        (self.root/'manifests'/(key+'.json')).write_bytes(b'{"truncated":')
        with self.assertRaises(ValueError):
            self.export(registry, key)
        with self.assertRaises(ValueError):
            self.registry()

    def test_catalog_bad_digest_and_manifest_path_escape_rejected(self):
        registry = self.registry()
        self.register(registry)
        path = self.root/'registry.json'
        envelope = json.loads(path.read_bytes())
        envelope['digest'] = '0'*64
        path.write_bytes(registry._encode(envelope))
        with self.assertRaises(ValueError):
            self.registry()
        envelope['payload']['manifest_ids'] = ['../../outside']
        envelope['digest'] = digest_json(envelope['payload'])
        path.write_bytes(registry._encode(envelope))
        with self.assertRaises(ValueError):
            self.registry()

    def test_rehashed_target_proof_cannot_fake_exact_generation(self):
        registry = self.registry()
        key = self.register(registry, mixed=True)
        def mutate(payload):
            proof = payload['occurrences'][0]['proof']
            proof.update(origin=SourceOrigin.EXACT.value, generation=0)
            payload['occurrences'][0]['proof_digest'] = digest_json(proof)
        self.overwrite_rehashed_payload(key, mutate)
        with self.assertRaisesRegex(ValueError, 'reconstructed execution'):
            self.registry()

    def test_rehashed_rank_or_pass_flag_cannot_replace_layer_reconstruction(self):
        for field in ('output_ranks', 'kv_ranks'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                registry = self.registry()
                key = self.register(registry, mixed=True)
                self.overwrite_rehashed_payload(key,
                    lambda payload: payload['executed_layers'][1].update({field: [0,0,0,0]}))
                with self.assertRaisesRegex(ValueError, 'recomputed ledger'):
                    self.registry()

    def test_rehashed_parent_model_generation_and_unknown_fields_rejected(self):
        for mutation in (
                lambda p: p['parent_sources'][0].update(model_signature='wrong'),
                lambda p: p['parent_sources'][0].update(generation=1, origin=SourceOrigin.MIXED.value),
                lambda p: p.update(native_runtime_qualified=True),
                lambda p: p.update(passed=True)):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                registry = self.registry()
                key = self.register(registry, mixed=True)
                self.overwrite_rehashed_payload(key, mutation)
                with self.assertRaises(ValueError):
                    self.registry()

    def test_live_memory_validation_also_rebuilds_ranks(self):
        registry = RequestManifestRegistry(max_bytes=1000000, max_manifest_bytes=100000)
        key = self.register(registry, mixed=True)
        payload = self.export(registry, key)
        payload['executed_layers'][1]['output_ranks'] = [0,0,0,0]
        forged = 'request-manifest-v2-' + digest_json(payload)
        registry._payloads[forged] = registry._encode(payload)
        with self.assertRaisesRegex(ValueError, 'recomputed ledger'):
            self.export(registry, forged)

    def test_restore_and_lookup_check_model_domain_and_byte_limits(self):
        registry = self.registry()
        key = self.register(registry)
        restored = self.registry()
        for fields in (dict(model_signature='other', authorization_domain='domain'),
                       dict(model_signature='model', authorization_domain='other')):
            with self.assertRaises(ValueError):
                restored.validate_manifest(key, **fields)
        with self.assertRaises(MemoryError):
            RequestManifestRegistry(max_bytes=1000000, max_manifest_bytes=10, persistent_root=self.root)

    def test_duplicate_json_field_not_silently_accepted(self):
        registry = self.registry()
        self.register(registry)
        path = self.root/'registry.json'
        raw = path.read_bytes()
        path.write_bytes(b'{"digest":"ignored",'+raw[1:])
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.registry()


if __name__ == '__main__':
    unittest.main()
