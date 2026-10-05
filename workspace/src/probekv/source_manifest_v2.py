"""Bounded shared execution manifests for v2 Source candidates.

The registry owns canonical JSON bytes once per completed birth request. Sources
receive only the existing lightweight ManifestReference; no tensor, parent
Artifact, lease, callback or live execution ledger is retained. Optional local
persistence uses an atomic catalog, not caller-supplied ``passed`` flags. This
is not an authorization service or real-model qualification evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from dataclasses import asdict, dataclass

from .segment_capture_v2 import ManifestReference
from .source_provenance_v2 import KVImport, RequestExecutionLedger, SourceOrigin
from .v8_schema10_execution import digest_json


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer within range')
    return value


def _digest(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(ch not in '0123456789abcdef' for ch in value)):
        raise ValueError(name + ' must be a lowercase SHA256 digest')
    return value


def _identity(token_ids, absolute_positions):
    if not isinstance(token_ids, (tuple, list)) or not isinstance(absolute_positions, (tuple, list)):
        raise ValueError('explicit integer token/position sequences required')
    tokens, positions = tuple(token_ids), tuple(absolute_positions)
    if not tokens or len(tokens) != len(positions):
        raise ValueError('nonempty aligned tokens and positions required')
    for token in tokens:
        _integer(token, 'token id')
    for position in positions:
        _integer(position, 'absolute position')
    if positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ValueError('contiguous original absolute positions required')
    return tokens, positions


def request_input_digest(token_ids, absolute_positions):
    """Identity used by both the native ledger hook and this registry."""
    tokens, positions = _identity(token_ids, absolute_positions)
    return digest_json(dict(token_ids=tokens, absolute_positions=positions))


@dataclass(frozen=True)
class ParentSourceMetadata:
    """Non-owning, previously verified parent identity/rank record.

    A string reference is not itself numerical evidence. The runtime publisher
    must supply these records from its validated Source metadata, never infer G
    from a Source ID or retrieve/pin parent KV to validate a child manifest.
    """
    source_id: str
    artifact_digest: str
    model_signature: str
    authorization_domain: str
    origin: SourceOrigin
    generation: int

    def __post_init__(self):
        for name in ('source_id', 'model_signature', 'authorization_domain'):
            _text(getattr(self, name), name)
        _digest(self.artifact_digest, 'parent artifact')
        _integer(self.generation, 'parent generation')
        if self.generation > 1:
            raise ValueError('main registry only references permitted G0/G1 Source parents')
        if (not isinstance(self.origin, SourceOrigin) or
                self.origin not in (SourceOrigin.EXACT, SourceOrigin.MIXED) or
                (self.origin == SourceOrigin.EXACT) != (self.generation == 0)):
            raise ValueError('unknown, partial or inconsistent parent provenance')


@dataclass(frozen=True)
class TargetOccurrence:
    """A canonical occurrence in request row indices, end exclusive."""
    occurrence_id: str
    start: int
    end: int

    def __post_init__(self):
        _text(self.occurrence_id, 'target occurrence')
        _integer(self.start, 'target start')
        _integer(self.end, 'target end', 1)
        if self.end <= self.start:
            raise ValueError('nonempty target range required')


class RequestManifestRegistry:
    """Single-process canonical registry with optional local JSON persistence.

    Registration requires a complete typed execution ledger and immutable
    original input. It does not infer GPU success from caller booleans. Native
    completion events still require separate GPU qualification.
    """
    durable = False

    def __init__(self, *, max_bytes, max_manifest_bytes, persistent_root=None):
        _integer(max_bytes, 'registry byte limit', 1)
        _integer(max_manifest_bytes, 'per-manifest byte limit', 1)
        if max_manifest_bytes > max_bytes:
            raise ValueError('per-manifest limit exceeds registry limit')
        self._max_bytes, self._max_manifest_bytes = max_bytes, max_manifest_bytes
        self._payloads = {}
        self._by_birth = {}
        self._bytes = 0
        self._revoked_domains = set()
        self._root = None
        self._catalog_bytes = 0
        self._catalog_snapshot = None
        self._orphan_bytes = 0
        # One bounded, non-owning reference to already stored immutable bytes.
        # Do not cache mutable parsed payloads, permissions, or disk validity.
        self._last_validated_manifest = None
        if persistent_root is not None:
            root = Path(persistent_root)
            if root.is_symlink():
                raise ValueError('manifest persistence root must not be a symlink')
            root.mkdir(parents=True, exist_ok=True)
            self._root = root.resolve()
            manifest_dir = self._root / 'manifests'
            if manifest_dir.is_symlink():
                raise ValueError('manifest directory must not be a symlink')
            manifest_dir.mkdir(exist_ok=True)
            self._restore()
        self.durable = self._root is not None

    def _authorize(self, domain):
        _text(domain, 'authorization domain')
        if domain in self._revoked_domains:
            raise ValueError('authorization domain revoked')

    def revoke_authorization_domain(self, domain):
        _text(domain, 'authorization domain')
        if domain in self._revoked_domains:
            return
        # Commit the deny-list before exposing revocation in memory. A failed
        # disk write is reported to the caller, never silently acknowledged.
        if self._root is not None:
            self._write_catalog(self._payloads, self._revoked_domains | {domain})
        self._revoked_domains.add(domain)

    @staticmethod
    def _encode(payload):
        return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')

    @staticmethod
    def _parse(encoded):
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate JSON field')
                result[key] = value
            return result
        def reject_constant(value):
            raise ValueError('nonfinite JSON number: ' + value)
        try:
            return json.loads(encoded, object_pairs_hook=unique_pairs,
                              parse_constant=reject_constant)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise ValueError('corrupt shared request manifest JSON') from exc

    def _atomic_write(self, path, encoded):
        if path.is_symlink():
            raise ValueError('refusing symlink in manifest persistence')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.manifest-', suffix='.tmp',
                    dir=str(path.parent), delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(path))
            temporary = None
            # POSIX also needs the directory entry durable. Windows does not
            # expose directory fsync through this API; atomic replace remains.
            if os.name != 'nt':
                descriptor = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary is not None and temporary.exists():
                # Only our newly-created incomplete temporary, never evidence.
                temporary.unlink()

    def _catalog(self, manifests, revoked):
        payload = dict(kind='source_request_manifest_catalog_v2',
                       manifest_ids=sorted(manifests), revoked_domains=sorted(revoked))
        return self._encode(dict(payload=payload, digest=digest_json(payload)))

    def _write_catalog(self, manifests, revoked):
        self._check_catalog_current()
        encoded = self._catalog(manifests, revoked)
        if len(encoded) > self._max_bytes:
            raise MemoryError('manifest catalog/revocation budget exceeded')
        self._atomic_write(self._root / 'registry.json', encoded)
        self._catalog_bytes = len(encoded)
        self._catalog_snapshot = encoded

    def _check_catalog_current(self):
        if self._root is not None and self._catalog_snapshot is not None:
            actual = self._read_bounded(self._root / 'registry.json', self._max_bytes)
            if actual != self._catalog_snapshot:
                raise ValueError('manifest catalog changed; reopen registry before access or publication')

    def _manifest_path(self, manifest_id):
        prefix = 'request-manifest-v2-'
        if type(manifest_id) is not str or not manifest_id.startswith(prefix):
            raise ValueError('invalid content-addressed manifest id')
        _digest(manifest_id[len(prefix):], 'manifest id')
        return self._root / 'manifests' / (manifest_id + '.json')

    def _read_bounded(self, path, limit):
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or unsafe manifest persistence file')
        if path.stat().st_size > limit:
            raise MemoryError('manifest persistence byte limit exceeded')
        with path.open('rb') as stream:
            encoded = stream.read(limit + 1)
        if len(encoded) > limit:
            raise MemoryError('manifest grew beyond byte limit')
        return encoded

    def _restore(self):
        catalog_path = self._root / 'registry.json'
        if not catalog_path.exists():
            if any((self._root / 'manifests').iterdir()):
                raise ValueError('missing catalog; refusing to infer publication or authorization')
            self._write_catalog({}, set())
            return
        encoded = self._read_bounded(catalog_path, self._max_bytes)
        envelope = self._parse(encoded)
        if (type(envelope) is not dict or set(envelope) != {'payload', 'digest'} or
                type(envelope['payload']) is not dict or
                envelope['digest'] != digest_json(envelope['payload'])):
            raise ValueError('manifest catalog digest/schema mismatch')
        catalog = envelope['payload']
        if (set(catalog) != {'kind', 'manifest_ids', 'revoked_domains'} or
                catalog['kind'] != 'source_request_manifest_catalog_v2' or
                type(catalog['manifest_ids']) is not list or
                type(catalog['revoked_domains']) is not list):
            raise ValueError('invalid manifest catalog schema')
        for domain in catalog['revoked_domains']:
            _text(domain, 'revoked authorization domain')
        for manifest_id in catalog['manifest_ids']:
            self._manifest_path(manifest_id)
        if (catalog['manifest_ids'] != sorted(set(catalog['manifest_ids'])) or
                catalog['revoked_domains'] != sorted(set(catalog['revoked_domains']))):
            raise ValueError('duplicate or unordered manifest catalog entries')
        payloads, by_birth, total = {}, {}, 0
        for manifest_id in catalog['manifest_ids']:
            body = self._read_bounded(self._manifest_path(manifest_id), self._max_manifest_bytes)
            payload = self._parse(body)
            self._validate_payload(manifest_id, payload)
            if payload['birth_request_id'] in by_birth:
                raise ValueError('multiple persisted executions for one birth request')
            total += len(body)
            if total > self._max_bytes:
                raise MemoryError('persisted manifest registry exceeds budget')
            payloads[manifest_id] = body
            by_birth[payload['birth_request_id']] = manifest_id
        # A payload written before a failed catalog commit is not published.
        # Preserve it, charge its bytes, and do not invent its visibility.
        known = {mid + '.json' for mid in payloads}
        orphan_bytes = 0
        for path in (self._root / 'manifests').iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError('unsafe entry in manifest directory')
            if path.name not in known:
                orphan_bytes += path.stat().st_size
        if total + orphan_bytes > self._max_bytes:
            raise MemoryError('manifest/orphan storage exceeds byte budget')
        self._payloads, self._by_birth, self._bytes = payloads, by_birth, total
        self._orphan_bytes, self._catalog_bytes = orphan_bytes, len(encoded)
        self._catalog_snapshot = encoded
        self._revoked_domains = set(catalog['revoked_domains'])

    def _validate_payload(self, manifest_id, payload):
        """Recompute ranks and TargetProof from primitive execution records.

        This checks integrity and internal consistency, not whether a GPU
        actually executed the claimed operations. Runtime qualification is
        separately required; no stored boolean can certify that fact.
        """
        try:
            if type(payload) is not dict or 'request-manifest-v2-' + digest_json(payload) != manifest_id:
                raise ValueError('shared request manifest digest mismatch')
            tokens, positions = _identity(payload['token_ids'], payload['absolute_positions'])
            ledger = RequestExecutionLedger(request_id=payload['birth_request_id'],
                input_digest=payload['input_digest'], model_signature=payload['model_signature'],
                token_count=len(tokens), num_layers=payload['num_layers'],
                exact_input_proof=payload['exact_input_proof_reference'])
            if type(payload['executed_layers']) is not list:
                raise ValueError('execution layer records required')
            for layer in payload['executed_layers']:
                imports = tuple(KVImport(**row) for row in layer['imported_kv'])
                rebuilt = ledger.record_layer(layer_1based=layer['layer_1based'],
                    qkv_rows=layer['qkv_rows'], effective_current_kv_rows=layer['current_kv_rows'],
                    attention_rows=layer['attention_rows'], output_mlp_rows=layer['output_mlp_rows'],
                    imported_kv=imports, completion_reference=layer['completion_reference'])
                if digest_json(asdict(rebuilt)) != digest_json(layer):
                    raise ValueError('persisted execution ranks differ from recomputed ledger')
            occurrences = tuple(TargetOccurrence(row['occurrence_id'], row['start'], row['end'])
                                for row in payload['occurrences'])
            parents = tuple(ParentSourceMetadata(**dict(row, origin=SourceOrigin(row['origin'])))
                            for row in payload['parent_sources'])
            # Reuse the registration contract without persistence or recursion:
            # first registration does not load anything from this temporary.
            verifier = RequestManifestRegistry(max_bytes=self._max_bytes,
                                               max_manifest_bytes=self._max_manifest_bytes)
            rebuilt_id = verifier.register_actual_execution(ledger=ledger, token_ids=tokens,
                absolute_positions=positions, authorization_domain=payload['authorization_domain'],
                occurrences=occurrences, parent_sources=parents,
                request_completed=payload['request_completed'], input_reference=payload['input_reference'])
            expected = self._parse(verifier._payloads[rebuilt_id])
            if payload['persistence'] not in ('in_memory_only', 'atomic_json'):
                raise ValueError('unknown manifest persistence format')
            expected['persistence'] = payload['persistence']
            if digest_json(expected) != digest_json(payload):
                raise ValueError('manifest recipe/proof differs from reconstructed execution')
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError('malformed execution manifest') from exc
        return payload

    def register_actual_execution(self, *, ledger, token_ids, absolute_positions,
                                  authorization_domain, occurrences, parent_sources=(),
                                  request_completed=False, input_reference):
        """Freeze actual recipe once; no intent or partially executed manifest.

        Explicit exact-Prefix imports currently fail closed. Verifying a Prefix
        proof requires a future native proof adapter, not a nonempty string.
        """
        if not isinstance(ledger, RequestExecutionLedger) or request_completed is not True:
            raise ValueError('completed typed request execution ledger required')
        self._authorize(authorization_domain)
        self._check_catalog_current()
        _text(input_reference, 'controlled input reference')
        tokens, positions = _identity(token_ids, absolute_positions)
        ledger._check_identity()
        if (len(tokens) != ledger.token_count or len(ledger.layers) != ledger.num_layers or
                request_input_digest(tokens, positions) != ledger.input_digest):
            raise ValueError('incomplete ledger or original input identity mismatch')
        # A typed object alone is not a proof: callers could have restored or
        # accidentally modified its private event list. Recompute before the
        # first disk/catalog write as well as when reading persisted records.
        replay = RequestExecutionLedger(request_id=ledger.request_id,
            input_digest=ledger.input_digest, model_signature=ledger.model_signature,
            token_count=ledger.token_count, num_layers=ledger.num_layers,
            exact_input_proof=ledger.exact_input_proof)
        for event in ledger.layers:
            checked = replay.record_layer(layer_1based=event.layer_1based,
                qkv_rows=event.qkv_rows, effective_current_kv_rows=event.current_kv_rows,
                attention_rows=event.attention_rows, output_mlp_rows=event.output_mlp_rows,
                imported_kv=event.imported_kv, completion_reference=event.completion_reference)
            if digest_json(asdict(checked)) != digest_json(asdict(event)):
                raise ValueError('registration event ranks differ from recomputed ledger')
        occurrences = tuple(occurrences)
        if not occurrences or any(type(o) is not TargetOccurrence for o in occurrences):
            raise ValueError('typed target occurrences required')
        if len({o.occurrence_id for o in occurrences}) != len(occurrences):
            raise ValueError('duplicate target occurrence id')
        ordered = tuple(sorted(occurrences, key=lambda o: (o.start, o.end, o.occurrence_id)))
        if ordered != occurrences or any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError('occurrences must follow non-overlapping request order')
        if any(o.end > len(tokens) for o in occurrences):
            raise ValueError('target outside request')
        causal_end = max(o.end for o in occurrences)
        parents = tuple(parent_sources)
        if any(type(p) is not ParentSourceMetadata for p in parents):
            raise ValueError('typed non-owning parent metadata required; no tensors or leases')
        if len({p.source_id for p in parents}) != len(parents):
            raise ValueError('duplicate parent identity')
        parent_by_id = {p.source_id: p for p in parents}
        for parent in parents:
            if (parent.model_signature != ledger.model_signature or
                    parent.authorization_domain != authorization_domain):
                raise ValueError('parent model/authorization mismatch')
        imported_ids = set()
        unverified_noncausal_ids = set()
        for layer in ledger.layers:
            for imported in layer.imported_kv:
                if imported.exact_prefix_proof is not None:
                    raise ValueError('native exact Prefix proof adapter not qualified for this registry')
                parent = parent_by_id.get(imported.source_id)
                if parent is None and imported.position >= causal_end:
                    # Preserve the full actual recipe, but do not let later
                    # unknown provenance invalidate an earlier causal target.
                    unverified_noncausal_ids.add(imported.source_id)
                    continue
                if parent is None or imported.generation != parent.generation:
                    raise ValueError('missing parent or imported generation mismatch')
                imported_ids.add(imported.source_id)
        if imported_ids != set(parent_by_id):
            raise ValueError('unreferenced parent metadata is not an actual execution recipe')
        occurrence_records = []
        for occurrence in occurrences:
            if occurrence.end > len(tokens):
                raise ValueError('target outside request')
            proof = ledger.target_proof(range(occurrence.start, occurrence.end))
            if proof.origin == SourceOrigin.UNKNOWN:
                raise ValueError('unknown target causal dependency in request manifest')
            record = asdict(occurrence)
            record.update(target_token_digest=digest_json(tokens[occurrence.start:occurrence.end]),
                          target_positions_digest=digest_json(positions[occurrence.start:occurrence.end]),
                          proof=asdict(proof), proof_digest=proof.proof_digest)
            occurrence_records.append(record)
        payload = dict(kind='source_request_execution_manifest_v2',
            birth_request_id=ledger.request_id, model_signature=ledger.model_signature,
            authorization_domain=authorization_domain, input_digest=ledger.input_digest,
            input_reference=input_reference, token_ids=tokens, absolute_positions=positions,
            num_layers=ledger.num_layers, exact_input_proof_reference=ledger.exact_input_proof,
            occurrences=occurrence_records, parent_sources=[asdict(p) for p in parents],
            unverified_noncausal_parent_ids=sorted(unverified_noncausal_ids),
            verified_target_causal_end=causal_end,
            executed_layers=[asdict(layer) for layer in ledger.layers],
            parent_references_own_KV=False, parent_references_hold_leases=False,
            request_completed=True, persistence='atomic_json' if self._root else 'in_memory_only',
            native_runtime_qualified=False)
        digest = digest_json(payload)
        manifest_id = 'request-manifest-v2-' + digest
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        prior = self._by_birth.get(ledger.request_id)
        if prior is not None:
            if prior != manifest_id:
                raise ValueError('birth request already frozen with different execution/content/domain')
            self._load(prior, model_signature=ledger.model_signature,
                       authorization_domain=authorization_domain)
            return prior
        path = self._manifest_path(manifest_id) if self._root is not None else None
        existing_bytes = 0
        if path is not None and path.exists():
            existing = self._read_bounded(path, self._max_manifest_bytes)
            if existing != encoded:
                raise ValueError('existing content-addressed manifest differs; refusing overwrite')
            existing_bytes = len(existing)
            if existing_bytes > self._orphan_bytes:
                raise ValueError('untracked persistent manifest appeared; reopen registry for audit')
        if (len(encoded) > self._max_manifest_bytes or
                self._bytes + self._orphan_bytes + len(encoded) - existing_bytes > self._max_bytes):
            raise MemoryError('shared manifest budget exceeded; registration not published')
        if self._root is not None:
            if not existing_bytes:
                self._atomic_write(path, encoded)
                self._orphan_bytes += len(encoded)
                existing_bytes = len(encoded)
            # A crash/failure here leaves a charged, invisible orphan only.
            self._write_catalog(dict(self._payloads, **{manifest_id: encoded}), self._revoked_domains)
            self._orphan_bytes -= existing_bytes
        self._payloads[manifest_id] = encoded
        self._by_birth[ledger.request_id] = manifest_id
        self._bytes += len(encoded)
        return manifest_id

    def _load(self, manifest_id, *, model_signature, authorization_domain):
        _text(manifest_id, 'manifest id')
        _text(model_signature, 'model signature')
        self._authorize(authorization_domain)
        self._check_catalog_current()
        encoded = self._payloads.get(manifest_id)
        if encoded is None:
            raise ValueError('unknown shared request manifest')
        if self._root is not None:
            disk = self._read_bounded(self._manifest_path(manifest_id), self._max_manifest_bytes)
            if disk != encoded:
                raise ValueError('persistent manifest changed after restore/registration')
        payload = self._parse(encoded)
        if self._last_validated_manifest != (manifest_id, encoded):
            payload = self._validate_payload(manifest_id, payload)
            self._last_validated_manifest = (manifest_id, encoded)
        if (payload['model_signature'] != model_signature or
                payload['authorization_domain'] != authorization_domain):
            raise ValueError('manifest model/authorization mismatch')
        return payload

    def validate_manifest(self, manifest_id, *, model_signature, authorization_domain):
        """Return detached payload; reconstruct once per unchanged byte identity.

        Catalog, persistent bytes, authorization and model are checked on every
        call. Reopening the registry does not inherit this validation cache.
        """
        return self._load(manifest_id, model_signature=model_signature,
                          authorization_domain=authorization_domain)

    def export_manifest(self, manifest_id, *, model_signature, authorization_domain):
        """Return a detached JSON object; caller edits never alter stored bytes."""
        return self._load(manifest_id, model_signature=model_signature,
                          authorization_domain=authorization_domain)

    def capture_reference(self, manifest_id, occurrence_id, *, model_signature,
                          authorization_domain, target_token_ids, target_positions):
        payload = self._load(manifest_id, model_signature=model_signature,
                             authorization_domain=authorization_domain)
        records = [o for o in payload['occurrences'] if o['occurrence_id'] == occurrence_id]
        if len(records) != 1:
            raise ValueError('unknown target occurrence')
        occurrence = records[0]
        tokens, positions = _identity(target_token_ids, target_positions)
        start, end = occurrence['start'], occurrence['end']
        if (list(tokens) != payload['token_ids'][start:end] or
                list(positions) != payload['absolute_positions'][start:end]):
            raise ValueError('target token/absolute position mismatch')
        proof = occurrence['proof']
        if (proof['origin'] not in (SourceOrigin.EXACT.value, SourceOrigin.MIXED.value) or
                proof['generation'] not in (0, 1)):
            raise ValueError('only complete known G0/G1 targets receive capture references')
        return ManifestReference(manifest_id, digest_json(payload), occurrence_id,
                                 positions[0], authorization_domain)

    def validate_capture_reference(self, reference, *, model_signature,
                                   authorization_domain, target_token_ids, target_positions):
        if type(reference) is not ManifestReference:
            raise ValueError('typed capture reference required')
        expected = self.capture_reference(reference.manifest_id, reference.target_occurrence,
            model_signature=model_signature, authorization_domain=authorization_domain,
            target_token_ids=target_token_ids, target_positions=target_positions)
        if reference != expected:
            raise ValueError('capture reference digest/target/domain was changed')
        return True

    def storage_audit(self):
        return dict(manifest_count=len(self._payloads), shared_manifest_json_bytes=self._bytes,
                    parent_owned_KV_bytes=0, parent_owned_lease_count=0,
                    durable=self.durable, byte_budget=self._max_bytes,
                    catalog_json_bytes=self._catalog_bytes,
                    unpublished_orphan_bytes=self._orphan_bytes,
                    total_persistent_bytes=(self._bytes+self._catalog_bytes+self._orphan_bytes
                                            if self.durable else 0),
                    catalog_byte_budget=self._max_bytes,
                    per_manifest_byte_budget=self._max_manifest_bytes,
                    python_container_overhead_included=False)
