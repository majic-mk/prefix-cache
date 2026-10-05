"""Isolated P0 target Source store; never relaxes the historical exact store.

Single writer, SSD backing, no GPU or model calls. EXACT and MIXED share one
capacity. This is a local transaction implementation, not a qualified online
dispatch. JSON/catalog and target tensors have separate, checked digests.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import os
from pathlib import Path
from threading import RLock
import time
import uuid

from .segment_capture_v2 import ManifestReference
from .source_provenance_v2 import PublicationScope, SourceOrigin, TargetProof, visible
from .source_manifest_v2 import ParentSourceMetadata
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest, file_digest
from .v8_schema10_layer_storage import LayerFile, write_layer_replica


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def parse_target_catalog_v2(envelope):
    """Validate and unwrap the durable catalog metadata, without I/O.

    Catalog files contain ``{payload, digest}``, never a flat config object.
    This shared parser verifies the envelope/schema/configuration; it does not
    certify backing tensor files or Source provenance. Opening the store still
    performs independent row, manifest, byte-budget and tensor verification.
    A detached payload prevents a consumer from mutating its input evidence.
    """
    if type(envelope) is not dict or set(envelope) != {'payload', 'digest'}:
        raise ValueError('invalid target catalog envelope; payload and digest required')
    payload = envelope['payload']; expected = envelope['digest']
    fields = {'kind', 'config', 'epoch', 'use_epoch', 'content_opportunities', 'rows'}
    if (type(payload) is not dict or set(payload) != fields
            or payload['kind'] != 'target_source_catalog_v2'
            or type(expected) is not str or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('invalid target catalog payload/schema/digest')
    try:
        actual = digest_json(payload)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('target catalog is not finite JSON metadata') from exc
    if actual != expected:
        raise ValueError('target catalog digest mismatch')
    config = payload['config']
    if (type(config) is not dict or set(config) != {
            'model_signature', 'authorization_domain', 'tokenizer_hash', 'policy',
            'max_bytes', 'staging_bytes', 'max_variants', 'probation_opportunities', 'purpose'}
            or any(type(config[k]) is not str or not config[k] for k in
                   ('model_signature', 'authorization_domain', 'tokenizer_hash'))
            or config['policy'] not in ('EXACT_ONLY', 'ALLOW_MIXED_G1')
            or config['purpose'] != 'isolated_P0_diagnostic'
            or any(type(config[k]) is not int or config[k] <= 0 for k in ('max_bytes', 'staging_bytes'))
            or type(config['max_variants']) is not int or config['max_variants'] not in (1, 2, 4)
            or type(config['probation_opportunities']) is not int or config['probation_opportunities'] < 0):
        raise ValueError('invalid target catalog configuration')
    if (type(payload['rows']) is not dict or type(payload['content_opportunities']) is not dict
            or any(type(payload[k]) is not int or payload[k] < 0 for k in ('epoch', 'use_epoch'))
            or any(type(k) is not str or not k or type(v) is not int or v < 0
                   for k, v in payload['content_opportunities'].items())
            or any(type(sid) is not str or not sid or type(row) is not dict or row.get('source_id') != sid
                   for sid, row in payload['rows'].items())):
        raise ValueError('invalid target catalog counters/row schema')
    return deepcopy(payload)


@dataclass(frozen=True)
class SourceSnapshotV2:
    snapshot_id: str
    request_id: str
    epoch: int
    rows: tuple


@dataclass(frozen=True)
class PublicationPlanV2:
    plan_id: str
    snapshot_id: str
    content_key: str
    source_id: str
    artifact_digest: str
    scope: PublicationScope
    existing_ids: tuple
    victim_id: str | None
    protection_epoch: int


class _LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        if self.stream.tell() + len(data) > self.limit:
            raise MemoryError('transaction staging file limit exceeded')
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class TargetSourceStoreV2:
    """All-or-nothing publication via one atomically replaced catalog.

    The root is an isolated v2 namespace. Existing result directories are not
    repurposed. Recovery verifies every referenced file and manifest; orphan
    files remain quarantined and counted, never silently promoted or deleted.
    ``max_bytes`` counts durable files and the shared manifest once. Extra
    transaction files have the independent, explicit ``staging_bytes`` limit.
    """
    def __init__(self, root, *, registry, model_signature, authorization_domain,
                 tokenizer_hash, policy, max_bytes, staging_bytes,
                 max_variants=4, probation_opportunities=2):
        if (policy not in ('EXACT_ONLY', 'ALLOW_MIXED_G1') or type(max_variants) is not int
                or max_variants not in (1, 2, 4)):
            raise ValueError('explicit v2 policy and shared K=1/2/4 required')
        if any(type(n) is not int or n <= 0 for n in (max_bytes, staging_bytes)):
            raise ValueError('positive explicit persistent/staging budgets required')
        if type(probation_opportunities) is not int or probation_opportunities < 0:
            raise ValueError('finite probation required')
        if any(type(s) is not str or not s for s in (model_signature, authorization_domain, tokenizer_hash)):
            raise ValueError('model, tokenizer and authorization required')
        if not registry.durable:
            raise ValueError('durable request registry required before durable Source publication')
        if Path(root).is_symlink():
            raise ValueError('isolated store root must not be a symlink')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.config = dict(model_signature=model_signature, authorization_domain=authorization_domain,
            tokenizer_hash=tokenizer_hash, policy=policy, max_bytes=max_bytes,
            staging_bytes=staging_bytes, max_variants=max_variants,
            probation_opportunities=probation_opportunities, purpose='isolated_P0_diagnostic')
        self.lock = RLock()
        self._lease_counts = {}; self._protection_epoch = 0
        self._snapshots = {}; self._plans = {}; self.events = []; self._lookup_markers = set()
        self._plan_comparisons = {}
        self._closed = False; self._poisoned = False; self._budget_blocked = False; self._catalog_encoded = None
        if (self.root/'writer.lock').is_symlink() or (self.root/'catalog.json').is_symlink():
            raise ValueError('store control files must not be symlinks')
        if not (self.root/'catalog.json').exists() and any(p.name!='writer.lock' for p in self.root.iterdir()):
            raise ValueError('nonempty root without v2 catalog; refusing to modify unrelated directory')
        self._writer = (self.root / 'writer.lock').open('a+b')
        try:
            if self._writer.seek(0, 2) == 0:
                self._writer.write(b'0'); self._writer.flush()
            self._writer.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._writer.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / 'catalog.json'
            if path.exists():
                self._catalog = self._recover(path)
                self._catalog_encoded = path.read_bytes()
            else:
                if any(p.name != 'writer.lock' for p in self.root.iterdir()):
                    raise ValueError('nonempty root without v2 catalog; explicit recovery required')
                self._catalog = dict(kind='target_source_catalog_v2', config=self.config,
                    epoch=0, use_epoch=0, content_opportunities={}, rows={})
                self._save_catalog(self._catalog)
        except Exception:
            self._writer.close()
            raise

    def _guard(self):
        if self._closed:
            raise RuntimeError('store closed')
        if self._poisoned:
            raise RuntimeError('catalog commit outcome uncertain; close and recover before further operations')
        if self._budget_blocked:
            raise MemoryError('committed backing cleanup incomplete; storage budget blocked')
        self.registry._authorize(self.config['authorization_domain'])
        self.registry._check_catalog_current()
        if self._catalog_encoded is not None:
            path = self.root/'catalog.json'
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size != len(self._catalog_encoded)
                    or path.read_bytes() != self._catalog_encoded):
                raise ValueError('store catalog changed outside its publication transaction')

    def content_key(self, tokens):
        tokens = tuple(tokens)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('exact target tokens required')
        return digest_json(dict(tokens=tokens, model=self.config['model_signature'],
            tokenizer=self.config['tokenizer_hash'], domain=self.config['authorization_domain']))

    def _path(self, name):
        # Catalogs do not get to choose arbitrary filesystem paths.
        if (type(name) is not str or Path(name).name != name or '/' in name or '\\' in name
                or not name.endswith(('.kv', '.states'))):
            raise ValueError('invalid owned artifact locator')
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError('artifact locator escapes isolated store')
        return path

    def _manifest_record(self, reference, tokens, positions):
        args = dict(model_signature=self.config['model_signature'],
                    authorization_domain=self.config['authorization_domain'])
        self.registry.validate_capture_reference(reference, target_token_ids=tokens,
            target_positions=positions, **args)
        manifest = self.registry.validate_manifest(reference.manifest_id, **args)
        record = next(o for o in manifest['occurrences'] if o['occurrence_id'] == reference.target_occurrence)
        return manifest, record

    def _validate_candidate(self, candidate, tokens):
        import torch
        proof = candidate.get('proof'); ref = candidate.get('manifest_reference')
        if not isinstance(proof, TargetProof) or not isinstance(ref, ManifestReference):
            raise ValueError('actual execution proof and shared manifest required')
        manifest, record = self._manifest_record(ref, tokens, proof.positions)
        if (digest_json(asdict(proof)) != record['proof_digest']
                or digest_json(asdict(proof)) != digest_json(record['proof'])
                or candidate.get('origin') != proof.origin.value
                or candidate.get('generation') != proof.generation
                or proof.generation not in (0, 1)
                or self.config['policy'] == 'EXACT_ONLY' and proof.generation != 0):
            raise ValueError('candidate provenance/policy disagrees with executed manifest')
        layers = candidate.get('layers', ())
        states = candidate.get('selection_states', {})
        if len(layers) != proof.expected_layers or not states:
            raise ValueError('missing full layers or independent SelectionState')
        shape = tuple(layers[0][0].shape)
        if len(shape) != 3 or shape[0] != len(tokens) or min(shape) <= 0:
            raise ValueError('invalid target geometry')
        tensors = []
        for pair in layers:
            if len(pair) != 2:
                raise ValueError('missing K/V pair')
            tensors.extend(pair)
        for depth, key in states.items():
            if type(depth) is not int or not 1 <= depth < len(layers):
                raise ValueError('invalid selection completed depth')
            if not torch.equal(key, layers[depth][0]):
                raise ValueError('SelectionState differs from observation-layer K')
        tensors.extend(states.values())
        pointers = []
        for t in tensors:
            if (not torch.is_tensor(t) or t.device.type != 'cpu' or t.dtype != torch.bfloat16
                    or tuple(t.shape) != shape or not t.is_contiguous() or t._base is not None
                    or t.storage_offset() != 0 or t.untyped_storage().nbytes() != t.numel()*t.element_size()
                    or not bool(torch.isfinite(t).all())):
                raise ValueError('Source must own finite compact CPU BF16 target allocations')
            pointers.append(t.data_ptr())
        if len(set(pointers)) != len(pointers):
            raise ValueError('Source/SelectionState must not alias other captured allocations')
        logical = tensor_digest(t for pair in layers for t in pair)
        identity = digest_json(dict(manifest=asdict(ref), proof=asdict(proof), artifact_digest=logical))
        if logical != candidate.get('artifact_digest') or identity != candidate.get('source_artifact_id'):
            raise ValueError('candidate Artifact identity or bytes changed')
        host_bytes = sum(t.numel()*t.element_size() for t in tensors)
        # Input owner remains the capture; allow room for one validation layer
        # and serialization copies, all explicitly bounded in this CPU path.
        if host_bytes * 3 > self.config['staging_bytes']:
            raise MemoryError('target validation/serialization host staging budget exceeded')
        return dict(source_id=identity, content_key=self.content_key(tokens), token_ids=list(tokens),
            model_signature=proof.model_signature, authorization_domain=ref.authorization_domain,
            origin=proof.origin.value, generation=proof.generation, birth_request_id=proof.request_id,
            birth_target_positions=list(proof.positions), manifest_reference=asdict(ref),
            proof_digest=proof.proof_digest, artifact_digest=logical, shape=list(shape),
            layer_count=len(layers), selection_depths=sorted(states),
            target_kv_bytes=sum(t.numel()*t.element_size() for p in layers for t in p),
            selection_state_bytes=sum(t.numel()*t.element_size() for t in states.values()),
            selection_digest=tensor_digest(states[d] for d in sorted(states)),
            parent_owned_kv_bytes=0, prefix_shadow_bytes=0)

    def _disk_bytes(self):
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def _manifest_bytes(self):
        audit = self.registry.storage_audit()
        return audit['total_persistent_bytes']

    def _save_catalog(self, catalog, *, reclaim_bytes=0):
        self._guard()
        encoded = _json(dict(payload=catalog, digest=digest_json(catalog)))
        if len(encoded) > self.config['staging_bytes']:
            raise MemoryError('catalog exceeds staging reservation')
        old_size = len(self._catalog_encoded) if self._catalog_encoded is not None else 0
        if self._disk_bytes()+self._manifest_bytes()-old_size-reclaim_bytes+len(encoded)>self.config['max_bytes']:
            raise MemoryError('catalog and shared manifest exceed persistent budget')
        temporary = self.root / (uuid.uuid4().hex + '.catalog.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.root / 'catalog.json')
            if os.name != 'nt':
                descriptor = os.open(str(self.root), os.O_RDONLY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
            self._catalog_encoded = encoded
        except Exception:
            # A replace may have succeeded before an I/O error was reported.
            # Never delete the new backing files in that ambiguous state.
            try:
                path = self.root/'catalog.json'
                actual = path.read_bytes() if path.exists() else None
                if actual != self._catalog_encoded:
                    self._poisoned = True
            except OSError:
                self._poisoned = True
            raise
        finally:
            if temporary.exists():
                temporary.unlink()  # Only this transaction's exact new path.

    def _recover(self, path):
        if path.stat().st_size > self.config['max_bytes']:
            raise MemoryError('catalog exceeds persistent budget')
        if self._disk_bytes()+self._manifest_bytes()>self.config['max_bytes']:
            raise MemoryError('persistent files including orphan quarantine exceed budget')
        payload = parse_target_catalog_v2(self.registry._parse(path.read_bytes()))
        if payload['config'] != self.config:
            raise ValueError('catalog identity/config/digest mismatch')
        groups = {}
        for sid, row in payload['rows'].items():
            if (row['target_kv_bytes']+row['selection_state_bytes'])*3>self.config['staging_bytes']:
                raise MemoryError('restored target verification exceeds host staging budget')
            self._verify_row(row, full=True)
            if (any(type(row[k]) is not int or row[k]<0 for k in
                    ('publication_epoch','last_request_use_epoch','grace_until'))
                    or sid != row['source_id'] or row['publication_epoch'] > payload['epoch']
                    or row['publication_epoch'] < 1 or row['last_request_use_epoch'] > payload['use_epoch']):
                raise ValueError('invalid catalog identity/epochs')
            groups.setdefault(row['content_key'], []).append(row)
        if any(len(rows)>self.config['max_variants'] for rows in groups.values()):
            raise ValueError('EXACT/MIXED shared capacity exceeded')
        if self._disk_bytes()+self._manifest_bytes() > self.config['max_bytes']:
            raise MemoryError('persistent files including orphan quarantine exceed budget')
        return payload

    def _verify_row(self, row, *, full):
        import torch
        ref = ManifestReference(**row['manifest_reference'])
        manifest, record = self._manifest_record(ref, row['token_ids'], row['birth_target_positions'])
        proof = record['proof']
        if (row['content_key'] != self.content_key(row['token_ids'])
                or row['model_signature'] != self.config['model_signature']
                or row['authorization_domain'] != self.config['authorization_domain']
                or row['origin'] != proof['origin'] or row['generation'] != proof['generation']
                or row['proof_digest'] != record['proof_digest']
                or row['birth_request_id'] != manifest['birth_request_id']
                or row['generation'] not in (0,1)
                or self.config['policy']=='EXACT_ONLY' and row['generation']!=0
                or row['source_id'] != digest_json(dict(manifest=asdict(ref),proof=proof,
                                                       artifact_digest=row['artifact_digest']))):
            raise ValueError('stored Source metadata differs from verified manifest')
        kv, selection = self._path(row['kv_file']), self._path(row['selection_file'])
        if not kv.is_file() or not selection.is_file():
            raise ValueError('missing Source backing/SelectionState')
        if not full:
            return
        if file_digest(kv)!=row['kv_file_digest'] or file_digest(selection)!=row['selection_file_digest']:
            raise ValueError('stored file digest mismatch')
        layers = LayerFile(kv)
        if (tuple(row['shape'])!=layers.shape or row['layer_count']!=len(layers)
                or layers.shape[0]!=len(row['token_ids']) or len(layers)!=proof['expected_layers']
                or row['target_kv_bytes']!=layers.full_kv_bytes
                or row['parent_owned_kv_bytes']!=0 or row['prefix_shadow_bytes']!=0):
            raise ValueError('stored Source geometry mismatch')
        if tensor_digest(t for pair in layers for t in pair)!=row['artifact_digest']:
            raise ValueError('stored Artifact logical digest mismatch')
        states = torch.load(selection,weights_only=True,map_location='cpu')
        if (type(states) is not dict or not states or sorted(states)!=row['selection_depths']
                or tensor_digest(states[d] for d in sorted(states))!=row['selection_digest']
                or sum(t.numel()*t.element_size() for t in states.values())!=row['selection_state_bytes']):
            raise ValueError('stored SelectionState identity mismatch')
        for d,k in states.items():
            if not 1<=d<len(layers) or not torch.equal(k,layers[d][0]):
                raise ValueError('stored SelectionState differs from Artifact')

    def begin_request(self, request_id):
        with self.lock:
            self._guard()
            if type(request_id) is not str or not request_id or self._snapshots:
                raise ValueError('one active request and explicit request ID required')
            snapshot = SourceSnapshotV2(uuid.uuid4().hex, request_id, self._catalog['epoch'],
                tuple(sorted(self._catalog['rows'])))
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def _snapshot(self, snapshot):
        self._guard()
        if type(snapshot) is not SourceSnapshotV2 or self._snapshots.get(snapshot.snapshot_id)!=snapshot:
            raise ValueError('unknown, changed or ended request snapshot')

    def end_request(self, snapshot):
        with self.lock:
            self._snapshot(snapshot)
            if any(self._lease_counts.values()):
                raise RuntimeError('request ends only after its physical reads/copies are fenced and released')
            self._snapshots.pop(snapshot.snapshot_id)
            self._plan_comparisons={k:v for k,v in self._plan_comparisons.items()
                                    if v[0].snapshot.snapshot_id!=snapshot.snapshot_id}
            self._plans = {k:v for k,v in self._plans.items() if v.snapshot_id!=snapshot.snapshot_id}
            self._lookup_markers = {m for m in self._lookup_markers if m[0]!=snapshot.snapshot_id}

    def lookup(self, snapshot, tokens):
        with self.lock:
            self._snapshot(snapshot)
            key = self.content_key(tokens)
            rows = [r for sid,r in self._catalog['rows'].items() if sid in snapshot.rows
                    and r['content_key']==key and visible(birth_request_id=r['birth_request_id'],
                        publication_epoch=r['publication_epoch'], reader_request_id=snapshot.request_id,
                        snapshot_epoch=snapshot.epoch)]
            return tuple(deepcopy(r) for r in sorted(rows,key=lambda r:(r['publication_epoch'],r['source_id'])))

    def _visible_row(self, snapshot, source_id):
        self._snapshot(snapshot)
        row = self._catalog['rows'].get(source_id)
        if (row is None or source_id not in snapshot.rows or not visible(
                birth_request_id=row['birth_request_id'],publication_epoch=row['publication_epoch'],
                reader_request_id=snapshot.request_id,snapshot_epoch=snapshot.epoch)):
            raise ValueError('Source is absent, self-born or outside frozen visible snapshot')
        self._verify_row(row,full=False)
        return row

    def record_lookup_opportunity(self, snapshot, tokens):
        with self.lock:
            self._snapshot(snapshot)
            key = self.content_key(tokens)
            marker = (snapshot.snapshot_id,key)
            if marker in self._lookup_markers: return
            # The operation observes this request's real frozen lookup scope.
            self.lookup(snapshot,tokens)
            new = deepcopy(self._catalog)
            new['content_opportunities'][key]=new['content_opportunities'].get(key,0)+1
            for sid,row in new['rows'].items():
                if row['content_key']==key and (sid not in snapshot.rows or row['birth_request_id']==snapshot.request_id):
                    # Publication-before-accounting must not count the birth
                    # request as one of the child's subsequent opportunities.
                    row['grace_until']+=1
            self._save_catalog(new); self._catalog=new
            self._lookup_markers.add(marker)

    def _protected(self, row):
        opportunities = self._catalog['content_opportunities'].get(row['content_key'],0)
        return bool(self._lease_counts.get(row['source_id']) or opportunities<row['grace_until'])

    def plan_publication(self, snapshot, candidate, *, token_ids, scope):
        """Low-level isolated CPU transaction input, not comparison evidence.

        Executed comparison paths must use plan_from_comparison instead.
        Keeping this rule-level entry does not grant production growth.
        """
        with self.lock:
            self._snapshot(snapshot)
            meta = self._validate_candidate(candidate,tuple(token_ids))
            if meta['birth_request_id']!=snapshot.request_id:
                raise ValueError('publication must belong to the completed birth request')
            rows = tuple(sorted(sid for sid,r in self._catalog['rows'].items() if r['content_key']==meta['content_key']))
            frozen = tuple(sorted(sid for sid in snapshot.rows if sid in self._catalog['rows']
                                  and self._catalog['rows'][sid]['content_key']==meta['content_key']))
            if rows!=frozen or type(scope) is not PublicationScope or scope.stored!=len(rows) or scope.rejection():
                raise ValueError('stale/incomplete actual stored scope or non-growth trigger')
            duplicate = next((r for r in self._catalog['rows'].values()
                if r['content_key']==meta['content_key'] and r['artifact_digest']==meta['artifact_digest']),None)
            victim = None
            if not duplicate and len(rows)>=self.config['max_variants']:
                available = [self._catalog['rows'][sid] for sid in rows if not self._protected(self._catalog['rows'][sid])]
                if not available: raise MemoryError('all shared-capacity victims protected')
                victim = min(available,key=lambda r:(r['last_request_use_epoch'],r['publication_epoch'],r['source_id']))['source_id']
            plan = PublicationPlanV2(uuid.uuid4().hex,snapshot.snapshot_id,meta['content_key'],
                meta['source_id'],meta['artifact_digest'],scope,rows,victim,self._protection_epoch)
            self._plans[plan.plan_id]=plan
            return plan

    def plan_from_comparison(self, snapshot, candidate, *, token_ids, comparison, receipt):
        from .source_comparison_v2 import ComparisonSessionV2, ComparisonReceiptV2
        with self.lock:
            if (type(comparison) is not ComparisonSessionV2 or type(receipt) is not ComparisonReceiptV2
                    or comparison.store is not self or comparison.snapshot!=snapshot
                    or tuple(token_ids)!=receipt.token_ids):
                raise ValueError('publication requires this store/request/content comparison issuer')
            scope=comparison.publication_scope(receipt)
            proof=candidate.get('proof')
            if (proof is None or proof.positions!=receipt.absolute_positions
                    or proof.input_digest!=receipt.request_input_digest):
                raise ValueError('candidate birth input/positions differ from actual comparison')
            plan=self.plan_publication(snapshot,candidate,token_ids=token_ids,scope=scope)
            self._plan_comparisons[plan.plan_id]=(comparison,receipt)
            return plan

    def _stage(self, suffix, writer, created):
        path = self.root/(uuid.uuid4().hex+suffix)
        created.append(path)
        remaining = self.config['staging_bytes']-sum(p.stat().st_size for p in created if p.exists())
        with path.open('xb') as stream:
            writer(_LimitedWriter(stream,remaining)); stream.flush(); os.fsync(stream.fileno())
        return path

    def commit_publication(self, snapshot, plan, candidate, *, token_ids):
        """Failure is a skipped publication, never an implicit model rerun."""
        import torch
        start = time.perf_counter_ns(); created=[]; published=False; comparison_binding=None
        comparison_verified=False
        with self.lock:
            try:
                self._snapshot(snapshot)
                if (type(plan) is not PublicationPlanV2 or self._plans.get(plan.plan_id)!=plan
                        or plan.snapshot_id!=snapshot.snapshot_id):
                    raise ValueError('unissued/stale publication transaction')
                self._plans.pop(plan.plan_id)
                comparison_binding=self._plan_comparisons.pop(plan.plan_id,None)
                if comparison_binding is not None:
                    issuer,receipt=comparison_binding
                    if issuer.publication_scope(receipt)!=plan.scope:
                        raise ValueError('comparison receipt changed before publication commit')
                    comparison_verified=True
                meta = self._validate_candidate(candidate,tuple(token_ids))
                ids=tuple(sorted(sid for sid,r in self._catalog['rows'].items() if r['content_key']==plan.content_key))
                if (ids!=plan.existing_ids or meta['source_id']!=plan.source_id
                        or meta['content_key']!=plan.content_key or meta['birth_request_id']!=snapshot.request_id
                        or plan.protection_epoch!=self._protection_epoch):
                    raise ValueError('publication/victim protection changed; replan without silent substitution')
                if plan.victim_id and self._protected(self._catalog['rows'][plan.victim_id]):
                    raise ValueError('preselected victim now protected')
                duplicate = next((r for r in self._catalog['rows'].values() if
                    r['content_key']==plan.content_key and r['artifact_digest']==meta['artifact_digest']),None)
                if duplicate:
                    return dict(status='DUPLICATE',source_id=duplicate['source_id'],existing_origin=duplicate['origin'],
                                rewritten_bytes=0,provenance_changed=False,publication_performed=False)
                kv = self._stage('.kv',lambda f:write_layer_replica(f,candidate['layers']),created)
                states = self._stage('.states',lambda f:torch.save(candidate['selection_states'],f),created)
                meta.update(kv_file=kv.name,selection_file=states.name,kv_file_digest=file_digest(kv),
                    selection_file_digest=file_digest(states),publication_epoch=self._catalog['epoch']+1,
                    last_request_use_epoch=0,grace_until=self._catalog['content_opportunities'].get(plan.content_key,0)
                        +self.config['probation_opportunities'])
                self._verify_row(meta,full=True)
                new=deepcopy(self._catalog); new['epoch']+=1
                victim = new['rows'].pop(plan.victim_id,None)
                new['rows'][meta['source_id']]=meta
                catalog_bytes=len(_json(dict(payload=new,digest=digest_json(new))))
                staged_bytes=sum(p.stat().st_size for p in created)
                if staged_bytes+catalog_bytes>self.config['staging_bytes']:
                    raise MemoryError('complete transaction exceeds staging budget')
                reclaimed=sum(self._path(victim[k]).stat().st_size for k in ('kv_file','selection_file')) if victim else 0
                current_catalog=(self.root/'catalog.json').stat().st_size
                if self._disk_bytes()+self._manifest_bytes()-reclaimed-current_catalog+catalog_bytes>self.config['max_bytes']:
                    raise MemoryError('shared persistent byte budget exceeded')
                self._save_catalog(new,reclaim_bytes=reclaimed)
                self._catalog=new; published=True
                cleanup_pending=[]
                if victim:
                    for k in ('kv_file','selection_file'):
                        try: self._path(victim[k]).unlink()
                        except OSError: cleanup_pending.append(victim[k])
                self._budget_blocked=self._disk_bytes()+self._manifest_bytes()>self.config['max_bytes']
                return dict(status='PUBLISHED_CLEANUP_PENDING' if cleanup_pending else 'PUBLISHED',
                            source_id=meta['source_id'],origin=meta['origin'],publication_performed=True,
                            generation=meta['generation'],publication_epoch=new['epoch'],evicted_source_id=plan.victim_id,
                            cleanup_pending=cleanup_pending,persistent_budget_satisfied=not self._budget_blocked)
            except (ValueError,KeyError,TypeError,MemoryError,OSError,RuntimeError) as exc:
                if published: raise
                if self._poisoned:
                    return dict(status='COMMIT_UNCERTAIN',reason=str(exc),publication_performed=None,
                                recovery_required=True)
                return dict(status='REJECTED',reason=type(exc).__name__+':'+str(exc),publication_performed=False)
            finally:
                if not published and not self._poisoned:
                    for path in created:
                        if path.exists() and path.parent==self.root:
                            try: path.unlink()
                            except OSError: pass  # Invisible quarantined files remain charged.
                self.events.append(dict(event='publication_attempt',request_id=snapshot.request_id,
                    publication_performed=None if self._poisoned else published,
                    recovery_required=self._poisoned,host_ms=(time.perf_counter_ns()-start)/1e6,
                    extra_forward_count=0,diagnostic_only=True,
                    comparison_receipt_id=comparison_binding[1].receipt_id if comparison_binding else None,
                    verified_comparison_binding=comparison_verified))
                if comparison_binding is not None:
                    comparison_binding[0].mark_publication_used(comparison_binding[1])

    def read_selection(self, snapshot, source_id, depth):
        import torch
        with self.lock:
            row=self._visible_row(snapshot,source_id)
            path=self._path(row['selection_file'])
            if file_digest(path)!=row['selection_file_digest']: raise ValueError('corrupt SelectionState')
            states=torch.load(path,weights_only=True,map_location='cpu')
            if depth not in states: raise KeyError('SelectionState absent; full-KV fallback prohibited')
            return states[depth]

    @contextmanager
    def leased_target(self, snapshot, source_id):
        with self.lock:
            row=self._visible_row(snapshot,source_id)
            layers=LayerFile(self._path(row['kv_file']))
            self._lease_counts[source_id]=self._lease_counts.get(source_id,0)+1
            self._protection_epoch+=1
        try: yield layers
        finally:
            with self.lock:
                self._lease_counts[source_id]-=1
                self._protection_epoch+=1

    def mark_request_use(self, snapshot, source_id):
        with self.lock:
            self._visible_row(snapshot,source_id)
            if not self._lease_counts.get(source_id): raise ValueError('use requires held target lease')
            new=deepcopy(self._catalog);new['use_epoch']+=1
            new['rows'][source_id]['last_request_use_epoch']=new['use_epoch']
            self._save_catalog(new);self._catalog=new;self._protection_epoch+=1

    def parent_metadata(self, snapshot, source_id):
        with self.lock:
            row=self._visible_row(snapshot,source_id)
            return ParentSourceMetadata(source_id,row['artifact_digest'],self.config['model_signature'],
                self.config['authorization_domain'],SourceOrigin(row['origin']),row['generation'])

    def storage_audit(self):
        with self.lock:
            referenced={r[k] for r in self._catalog['rows'].values() for k in ('kv_file','selection_file')}
            orphan=[p for p in self.root.iterdir() if p.is_file() and p.name not in referenced|{'catalog.json','writer.lock'}]
            return dict(source_count=len(self._catalog['rows']),shared_max_variants=self.config['max_variants'],
                target_kv_bytes=sum(r['target_kv_bytes'] for r in self._catalog['rows'].values()),
                selection_state_bytes=sum(r['selection_state_bytes'] for r in self._catalog['rows'].values()),
                store_disk_bytes=self._disk_bytes(),shared_manifest_bytes=self._manifest_bytes(),
                quarantine_bytes=sum(p.stat().st_size for p in orphan),quarantine_files=[p.name for p in orphan],
                persistent_budget_bytes=self.config['max_bytes'],transaction_staging_budget_bytes=self.config['staging_bytes'],
                persistent_budget_satisfied=self._disk_bytes()+self._manifest_bytes()<=self.config['max_bytes'],
                recovery_required=self._poisoned,cleanup_budget_blocked=self._budget_blocked,
                power_loss_durability='POSIX_directory_fsync' if os.name!='nt' else 'atomic_replace_only',
                parent_owned_kv_bytes=0,parent_lease_count=0,prefix_shadow_bytes=0,
                physical_backing='SSD_ONLY_P0',single_writer=True,production_dispatch_integrated=False,
                diagnostic_only=True,native_runtime_qualified=False)

    def close(self):
        with self.lock:
            if self._closed: return
            if any(self._lease_counts.values()): raise RuntimeError('cannot close with in-flight target leases')
            self._writer.close();self._closed=True

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
