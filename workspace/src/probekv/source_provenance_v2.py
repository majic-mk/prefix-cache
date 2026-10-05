"""Versioned P0 provenance ledger. CPU-tested; native execution hooks pending.

Ranks bound dependency ancestry, NOT numerical error. Callers must record actual
executed rows after successful layer completion; a ratio or function name is not
execution evidence. This module never relaxes legacy DENSE_EXACT publication.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
import math
from .v8_schema10_execution import digest_json


class SourceOrigin(str, Enum):
    EXACT = 'EXACT_CONTEXT'
    MIXED = 'MIXED_CONTEXT_FULL_SEGMENT'
    UNKNOWN = 'UNKNOWN_PROVENANCE'
    PARTIAL = 'PARTIALLY_REPAIRED_WORKING_KV'


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid integer geometry/rank')
    return value


def inherited_rank(values):
    values = tuple(values)
    for v in values:
        if v is not None:
            _integer(v)
    return None if None in values else max(values, default=0)


def imported_rank(generation, *, exact_prefix_proof=None):
    if generation is not None:
        _integer(generation)
    if exact_prefix_proof is not None:
        if generation != 0 or not isinstance(exact_prefix_proof, str) or not exact_prefix_proof:
            raise ValueError('exact Prefix requires verified G0 proof reference')
        return 0
    return None if generation is None else generation + 1


@dataclass(frozen=True)
class KVImport:
    position: int
    source_id: str
    generation: int | None
    exact_prefix_proof: str | None = None

    def __post_init__(self):
        _integer(self.position)
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError('source identity required')
        imported_rank(self.generation, exact_prefix_proof=self.exact_prefix_proof)


@dataclass(frozen=True)
class LayerExecution:
    layer_1based: int
    qkv_rows: tuple
    current_kv_rows: tuple
    attention_rows: tuple
    output_mlp_rows: tuple
    imported_kv: tuple
    output_ranks: tuple
    kv_ranks: tuple
    completion_reference: str


@dataclass(frozen=True)
class TargetProof:
    request_id: str
    input_digest: str
    model_signature: str
    positions: tuple
    origin: SourceOrigin
    generation: int | None
    expected_layers: int
    ledger_digest: str

    def __post_init__(self):
        if any(not isinstance(v,str) or not v for v in
               (self.request_id,self.input_digest,self.model_signature,self.ledger_digest)):
            raise ValueError('proof identity required')
        _integer(self.expected_layers,1)
        if self.generation is not None:
            _integer(self.generation)
        if not isinstance(self.origin,SourceOrigin):
            raise ValueError('typed origin required')
        if not isinstance(self.positions,tuple) or not self.positions:
            raise ValueError('immutable target positions required')
        for i in self.positions:
            _integer(i)
        if self.positions!=tuple(range(self.positions[0],self.positions[-1]+1)):
            raise ValueError('noncontiguous proof positions')
        if ((self.origin==SourceOrigin.EXACT and self.generation!=0)
            or (self.origin==SourceOrigin.MIXED and (self.generation is None or self.generation<1))
            or (self.origin in (SourceOrigin.UNKNOWN,SourceOrigin.PARTIAL) and self.generation is not None)):
            raise ValueError('inconsistent origin/rank')

    @property
    def proof_digest(self):
        return digest_json(asdict(self))


class RequestExecutionLedger:
    """Conservative full-causal row tracker; missing rows become UNKNOWN.

    A finer sliding-window implementation may reduce overestimation only after
    separate verification. No tensor or parent Source leases are retained here.
    """
    def __init__(self, *, request_id, input_digest, model_signature, token_count,
                 num_layers, exact_input_proof=None):
        if any(not isinstance(v, str) or not v for v in (request_id,input_digest,model_signature)):
            raise ValueError('bound request/input/model identity required')
        _integer(token_count, 1); _integer(num_layers, 1)
        if exact_input_proof is not None and (not isinstance(exact_input_proof,str) or not exact_input_proof):
            raise ValueError('invalid exact input proof reference')
        self.request_id, self.input_digest, self.model_signature = request_id,input_digest,model_signature
        self.token_count, self.num_layers = token_count,num_layers
        # Trusted-verifier reference only. Native hook must verify model,
        # tokens and positions before supplying this; strings are not a proof.
        self.exact_input_proof = exact_input_proof
        self._bound_identity = (request_id, input_digest, model_signature, token_count,
                                num_layers, exact_input_proof)
        self._ranks = (0 if exact_input_proof else None,) * token_count
        self._layers = []

    @property
    def layers(self):
        return tuple(self._layers)

    def _rows(self, rows):
        rows = tuple(rows)
        if any(type(i) is not int or not 0 <= i < self.token_count for i in rows) or len(set(rows)) != len(rows):
            raise ValueError('duplicate/out-of-range execution positions')
        return tuple(sorted(rows))

    def _check_identity(self):
        if (self.request_id,self.input_digest,self.model_signature,self.token_count,
                self.num_layers,self.exact_input_proof) != self._bound_identity:
            raise ValueError('execution identity changed after ledger creation')

    def record_layer(self, *, layer_1based, qkv_rows, attention_rows, output_mlp_rows,
                     imported_kv=(), completion_reference, effective_current_kv_rows=None):
        self._check_identity()
        if type(layer_1based) is not int or layer_1based != len(self._layers)+1 or layer_1based>self.num_layers:
            raise ValueError('missing, duplicate or out-of-order layer')
        if not isinstance(completion_reference,str) or not completion_reference:
            raise ValueError('successful execution completion reference required')
        q,a,o = (self._rows(x) for x in (qkv_rows,attention_rows,output_mlp_rows))
        if not set(o) <= set(a) <= set(q):
            raise ValueError('output/attention requires current QKV rows')
        current = q if effective_current_kv_rows is None else self._rows(effective_current_kv_rows)
        if not set(a) <= set(current) <= set(q):
            raise ValueError('effective KV rows must match actual projection and attention')
        imports = tuple(imported_kv)
        if any(not isinstance(x,KVImport) for x in imports):
            raise ValueError('typed KV import required')
        positions=self._rows(x.position for x in imports)
        if set(positions)&set(current):
            raise ValueError('record only effective reused rows, not overwritten repair rows')
        kv=[None]*self.token_count
        for i in current:
            kv[i]=self._ranks[i]
        for x in imports:
            kv[x.position]=imported_rank(x.generation,exact_prefix_proof=x.exact_prefix_proof)
        causal=[]; rank=0
        for v in kv:
            rank=inherited_rank((rank,v)); causal.append(rank)
        out=[None]*self.token_count
        for i in o:
            out[i]=inherited_rank((self._ranks[i],causal[i]))
        row=LayerExecution(layer_1based,q,current,a,o,imports,tuple(out),tuple(kv),completion_reference)
        self._layers.append(row); self._ranks=tuple(out)
        return row

    def target_proof(self, positions):
        self._check_identity()
        positions=self._rows(positions)
        if not positions or positions != tuple(range(positions[0],positions[-1]+1)):
            raise ValueError('nonempty contiguous target required')
        complete=len(self._layers)==self.num_layers and all(
            set(positions)<=set(r.qkv_rows)&set(r.attention_rows)&set(r.output_mlp_rows)
            and not set(positions)&{x.position for x in r.imported_kv} for r in self._layers)
        rank=inherited_rank(v for r in self._layers for i in positions for v in (r.output_ranks[i],r.kv_ranks[i])) if complete else None
        origin=(SourceOrigin.PARTIAL if not complete else SourceOrigin.UNKNOWN if rank is None
                else SourceOrigin.EXACT if rank==0 else SourceOrigin.MIXED)
        return TargetProof(self.request_id,self.input_digest,self.model_signature,positions,origin,rank,
            self.num_layers,digest_json(dict(exact_input_proof=self.exact_input_proof,
                request_id=self.request_id,input_digest=self.input_digest,model_signature=self.model_signature,
                token_count=self.token_count,num_layers=self.num_layers,layers=[asdict(r) for r in self._layers])))


@dataclass(frozen=True)
class PublicationScope:
    reason: str
    stored: int
    eligible: int
    available: int
    compared: int
    best_score: float | None = None
    threshold: float | None = None

    def rejection(self):
        for n in (self.stored,self.eligible,self.available,self.compared):
            _integer(n)
        if not 0 <= self.compared <= self.available <= self.eligible <= self.stored:
            return 'invalid_scope'
        for x in (self.best_score,self.threshold):
            if x is not None and (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0):
                return 'nonfinite_or_negative_score'
        if self.reason=='content_miss':
            return None if self.stored==0 else 'not_content_miss'
        if self.reason!='complete_scope_mismatch':
            return 'not_growth_trigger'
        if not self.stored or len({self.stored,self.eligible,self.available,self.compared})!=1:
            return 'incomplete_stored_scope'
        if self.best_score is None or self.threshold is None:
            return 'missing_threshold'
        return None if self.best_score>self.threshold else 'compatible_source_exists'


def publication_rejection(proof, scope, *, policy, permission_verified,
                          request_completed, artifact_complete, budget_ok, capacity_ok,
                          snapshot_current, duplicate=False):
    if policy not in ('EXACT_ONLY','ALLOW_MIXED_G1'):
        raise ValueError('unknown v2 provenance policy')
    if not isinstance(proof,TargetProof) or not isinstance(scope,PublicationScope):
        raise ValueError('typed execution proof and scope required')
    if proof.origin not in (SourceOrigin.EXACT,SourceOrigin.MIXED) or proof.generation is None:
        return 'incomplete_or_unknown_provenance'
    if (proof.origin==SourceOrigin.EXACT)!=(proof.generation==0) or proof.generation<0:
        return 'inconsistent_provenance'
    if proof.generation > (0 if policy=='EXACT_ONLY' else 1):
        return 'generation_limit'
    if scope.rejection():
        return scope.rejection()
    for name,value in dict(permission_verified=permission_verified,request_completed=request_completed,
        artifact_complete=artifact_complete,budget_ok=budget_ok,capacity_ok=capacity_ok,
        snapshot_current=snapshot_current).items():
        if value is not True:
            return name
    return 'duplicate' if duplicate else None


def visible(*, birth_request_id, publication_epoch, reader_request_id, snapshot_epoch):
    _integer(publication_epoch); _integer(snapshot_epoch)
    if not birth_request_id or not reader_request_id:
        raise ValueError('bound requests required')
    return birth_request_id!=reader_request_id and publication_epoch<=snapshot_epoch
