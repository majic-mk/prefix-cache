"""Bounded target-only capture building block, not a qualified native publisher.

No forward calls, no Prefix shadow, no owning references to parent KV. Engine
integration must supply a real completed row ledger; P0 tensor tests alone do
not authorize mixed objects in the legacy Source Pool.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from .source_provenance_v2 import SourceOrigin
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest


@dataclass(frozen=True)
class ManifestReference:
    manifest_id: str
    manifest_digest: str
    target_occurrence: str
    prefix_end: int
    authorization_domain: str

    def __post_init__(self):
        if any(not isinstance(x,str) or not x for x in (
            self.manifest_id,self.manifest_digest,self.target_occurrence,self.authorization_domain)):
            raise ValueError('shared manifest and authorization references required')
        if type(self.prefix_end) is not int or self.prefix_end<0:
            raise ValueError('invalid prefix boundary')


class SegmentCapture:
    def __init__(self, ledger, positions, *, byte_budget, selection_depths):
        self.ledger=ledger
        self.positions=ledger._rows(positions)
        if not self.positions or self.positions != tuple(range(self.positions[0],self.positions[-1]+1)):
            raise ValueError('contiguous target required')
        if type(byte_budget) is not int or byte_budget<=0:
            raise ValueError('explicit positive capture budget required')
        self.depths=tuple(selection_depths)
        if len(set(self.depths))!=len(self.depths) or any(type(d) is not int or not 1<=d<ledger.num_layers for d in self.depths):
            raise ValueError('invalid completed-depth SelectionState')
        self.byte_budget=byte_budget; self._layers=[]; self._shape=None; self._closed=False

    def record_layer(self, layer_1based, key, value, *, row_positions):
        return self._record_layer(layer_1based, key, value, row_positions=row_positions,
                                  projection_positions=None)

    def record_projected_slice(self, packet):
        """Explicit target-only packet from the same layer's scoped QKV hook.

        It is not a composite layout. The original projection layout is bound
        separately, so a subset cannot impersonate the full projected tensor.
        """
        from .native_capture_hook_v2 import ProjectedKVSlice
        if not isinstance(packet, ProjectedKVSlice):
            raise ValueError('typed actual-projection packet required')
        return self._record_layer(packet.layer_1based, packet.key, packet.value,
            row_positions=packet.captured_positions, projection_positions=packet.projection_positions)

    def _record_layer(self, layer_1based, key, value, *, row_positions, projection_positions):
        import torch
        if self._closed or layer_1based!=len(self._layers)+1 or len(self.ledger.layers)!=layer_1based:
            raise ValueError('capture must follow completed execution without missing early layers')
        event=self.ledger.layers[-1]
        if not set(self.positions)<=set(event.output_mlp_rows):
            raise ValueError('target did not execute complete layer')
        rows=tuple(row_positions)
        if projection_positions is None and rows != event.qkv_rows:
            raise ValueError('capture requires the recorded current QKV projection layout')
        if projection_positions is not None and (tuple(projection_positions) != event.qkv_rows
                or not set(rows) <= set(projection_positions)):
            raise ValueError('target packet does not match actual projection layout')
        if len(set(rows))!=len(rows) or not set(self.positions)<=set(rows):
            raise ValueError('capture row identity mismatch')
        if any(type(i) is not int or not 0<=i<self.ledger.token_count for i in rows):
            raise ValueError('invalid absolute row positions')
        if (any(t.dtype!=torch.bfloat16 or t.ndim!=3 or t.shape[0]!=len(rows) for t in (key,value))
                or key.shape!=value.shape or key.device!=value.device):
            raise ValueError('BF16 [rows,KV heads,head dim] required')
        shape=(len(self.positions),)+tuple(key.shape[1:])
        if min(shape)<=0 or self._shape is not None and shape!=self._shape:
            raise ValueError('inconsistent capture geometry')
        layer_bytes=2*len(self.positions)*key.shape[1]*key.shape[2]*key.element_size()
        required=layer_bytes*self.ledger.num_layers+layer_bytes//2*len(self.depths)
        if required>self.byte_budget:
            raise MemoryError('capture and selection copies exceed reserved budget')
        indices=torch.tensor([rows.index(i) for i in self.positions],device=key.device)
        pair=tuple(t.detach().index_select(0,indices).to(device='cpu',copy=True).contiguous() for t in (key,value))
        if any(not bool(torch.isfinite(t).all()) for t in pair):
            raise ValueError('nonfinite target KV')
        self._shape=shape; self._layers.append(pair)

    def finalize(self, manifest, *, request_completed):
        if self._closed or request_completed is not True or len(self._layers)!=self.ledger.num_layers:
            raise ValueError('incomplete/closed capture')
        if not isinstance(manifest,ManifestReference) or manifest.prefix_end!=self.positions[0]:
            raise ValueError('target manifest reference mismatch')
        proof=self.ledger.target_proof(self.positions)
        if proof.origin not in (SourceOrigin.EXACT,SourceOrigin.MIXED) or proof.generation>1:
            raise ValueError('unknown/partial/G2 candidate cannot finalize in main policy')
        layers=tuple(self._layers)
        states={d:layers[d][0].clone() for d in self.depths}
        logical=tensor_digest(t for pair in layers for t in pair)
        identity=digest_json(dict(manifest=asdict(manifest),proof=asdict(proof),artifact_digest=logical))
        size=sum(t.untyped_storage().nbytes() for pair in layers for t in pair)
        state_size=sum(t.untyped_storage().nbytes() for t in states.values())
        self._closed=True; self._layers=[]
        return dict(source_artifact_id=identity,origin=proof.origin.value,generation=proof.generation,
            proof=proof,manifest_reference=manifest,layers=layers,selection_states=states,
            artifact_digest=logical,storage_audit=dict(target_kv_bytes=size,selection_state_bytes=state_size,
                parent_owned_kv_bytes=0,prefix_shadow_bytes=0),
            publication_state='VALIDATED_CANDIDATE_NOT_PUBLISHED',native_runtime_qualified=False)

    def close(self):
        self._layers=[]; self._closed=True
