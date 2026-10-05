"""Opt-in P0 bridge from actual layer calls to target-only Source candidates.

This is NOT an online publisher or GPU qualification. Capture failure skips
materialization; model/CUDA exceptions still abort execution. Default historical
paths do not instantiate this object or install its hooks.
"""
from __future__ import annotations
from dataclasses import replace
import time

from .source_provenance_v2 import KVImport, RequestExecutionLedger
from .segment_capture_v2 import SegmentCapture


class NativeTargetCaptureV2:
    def __init__(self, *, request_id, token_ids, model_signature, authorization_domain,
                 num_layers, targets, selection_depths, host_budget_bytes,
                 kv_heads, head_dim, registry):
        from .source_manifest_v2 import request_input_digest
        if not authorization_domain or not targets:
            raise ValueError('explicit domain and target reservations required')
        if type(host_budget_bytes) is not int or host_budget_bytes <= 0:
            raise ValueError('explicit host capture budget required')
        if any(type(v) is not int or v <= 0 for v in (kv_heads,head_dim)):
            raise ValueError('positive KV geometry required')
        self.tokens=tuple(token_ids); self.positions=tuple(range(len(self.tokens)))
        self.request_id=request_id; self.model_signature=model_signature
        self.authorization_domain=authorization_domain; self.registry=registry
        self.targets={sid:tuple(pos) for sid,pos in targets.items()}
        flattened=[p for pos in self.targets.values() for p in pos]
        if len(set(flattened))!=len(flattened):
            raise ValueError('capture target reservations overlap')
        identity=request_input_digest(self.tokens,self.positions)
        # This reference binds the native request's original embedding inputs.
        # It does not certify a Source, Prefix hit, model revision or GPU patch.
        self.ledger=RequestExecutionLedger(request_id=request_id,input_digest=identity,
            model_signature=model_signature,token_count=len(self.tokens),num_layers=num_layers,
            exact_input_proof='native_original_embedding:'+identity)
        self.layer_packet_bytes=len(flattened)*kv_heads*head_dim*4
        self.persistent_host_bytes=self.layer_packet_bytes*num_layers + self.layer_packet_bytes//2*len(selection_depths)
        # CPU packet plus bounded index/copy scratch during independent slice
        # export. Do not present persistent bytes as peak capture memory.
        self.transient_host_bytes=3*self.layer_packet_bytes
        self.required_host_bytes=self.persistent_host_bytes+self.transient_host_bytes
        self.captures={sid:SegmentCapture(self.ledger,pos,byte_budget=host_budget_bytes,
            selection_depths=selection_depths) for sid,pos in self.targets.items()}
        self.parents={}; self.events=[]; self.rejections={}; self._step=None
        self._failed=False; self._finalized=False; self._submitted_layers=0
        if self.required_host_bytes>host_budget_bytes:
            self.disable_capture('host_capture_budget_exceeded')

    def bind_parent(self, parent):
        from .source_manifest_v2 import ParentSourceMetadata
        if not isinstance(parent,ParentSourceMetadata):
            raise ValueError('typed parent provenance required; unknown ancestry cannot default to exact')
        if (parent.model_signature!=self.model_signature
                or parent.authorization_domain!=self.authorization_domain):
            raise ValueError('parent model/authorization mismatch')
        source_id=parent.source_id
        old=self.parents.get(source_id)
        if old is not None and old!=parent:
            raise ValueError('bound parent identity changed')
        self.parents[source_id]=parent

    def disable_capture(self, reason):
        for sid,capture in self.captures.items():
            capture.close(); self.rejections.setdefault(sid,reason)
        self.captures.clear()
        self._failed=True

    def abort_execution(self, reason):
        self.disable_capture(reason)
        self.events.append(dict(event='execution_failed',reason=reason))

    def prepare_step(self, session, layer, before, after):
        if self._failed:
            return
        if (session.model_signature!=self.model_signature or session.exact_prefix_tokens
                or tuple(session.token_ids)!=self.tokens or tuple(session.absolute_positions)!=self.positions):
            self.disable_capture('unverified_model_or_exact_prefix_context')
            return
        if (layer!=len(self.ledger.layers)+1 or tuple(before)!=tuple(sorted(before))
                or not set(after)<=set(before)):
            self.disable_capture('nonconsecutive_or_invalid_execution_rows')
            return
        # Removing a target token at any layer permanently cancels its capture.
        for sid in tuple(self.captures):
            if not set(self.targets[sid])<=set(after):
                self.captures.pop(sid).close(); self.rejections[sid]='target_not_full_this_layer'
        if not self.captures:
            self.disable_capture('no_complete_targets_remaining')
            return
        imports=[]
        for sid,commit in session.commits.items():
            parent=self.parents.get(commit.source_id)
            generation=parent.generation if parent is not None else None
            positions=session.committed_segment_positions.get(sid,())
            imports.extend(KVImport(p,commit.source_id,generation) for p in positions if p not in set(after))
        self._step=(layer,tuple(before),tuple(after),tuple(imports))

    def run_actual_layer(self, inner_model, arguments, call):
        if self._failed:
            return call()
        from .native_capture_hook_v2 import CurrentLayerCapture
        layer=arguments['layer']; before=tuple(arguments['active_positions'])
        after=tuple(arguments['target_active_positions'])
        if self._step is None or self._step[:3]!=(layer,before,after):
            self.disable_capture('missing_actual_execution_step')
            return call()
        attention=inner_model.layers[layer-1].self_attn
        target=tuple(sorted(p for sid in self.captures for p in self.targets[sid]))
        hook=CurrentLayerCapture(attention=attention,layer_1based=layer,
            projection_positions=before,target_positions=target,max_target_bytes=self.layer_packet_bytes)
        with hook:
            result=call()
        self._submitted_layers+=1
        if (result.hidden_states.shape[0]!=len(after) or
                result.residual is not None and result.residual.shape[0]!=len(after)):
            self.disable_capture('returned_execution_row_count_mismatch')
            return result
        packet=hook.packet_after_block()
        if packet is None:
            self.disable_capture(hook.failure or 'actual_qkv_hook_missing')
            return result
        started=time.perf_counter_ns()
        try:
            # Same-stream CPU transfer after block return fences the captured
            # layer. It is intentionally measured P0 overhead, not hidden or
            # claimed to be an asynchronous optimized production path.
            cpu_packet=replace(packet,key=packet.key.to(device='cpu',copy=True),
                               value=packet.value.to(device='cpu',copy=True))
            self.ledger.record_layer(layer_1based=layer,qkv_rows=before,
                effective_current_kv_rows=after,attention_rows=after,output_mlp_rows=after,
                imported_kv=self._step[3],completion_reference=f'target_cpu_fence:{self.request_id}:{layer}')
            for capture in self.captures.values():
                capture.record_projected_slice(cpu_packet)
        except (ValueError,MemoryError) as exc:
            self.disable_capture(type(exc).__name__+':'+str(exc))
        finally:
            self.events.append(dict(event='target_capture_after_layer',layer=layer,
                host_ms=(time.perf_counter_ns()-started)/1e6,
                projected_rows=len(before),effective_current_rows=len(after),captured_rows=len(target),
                transient_target_packet_bytes=self.layer_packet_bytes,
                capture_failure_skips_publication=True))
            self._step=None
        return result

    def finalize(self, *, request_completed, actual_token_ids):
        if self._finalized:
            raise RuntimeError('capture finalization is not repeatable')
        self._finalized=True
        if request_completed is not True or tuple(actual_token_ids)!=self.tokens:
            self.disable_capture('request_incomplete_or_input_changed')
        if self._failed or not self.captures:
            return {}
        from .source_manifest_v2 import TargetOccurrence
        try:
            occurrences=tuple(TargetOccurrence(sid,self.targets[sid][0],self.targets[sid][-1]+1)
                for sid in sorted(self.captures,key=lambda x:self.targets[x][0]))
            used_ids={ref.source_id for layer in self.ledger.layers for ref in layer.imported_kv}
            mid=self.registry.register_actual_execution(ledger=self.ledger,token_ids=self.tokens,
                absolute_positions=self.positions,authorization_domain=self.authorization_domain,
                occurrences=occurrences,parent_sources=tuple(p for sid,p in self.parents.items() if sid in used_ids),
                request_completed=True,input_reference='native-request:'+self.request_id+':'+self.ledger.input_digest)
        except (ValueError,MemoryError) as exc:
            self.disable_capture('manifest_rejected:'+str(exc));return {}
        candidates={}
        for sid,capture in tuple(self.captures.items()):
            try:
                ref=self.registry.capture_reference(mid,sid,model_signature=self.model_signature,
                    authorization_domain=self.authorization_domain,
                    target_token_ids=tuple(self.tokens[p] for p in self.targets[sid]),
                    target_positions=self.targets[sid])
                candidates[sid]=capture.finalize(ref,request_completed=True)
            except (ValueError,MemoryError) as exc:
                capture.close();self.rejections[sid]=type(exc).__name__+':'+str(exc)
        self.captures.clear()
        return candidates

    def audit(self):
        return dict(kind='actual_layer_target_capture_v2',native_runtime_qualified=False,
            layers_recorded=len(self.ledger.layers),layers_submitted=self._submitted_layers,
            target_reservations=list(self.targets),required_host_bytes=self.required_host_bytes,
            persistent_host_budget_bytes=self.persistent_host_bytes,
            transient_host_budget_bytes=self.transient_host_bytes,
            peak_target_packet_budget_bytes=self.layer_packet_bytes,events=list(self.events),
            rejected_targets=dict(self.rejections),extra_forward_count=0,
            prefix_shadow_created=False,publication_performed=False)

    def close(self):
        for capture in self.captures.values():
            capture.close()
        self.captures.clear(); self._step=None
