"""Restrictive provenance for an independent P0 mixed-reference execution.

This marker grants no execution, Source, or qualification authority. It only
prevents the native sampling endpoint and publication APIs from mistaking an
explicit diagnostic mixed prefill for exact dense computation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .source_manifest_v2 import request_input_digest


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + ' must be nonempty text')
    return value


def _sha(value, name):
    if (type(value) is not str or len(value) != 64
            or any(ch not in '0123456789abcdef' for ch in value)):
        raise ValueError(name + ' must be lowercase SHA256')
    return value


@dataclass(frozen=True)
class P0DiagnosticContextV2:
    request_id: str
    input_digest: str
    model_signature: str
    recipe_sha256: str
    kind: str = 'explicit_mixed_reference'

    def __post_init__(self):
        _text(self.request_id, 'request_id')
        _sha(self.input_digest, 'input_digest')
        _text(self.model_signature, 'model_signature')
        _sha(self.recipe_sha256, 'recipe_sha256')
        if self.kind not in ('explicit_mixed_reference', 'mixed_sparse_control'):
            raise ValueError('unsupported P0 diagnostic context kind')

    def audit(self):
        return {**asdict(self), 'grants_execution_authority': False,
                'exact_prefix_publication_allowed': False,
                'source_publication_allowed': False,
                'native_gpu_qualified': False}


def diagnostic_context_v2(context):
    """Validate a present marker before any endpoint or publication operation."""
    marker = getattr(context, 'p0_diagnostic_v2', None)
    if marker is None:
        return None
    if type(marker) is not P0DiagnosticContextV2:
        raise ValueError('typed P0 diagnostic context required')
    marker.__post_init__()
    tokens = tuple(context.request['token_ids'])
    positions = tuple(range(len(tokens)))
    if (context.request['request_id'] != marker.request_id
            or request_input_digest(tokens, positions) != marker.input_digest
            or context.adapter.provenance['model_signature'] != marker.model_signature
            or context.cached_prefix_tokens != 0):
        raise ValueError('stale P0 diagnostic context identity or Prefix state')
    return marker


def bind_p0_diagnostic_context(context, *, recipe_sha256,
                               kind='explicit_mixed_reference'):
    """Bind only a fresh, uncaptured native context; never accept request flags."""
    from .v8_schema10_native_adapter import NativeRequestContext
    if type(context) is not NativeRequestContext:
        raise ValueError('fresh NativeRequestContext required')
    if (context.finished or context.closed or context.current_completed_depth != 0
            or context.engine is not None or context.cached_prefix_tokens
            or getattr(context, 'p0_diagnostic_v2', None) is not None
            or getattr(context, 'p0_diagnostic_resources_v2', None) is not None
            or getattr(context, 'probe_fallback_reason', None)
            or context.prepared or context.frozen or context.committed
            or getattr(context, 'source_capture_v2', None) is not None
            or getattr(context, 'source_consumption_v2', None) is not None
            or getattr(context, 'target_candidates_v2', {})
            or getattr(context, 'canonical_exports', {})
            or getattr(context, 'capture_reservation', None) is not None
            or getattr(context, 'capture_collector', None) is not None
            or context.request.get('capture_original_full_prefill')
            or context.request.get('publish_exact_prefix_shadow')):
        raise ValueError('P0 diagnostic binding requires fresh no-Source/no-capture/no-Prefix context')
    tokens = tuple(context.request['token_ids'])
    marker = P0DiagnosticContextV2(context.request['request_id'],
        request_input_digest(tokens, tuple(range(len(tokens)))),
        context.adapter.provenance['model_signature'], recipe_sha256, kind)
    context.p0_diagnostic_v2 = marker
    return marker


def reject_diagnostic_publication_v2(context):
    if diagnostic_context_v2(context) is not None:
        raise ValueError('P0 diagnostic mixed reference cannot capture or publish Sources')
