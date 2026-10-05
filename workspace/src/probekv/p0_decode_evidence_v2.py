"""Actual decode-input records for P0; not a model/provenance certificate.

Teacher logits can only be compared at identical conditioning prefixes. Keeping
argmax outputs alone cannot prove that: teacher inputs may differ from argmax.
This recorder has no tensors, model calls, default tolerance or GPU authority.
"""
from __future__ import annotations

from .source_manifest_v2 import request_input_digest
from .v8_schema10_execution import digest_json


def _tokens(values, name, *, allow_empty=False):
    if not isinstance(values, (list, tuple)) or (not values and not allow_empty):
        raise ValueError(name + ' requires explicit token IDs')
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError(name + ' requires nonnegative integer token IDs')
    return tuple(values)


class DecodeInputRecorderV2:
    """Record a feed only AFTER its native decode step completed successfully."""

    def __init__(self, request, *, cached_prefix_tokens):
        self.prompt = _tokens(request['token_ids'], 'prompt')
        self.count = request['max_new_tokens']
        if type(self.count) is not int or self.count < 1:
            raise ValueError('explicit positive decode bound required')
        if type(cached_prefix_tokens) is not int or not 0 <= cached_prefix_tokens <= len(self.prompt):
            raise ValueError('actual Prefix count outside prompt')
        self.prefix = cached_prefix_tokens
        self.teacher = ('teacher_token_ids' in request)
        self.expected = (_tokens(request['teacher_token_ids'], 'teacher', allow_empty=True)
                         if self.teacher else ())
        if self.teacher and len(self.expected) != self.count - 1:
            raise ValueError('teacher must contain exactly bound-1 feed tokens')
        self.predicted, self.fed = [], []
        self.closed = False

    def append(self, predicted_token, *, fed_token=None):
        if self.closed or len(self.predicted) >= self.count:
            raise ValueError('decode trace closed or exceeds bound')
        _tokens((predicted_token,), 'predicted')
        if not self.predicted:
            if fed_token is not None:
                raise ValueError('first logit comes from prefill, not a decode feed')
        else:
            _tokens((fed_token,), 'actual decode feed')
            expected = self.expected[len(self.fed)] if self.teacher else self.predicted[-1]
            if fed_token != expected:
                raise ValueError('actual decode feed differs from declared teacher/greedy path')
            self.fed.append(fed_token)
        self.predicted.append(predicted_token)

    def finish(self, *, logit_rows):
        if self.closed or type(logit_rows) is not int or logit_rows != len(self.predicted) or not logit_rows:
            raise ValueError('actual raw logit count must equal completed decode trace')
        if self.teacher and logit_rows != self.count:
            raise ValueError('incomplete teacher sequence cannot qualify')
        self.closed = True
        result = dict(kind='actual_decode_inputs_v2',
            input_digest=request_input_digest(self.prompt, tuple(range(len(self.prompt)))),
            prompt_tokens=len(self.prompt), cached_prefix_tokens=self.prefix,
            mode='teacher_forced' if self.teacher else 'greedy',
            max_new_tokens=self.count, fed_token_ids=list(self.fed),
            predicted_token_ids=list(self.predicted), logit_rows=logit_rows,
            logit_absolute_positions=list(range(len(self.prompt)-1, len(self.prompt)-1+logit_rows)))
        result['trace_sha256'] = digest_json(result)
        return result


def validate_decode_trace(trace, *, request, answer, logit_rows):
    """Reconstruct, rather than trusting a claimed matched-teacher flag."""
    if type(trace) is not dict:
        raise ValueError('actual decode-input trace missing')
    recorder = DecodeInputRecorderV2(request, cached_prefix_tokens=trace['cached_prefix_tokens'])
    predicted = _tokens(trace['predicted_token_ids'], 'recorded predictions')
    feeds = _tokens(trace['fed_token_ids'], 'recorded feeds', allow_empty=True)
    if len(feeds) != len(predicted)-1 or list(predicted) != answer.get('token_ids'):
        raise ValueError('decode trace differs from raw answer or feed count')
    for i, token in enumerate(predicted):
        recorder.append(token, fed_token=feeds[i-1] if i else None)
    rebuilt = recorder.finish(logit_rows=logit_rows)
    if rebuilt != trace:
        raise ValueError('decode trace digest/positions/recipe mismatch')
    return rebuilt


def assert_same_logit_conditioning(reference, candidate):
    # Predictions need not match for teacher-forced logits. Feeds MUST match.
    fields = ('input_digest', 'prompt_tokens', 'cached_prefix_tokens', 'mode',
              'max_new_tokens', 'fed_token_ids', 'logit_rows', 'logit_absolute_positions')
    if any(reference[k] != candidate[k] for k in fields):
        raise ValueError('logits have different conditioning inputs or positions')
    return True
