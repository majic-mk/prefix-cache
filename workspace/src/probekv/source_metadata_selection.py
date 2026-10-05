"""Optional structural tie-break for state-compatible Sources, never CFO.

Only immutable token-identity metadata is retained. No KV/attention reads,
candidate pruning, probability interpretation or preparation authorization.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Optional, Tuple


class MetadataError(ValueError):
    """Contradictory metadata, distinct from unavailable metadata."""


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise MetadataError(name + ' must be a nonempty string')


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise MetadataError(name + ' must be an integer in range')


def _scalar(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise MetadataError(name + ' must be finite and nonnegative')


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class PrefixOccurrence:
    content_key: str
    role: str
    token_count: int
    order_index: int
    token_start: int
    token_end: int

    def __post_init__(self):
        _text(self.content_key, 'content_key')
        _text(self.role, 'role')
        if self.role.lower() in ('padding', 'target', 'suffix'):
            raise MetadataError('padding/target/suffix cannot be prefix occurrences')
        _integer(self.token_count, 'token_count', 1)
        _integer(self.order_index, 'order_index')
        _integer(self.token_start, 'token_start')
        _integer(self.token_end, 'token_end', 1)
        if self.token_end - self.token_start != self.token_count:
            raise MetadataError('token length and half-open interval disagree')


@dataclass(frozen=True)
class PrefixManifest:
    tokenizer_signature: str
    target_start: int
    occurrences: Tuple[PrefixOccurrence, ...]
    complete: bool = True

    def __post_init__(self):
        _text(self.tokenizer_signature, 'tokenizer_signature')
        _integer(self.target_start, 'target_start')
        if type(self.complete) is not bool or type(self.occurrences) is not tuple:
            raise MetadataError('immutable occurrences and explicit completeness required')
        end, order, lengths = 0, -1, {}
        for row in self.occurrences:
            if type(row) is not PrefixOccurrence:
                raise MetadataError('typed prefix occurrences required')
            if row.order_index <= order or row.token_start < end:
                raise MetadataError('overlapping intervals or contradictory order')
            if row.token_end > self.target_start:
                raise MetadataError('prefix includes target or subsequent tokens')
            if row.content_key in lengths and lengths[row.content_key] != row.token_count:
                raise MetadataError('one exact content identity has inconsistent lengths')
            lengths[row.content_key] = row.token_count
            end, order = row.token_end, row.order_index

    @property
    def fully_covered(self):
        end = 0
        for row in self.occurrences:
            if row.token_start != end:
                return False
            end = row.token_end
        return self.complete and end == self.target_start

    @property
    def manifest_id(self):
        return 'prefix-structure-v1:' + _digest(asdict(self))


def token_content_key(tokens, tokenizer_signature):
    _text(tokenizer_signature, 'tokenizer_signature')
    values = tuple(tokens)
    for token in values:
        _integer(token, 'token_id')
    return _digest(dict(tokens=values, tokenizer=tokenizer_signature))


def build_prefix_manifest(token_ids, *, target_start, role_spans, tokenizer_signature,
                          complete=True):
    """Build once from renderer-owned (start, end, role) spans, never guess roles.

    All prefix rows, including separators/system/instructions, need ownership.
    Missing spans remain invalid; they are not silently converted to documents.
    """
    _integer(target_start, 'target_start')
    if target_start > len(token_ids):
        raise MetadataError('target starts beyond request')
    rows = []
    for order, (start, end, role) in enumerate(role_spans):
        _integer(start, 'span start')
        _integer(end, 'span end', 1)
        if end <= start or end > target_start:
            raise MetadataError('invalid prefix span')
        rows.append(PrefixOccurrence(token_content_key(token_ids[start:end], tokenizer_signature),
            role, end-start, order, start, end))
    return PrefixManifest(tokenizer_signature, target_start, tuple(rows), complete)


def validate_prefix_tokens(manifest, token_ids, *, target_start, tokenizer_signature):
    """Check against existing input manifests, not historical prefix KV."""
    if manifest is None:
        return
    if type(manifest) is not PrefixManifest:
        raise MetadataError('typed prefix manifest required')
    if (manifest.target_start != target_start or target_start > len(token_ids)
            or manifest.tokenizer_signature != tokenizer_signature):
        raise MetadataError('prefix manifest target/tokenizer identity mismatch')
    for row in manifest.occurrences:
        if row.content_key != token_content_key(token_ids[row.token_start:row.token_end], tokenizer_signature):
            raise MetadataError('prefix content identity differs from actual input tokens')


class SharedPrefixManifests:
    """Bounded optional sidecar; many Sources may reference the same manifest ID.

    Keeps metadata only. Persistence can serialize to_payload(); readers verify
    its content-addressed reference. Missing references resolve to None.
    """
    def __init__(self, *, max_bytes):
        _integer(max_bytes, 'metadata byte budget', 1)
        self.max_bytes, self.bytes_used, self._rows = max_bytes, 0, {}

    def register(self, manifest):
        if type(manifest) is not PrefixManifest:
            raise MetadataError('typed prefix manifest required')
        key = manifest.manifest_id
        if key not in self._rows:
            size = len(json.dumps(asdict(manifest), sort_keys=True, separators=(',', ':')).encode('utf-8'))
            if self.bytes_used + size > self.max_bytes:
                raise MemoryError('shared prefix metadata budget exhausted')
            self._rows[key] = manifest
            self.bytes_used += size
        return key

    def resolve(self, reference):
        return self._rows.get(reference)

    def to_payload(self, reference):
        row = self.resolve(reference)
        return None if row is None else dict(manifest_id=reference, payload=asdict(row))

    def import_payload(self, envelope):
        try:
            payload = dict(envelope['payload'])
            payload['occurrences'] = tuple(PrefixOccurrence(**r) for r in payload['occurrences'])
            manifest = PrefixManifest(**payload)
            if manifest.manifest_id != envelope['manifest_id']:
                raise MetadataError('prefix manifest digest mismatch')
        except (TypeError, KeyError) as exc:
            raise MetadataError('invalid prefix manifest envelope') from exc
        return self.register(manifest)


@dataclass(frozen=True)
class MetadataScore:
    valid: bool
    reason: str
    L_old: Optional[int] = None
    L_new: Optional[int] = None
    W_match: Optional[int] = None
    matched_count: Optional[int] = None
    overlap: Optional[float] = None
    order_penalty: Optional[float] = None
    metadata_score: Optional[float] = None
    both_prefixes_empty: bool = False


def _role_map(groups):
    if type(groups) is not tuple:
        raise MetadataError('role compatibility groups must be immutable')
    result = {}
    for group in groups:
        if type(group) is not tuple or len(group) < 2:
            raise MetadataError('each role compatibility group needs at least two roles')
        for role in group:
            _text(role, 'compatible role')
        for role in group:
            if role in result:
                raise MetadataError('overlapping role compatibility groups')
            result[role] = min(group)
    return result


def score_prefixes(old, new, *, compatible_role_groups=()):
    roles = _role_map(compatible_role_groups)
    for prefix in (old, new):
        if prefix is not None and type(prefix) is not PrefixManifest:
            raise MetadataError('typed prefix manifest required')
    if old is None or new is None:
        return MetadataScore(False, 'METADATA_MISSING')
    if old.tokenizer_signature != new.tokenizer_signature:
        raise MetadataError('prefix tokenizer namespaces differ')
    lengths = {}
    for row in old.occurrences + new.occurrences:
        if row.content_key in lengths and lengths[row.content_key] != row.token_count:
            raise MetadataError('exact content identity length mismatch across prefixes')
        lengths[row.content_key] = row.token_count
    if not old.fully_covered or not new.fully_covered:
        return MetadataScore(False, 'METADATA_INCOMPLETE')
    left, right = old.target_start, new.target_start
    if left == right == 0:
        return MetadataScore(True, 'BOTH_PREFIXES_EMPTY', 0, 0, 0, 0, 1., 0., 1., True)
    queues = defaultdict(deque)
    for row in new.occurrences:
        queues[(row.content_key, roles.get(row.role, row.role))].append(row)
    matched = []
    for row in old.occurrences:
        queue = queues[(row.content_key, roles.get(row.role, row.role))]
        if queue:
            match = queue.popleft()
            matched.append((row.token_count, match.order_index))
    weight = sum(n for n, _ in matched)
    # Weighted inversions, O(m log m); integer accumulation avoids pair rounding.
    ranks = {p: i+1 for i, p in enumerate(sorted(p for _, p in matched))}
    tree = [0] * (len(matched)+1)
    seen = pairs = inversions = 0
    for n, position in matched:
        rank = ranks[position]
        index, before = rank, 0
        while index:
            before += tree[index]
            index -= index & -index
        inversions += n * (seen-before)
        pairs += n * seen
        index = rank
        while index < len(tree):
            tree[index] += n
            index += index & -index
        seen += n
    overlap = weight / (left+right-weight)
    penalty = inversions / pairs if pairs else 0.
    reason = 'MATCHED' if matched else 'NO_COMMON_OCCURRENCE'
    return MetadataScore(True, reason, left, right, weight, len(matched),
                         overlap, penalty, overlap*(1-penalty))


@dataclass(frozen=True)
class MetadataSelectionPolicy:
    enabled: bool = False
    tau: Optional[float] = None
    delta: Optional[float] = None
    compatible_role_groups: tuple = ()

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise MetadataError('enabled must be boolean')
        for name in ('tau', 'delta'):
            value = getattr(self, name)
            if value is not None or self.enabled:
                _scalar(value, name)
        _role_map(self.compatible_role_groups)


@dataclass(frozen=True)
class StateCandidate:
    source_id: str
    state_difference: float
    admission_eligible: bool = True

    def __post_init__(self):
        _text(self.source_id, 'source_id')
        _scalar(self.state_difference, 'state_difference')
        if type(self.admission_eligible) is not bool:
            raise MetadataError('admission eligibility must be boolean')


@dataclass(frozen=True)
class MetadataSelectionDecision:
    policy: MetadataSelectionPolicy
    original_source_id: Optional[str]
    selected_source_id: Optional[str]
    reason: str
    band_source_ids: tuple
    scores: tuple = ()
    prefix_references: tuple = ()
    # Ranking never authorizes transfer, lease, freeze or reuse commit.
    requires_selected_source_admission: bool = True


def select_source(candidates, *, policy, current_prefix=None, historical_prefixes=None):
    """Input order is the original state selector's deterministic tie ordering.

    Other established preconditions are supplied via admission_eligible; the
    chosen Source still needs its OWN runtime cost/resource checks afterwards.
    """
    if type(policy) is not MetadataSelectionPolicy:
        raise MetadataError('explicit typed metadata selection policy required')
    rows = tuple(candidates)
    if any(type(row) is not StateCandidate for row in rows):
        raise MetadataError('typed state candidates required')
    if len({r.source_id for r in rows}) != len(rows):
        raise MetadataError('duplicate Source candidate')
    if any(a.state_difference > b.state_difference for a, b in zip(rows, rows[1:])):
        raise MetadataError('candidates must retain original ascending state order')
    eligible = tuple(r for r in rows if r.admission_eligible and
                     (not policy.enabled or r.state_difference <= policy.tau))
    original = eligible[0].source_id if eligible else None
    if not policy.enabled:
        return MetadataSelectionDecision(policy, original, original, 'DISABLED', ())
    if not eligible:
        return MetadataSelectionDecision(policy, None, None, 'NO_COMPATIBLE_CANDIDATE', ())
    best = eligible[0].state_difference
    band = tuple(r for r in eligible if r.state_difference <= best+policy.delta)
    ids = tuple(r.source_id for r in band)
    if len(band) == 1:
        return MetadataSelectionDecision(policy, original, original, 'SINGLE_NEAR_CANDIDATE', ids)
    if historical_prefixes is not None and not isinstance(historical_prefixes, Mapping):
        raise MetadataError('historical prefixes must be a Source-to-manifest mapping')
    historical_prefixes = historical_prefixes if historical_prefixes is not None else {}
    scores, refs = [], []
    for row in band:
        old = historical_prefixes.get(row.source_id)
        score = score_prefixes(old, current_prefix, compatible_role_groups=policy.compatible_role_groups)
        scores.append((row.source_id, score))
        refs.append((row.source_id, old.manifest_id if old is not None else None))
    refs.append(('current_prefix', current_prefix.manifest_id if current_prefix is not None else None))
    if any(not score.valid for _, score in scores):
        return MetadataSelectionDecision(policy, original, original, 'BAND_METADATA_UNAVAILABLE',
                                         ids, tuple(scores), tuple(refs))
    # max is stable on ties: preserve the original state selector ordering.
    winner = max(scores, key=lambda item: item[1].metadata_score)[0]
    return MetadataSelectionDecision(policy, original, winner, 'METADATA_NEAR_STATE_RERANK',
                                     ids, tuple(scores), tuple(refs))
