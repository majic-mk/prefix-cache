"""Policy-level reference for ProbeKV plan v2 (Python >=3.10).

No model, tensors, repository imports, GPU calls, or network access.
The execution engine must independently prove the ledger and rank inputs.
These rules do NOT bound numerical error or certify answer quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class Origin(str, Enum):
    EXACT = 'EXACT_CONTEXT'
    MIXED = 'MIXED_CONTEXT_FULL_SEGMENT'
    UNKNOWN = 'UNKNOWN_PROVENANCE'
    PARTIAL = 'PARTIALLY_REPAIRED_WORKING_KV'


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')


def _rank(value: int | None) -> None:
    if value is not None:
        _count(value, 'rank')


def inherited_rank(ranks: Iterable[int | None]) -> int | None:
    """Ordinary within-request propagation: max, never increment/reset."""
    rows = tuple(ranks)
    for item in rows:
        _rank(item)
    if any(item is None for item in rows):
        return None
    return max(rows, default=0)


def imported_rank(stored_generation: int | None, *, exact_prefix_proof: bool = False) -> int | None:
    """A cross-context reuse edge adds one. Verified exact Prefix adds none."""
    _rank(stored_generation)
    if exact_prefix_proof:
        if stored_generation != 0:
            raise ValueError('Mixed or unknown Source cannot claim exact Prefix proof')
        return 0
    return None if stored_generation is None else stored_generation + 1


@dataclass(frozen=True)
class DependencySpan:
    start: int
    end: int
    rank: int | None

    def __post_init__(self) -> None:
        _count(self.start, 'start')
        if self.end <= self.start:
            raise ValueError('Nonempty dependency span required')
        _rank(self.rank)


def prefix_rank(target_start: int, dependencies: Iterable[DependencySpan]) -> int | None:
    """Conservative prefix summary. Overlaps must be split by the caller.

    This demonstrates that suffixes cannot taint an earlier target. It is not
    a substitute for layer/token-level dependency tracking in an actual model.
    """
    _count(target_start, 'target_start')
    used = []
    for dep in dependencies:
        if dep.start >= target_start:
            continue
        if dep.end > target_start:
            raise ValueError('Split overlapping dependency spans at target boundary')
        used.append(dep.rank)
    return inherited_rank(used)


@dataclass(frozen=True)
class LayerLedger:
    layer_1based: int
    target_tokens: int
    qkv_rows: int
    attention_rows: int
    output_mlp_rows: int
    target_rows_from_old_kv: int = 0

    def __post_init__(self) -> None:
        for field in ('layer_1based','target_tokens','qkv_rows','attention_rows',
                      'output_mlp_rows','target_rows_from_old_kv'):
            _count(getattr(self, field), field)
        if self.layer_1based < 1 or self.target_tokens < 1:
            raise ValueError('Positive layer and token count required')
        if any(getattr(self, f) > self.target_tokens for f in (
            'qkv_rows','attention_rows','output_mlp_rows','target_rows_from_old_kv')):
            raise ValueError('Row count exceeds target geometry')

    @property
    def full(self) -> bool:
        return (self.qkv_rows == self.attention_rows == self.output_mlp_rows == self.target_tokens
                and self.target_rows_from_old_kv == 0)


def full_target(layers: Iterable[LayerLedger], expected_layers: int) -> bool:
    _count(expected_layers, 'expected_layers')
    if expected_layers == 0:
        raise ValueError('Model must have layers')
    rows = tuple(layers)
    if tuple(row.layer_1based for row in rows) != tuple(range(1, expected_layers + 1)):
        return False
    return len({row.target_tokens for row in rows}) == 1 and all(row.full for row in rows)


def classify_target(layers: Iterable[LayerLedger], expected_layers: int,
                    dependency_ranks: Iterable[int | None]) -> tuple[Origin, int | None]:
    if not full_target(layers, expected_layers):
        return Origin.PARTIAL, None
    generation = inherited_rank(dependency_ranks)
    if generation is None:
        return Origin.UNKNOWN, None
    return (Origin.EXACT if generation == 0 else Origin.MIXED), generation


@dataclass(frozen=True)
class Admission:
    reason: str
    stored_count: int
    eligible_count: int
    compared_count: int
    best_score: float | None
    add_threshold: float | None
    provenance: Origin
    generation: int | None
    max_generation: int = 1
    duplicate: bool = False
    scope_known: bool = True
    budget_ok: bool = True
    capacity_ok: bool = True
    permission_ok: bool = True
    artifact_complete: bool = True
    early_layers_captured: bool = True
    snapshot_current: bool = True


def decide(a: Admission) -> tuple[bool, str]:
    """Publication admission only. Inputs are asserted facts from engine/audit."""
    for name in ('stored_count','eligible_count','compared_count','max_generation'):
        _count(getattr(a, name), name)
    _rank(a.generation)
    if not 0 <= a.compared_count <= a.eligible_count <= a.stored_count:
        return False, 'invalid_candidate_counts'
    for name in ('best_score', 'add_threshold'):
        v = getattr(a, name)
        if v is not None and (isinstance(v, bool) or not math.isfinite(v) or v < 0):
            return False, 'nonfinite_or_negative_score'
    if a.provenance not in (Origin.EXACT, Origin.MIXED) or a.generation is None:
        return False, 'incomplete_or_unknown_provenance'
    if ((a.provenance == Origin.EXACT and a.generation != 0)
            or (a.provenance == Origin.MIXED and a.generation < 1)):
        return False, 'inconsistent_origin_and_generation'
    if a.generation > a.max_generation:
        return False, 'generation_limit'
    for field in ('scope_known','budget_ok','capacity_ok','permission_ok',
                  'artifact_complete','early_layers_captured','snapshot_current'):
        if not getattr(a, field):
            return False, field
    if a.duplicate:
        return False, 'duplicate'
    if a.reason == 'content_miss':
        return (True, 'eligible') if a.stored_count == 0 else (False, 'not_a_content_miss')
    if a.reason != 'complete_scope_mismatch':
        return False, 'not_a_growth_trigger'
    if a.eligible_count == 0 or a.compared_count != a.eligible_count:
        return False, 'no_complete_eligible_comparison'
    # A filtered logical pool cannot be called entirely incompatible.
    # This reference chooses the conservative v2 no-growth behavior.
    if a.eligible_count != a.stored_count:
        return False, 'filtered_pool_not_a_full_stored_mismatch'
    if a.best_score is None or a.add_threshold is None:
        return False, 'missing_threshold_evidence'
    if a.best_score <= a.add_threshold:
        return False, 'already_covered_by_rule'
    return True, 'eligible'


def visible(*, birth_request_id: str, publication_epoch: int,
            reading_request_id: str, reader_snapshot_epoch: int,
            completed_publication: bool) -> bool:
    _count(publication_epoch, 'publication_epoch')
    _count(reader_snapshot_epoch, 'reader_snapshot_epoch')
    if not birth_request_id or not reading_request_id:
        raise ValueError('Request IDs required')
    return (completed_publication and birth_request_id != reading_request_id
            and publication_epoch <= reader_snapshot_epoch)
