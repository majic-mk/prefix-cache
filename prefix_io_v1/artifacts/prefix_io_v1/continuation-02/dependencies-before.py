"""Bounded, read-only release analysis. No allocator calls or ownership mutations.

Inputs must come from an owner-published snapshot. Current tests supply CPU
fixtures; there is no live GPU generation/reference adapter yet.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import islice

@dataclass(frozen=True)
class ResourceId:
    run_id: str
    pool: str
    group: int
    block: int
    generation: int

@dataclass(frozen=True)
class Parent:
    run_id: str
    job_id: int
    total_files: int
    done_files: int
    inflight_files: int
    future_done: bool
    failed: bool
    remaining_stages: int | None
    completion_estimate_ns: int | None = None

@dataclass(frozen=True)
class Resource:
    identity: ResourceId
    bytes: int
    active_refs: int | None
    protecting_jobs: frozenset[int]
    evidence: str  # observed_blocking / candidate / unknown
    clean_evictable: bool = False

@dataclass(frozen=True)
class ReleaseAnalysis:
    resource: ResourceId
    reusable_now_bytes: int
    potential_bytes: int
    pending_parents: tuple[int, ...]
    completion_estimate_ns: int | None
    reason: str

def analyze(resource: Resource, parents: tuple[Parent, ...], *, run_id: str,
            current_generation: int) -> ReleaseAnalysis:
    """AND all parent guards; never count an early failed Future as safe."""
    def result(reason, now=0, potential=0, pending=(), estimate=None):
        return ReleaseAnalysis(resource.identity, now, potential, pending, estimate, reason)
    if resource.identity.run_id != run_id or resource.identity.generation != current_generation:
        return result("stale_identity")
    if resource.bytes <= 0 or resource.evidence not in ("observed_blocking", "candidate"):
        return result("unproved_resource")
    if resource.active_refs is None or resource.active_refs < 0:
        return result("unknown_active_refs")
    if resource.active_refs:
        return result("active_reference")
    if not resource.protecting_jobs:
        # A candidate in a free list is not a proof of allocator ownership.
        if resource.clean_evictable:
            return result("native_zero_io_reclaim", potential=resource.bytes)
        return result("no_confirmed_blocker")
    if len(parents) > 32:
        return result("parent_window_exceeded")
    if len({(p.run_id, p.job_id) for p in parents}) != len(parents):
        return result("ambiguous_parent_identity")
    known = {p.job_id: p for p in parents if p.run_id == run_id}
    if not resource.protecting_jobs.issubset(known):
        return result("incomplete_parent_closure")
    pending, estimates = [], []
    for job_id in sorted(resource.protecting_jobs):
        p = known[job_id]
        if p.failed:
            return result("failed_draining" if p.inflight_files else "failed_final")
        if (p.total_files < 1 or p.done_files < 0 or p.inflight_files < 0 or
                p.done_files > p.total_files or p.done_files + p.inflight_files > p.total_files):
            return result("inconsistent_parent")
        complete = p.future_done and p.done_files == p.total_files and p.inflight_files == 0
        if p.future_done and not complete:
            return result("inconsistent_parent")
        if not complete:
            if p.remaining_stages is None:
                return result("unknown_stage_depth")
            if not 0 <= p.remaining_stages <= 3:
                return result("stage_depth_exceeded")
            pending.append(job_id)
            if p.completion_estimate_ns is not None and p.completion_estimate_ns >= 0:
                estimates.append(p.completion_estimate_ns)
    if not pending:
        return result("all_parent_conditions_complete", now=resource.bytes)
    # Potential is a read-only prediction; candidate paths are not confirmed blockers.
    estimate = max(estimates) if len(estimates) == len(pending) else None
    return result("observed_closure" if resource.evidence == "observed_blocking" else "candidate_closure",
                  potential=resource.bytes, pending=tuple(pending), estimate=estimate)

def bounded_candidates(resources, *, limit=8):
    """Bound observation only; the native queue is never truncated or consumed."""
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("closure limit must be between 1 and 8")
    result, seen = [], set()
    for resource in islice(resources, 64):
        if resource.identity not in seen:
            seen.add(resource.identity)
            result.append(resource)
            if len(result) == limit:
                break
    return tuple(result)
