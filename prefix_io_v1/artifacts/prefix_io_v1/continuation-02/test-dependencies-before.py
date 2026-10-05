from dataclasses import replace
import pytest
from prefix_io_control.dependencies import Parent, Resource, ResourceId, analyze, bounded_candidates

ID = ResourceId("run", "gpu", 0, 9, 7)
R = Resource(ID, 4096, 0, frozenset({1, 2}), "observed_blocking")
P1 = Parent("run", 1, 5, 5, 0, True, False, 0)
P2 = Parent("run", 2, 8, 8, 0, True, False, 0)

def run(resource=R, parents=(P1, P2), generation=7):
    return analyze(resource, parents, run_id="run", current_generation=generation)

def test_all_parent_conditions_and_deduplicated_resource():
    assert run().reusable_now_bytes == 4096
    assert bounded_candidates(iter([R, R])) == (R,)

@pytest.mark.parametrize("refs", [1, 5, None, -1])
def test_active_or_unknown_refs_never_release(refs):
    assert run(replace(R, active_refs=refs)).reusable_now_bytes == 0
    assert run(replace(R, active_refs=refs)).potential_bytes == 0

def test_child_or_d2h_completion_is_not_parent_completion():
    pending = replace(P2, done_files=7, inflight_files=1, future_done=False, remaining_stages=1)
    result = run(parents=(P1, pending))
    assert result.reusable_now_bytes == 0
    assert result.pending_parents == (2,)
    assert result.completion_estimate_ns is None

def test_more_than_three_files_allowed_but_stage_depth_is_bounded():
    pending = replace(P2, done_files=0, inflight_files=1, future_done=False, remaining_stages=3)
    assert run(parents=(P1, pending)).potential_bytes == 4096
    assert run(parents=(P1, replace(pending, remaining_stages=4))).potential_bytes == 0

def test_incomplete_multi_guard_closure_has_no_benefit():
    assert run(parents=(P1,)).potential_bytes == 0

def test_old_generation_and_old_run_fail_closed():
    assert run(generation=8).reason == "stale_identity"
    assert run(replace(R, identity=replace(ID, run_id="previous"))).potential_bytes == 0
    assert run(parents=(P1, replace(P2, run_id="previous"))).potential_bytes == 0

@pytest.mark.parametrize("inflight,reason", [(1, "failed_draining"), (0, "failed_final")])
def test_failed_future_never_releases(inflight, reason):
    failed = replace(P2, done_files=6, inflight_files=inflight, failed=True)
    result = run(parents=(P1, failed))
    assert result.reason == reason
    assert result.reusable_now_bytes == result.potential_bytes == 0

def test_future_done_with_unfinished_children_is_inconsistent():
    assert run(parents=(P1, replace(P2, done_files=7, inflight_files=1))).reason == "inconsistent_parent"

def test_native_clean_reclaim_is_zero_io_opportunity():
    r = replace(R, protecting_jobs=frozenset(), clean_evictable=True)
    result = run(r, ())
    assert result.reason == "native_zero_io_reclaim"
    assert result.pending_parents == () and result.reusable_now_bytes == 0

def test_candidate_is_not_confirmed_blocker():
    p = replace(P2, done_files=0, inflight_files=1, future_done=False, remaining_stages=2,
                completion_estimate_ns=100)
    result = run(replace(R, evidence="candidate"), (P1, p))
    assert result.reason == "candidate_closure"
    assert result.completion_estimate_ns == 100

def test_no_unbounded_scan_and_no_discard_of_native_window_remainder():
    items = iter(replace(R, identity=replace(ID, block=i)) for i in range(100))
    observed = bounded_candidates(items)
    assert len(observed) == 8
    assert next(items).identity.block == 8

def test_unknown_evidence_and_duplicate_parent_ids_fail_closed():
    assert run(replace(R, evidence="unknown")).potential_bytes == 0
    assert run(parents=(P1, P1, P2)).reason == "ambiguous_parent_identity"

def test_unknown_depth_is_not_zero_or_an_eligible_release_closure():
    pending = replace(P2, done_files=7, inflight_files=1, future_done=False, remaining_stages=None)
    result = run(parents=(P1, pending))
    assert result.reason == "unknown_stage_depth" and result.potential_bytes == 0
