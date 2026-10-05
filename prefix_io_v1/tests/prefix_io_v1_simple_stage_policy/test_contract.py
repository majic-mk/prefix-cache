"""CPU-only contracts. No model, CUDA, backend calls, payload or resource queue."""
from concurrent.futures import Future
from dataclasses import FrozenInstanceError, replace
from threading import Thread

import pytest

from prefix_io_control.dispatch_budget import Amount, STAGES, ZERO
from prefix_io_control.dispatch_shadow import ShadowState
from prefix_io_control.simple_stage_policy import (
    DispatchController, SimpleStageConfig, make_dispatch_controller,
)


def config(**changes):
    base = SimpleStageConfig(
        mode="fixed", epoch_ns=1_000_000, byte_quantum=16,
        cumulative=(Amount(4, 256),) * 4,
        inflight=(Amount(4, 256),) * 4,
        shared_ssd_cumulative=Amount(4, 256),
        shared_ssd_inflight=Amount(4, 256),
        shared_copy_cumulative_bytes=256, shared_copy_inflight_bytes=256,
        reserve_staging_bytes=0, max_accepted_parents=2,
        sample_max_age_ns=100_000, max_wait_ns=200_000)
    return replace(base, **changes)


def controller(**changes):
    c = DispatchController("run", config(**changes))
    c.bind()
    return c


def state(now=10, **changes):
    values = dict(run_id="run", captured_ns=now, inflight=ZERO,
                  free_staging_bytes=512, accepted_parents=0, native_issue_safe=True)
    values.update(changes)
    return ShadowState(**values)


def decide(c, stage="ssd_read", size=16, now=10, key=1, sample=None, **kwargs):
    return c.decide(stage, size, state(now) if sample is None else sample,
                    now_ns=now, work_id=key, **kwargs)


def test_off_factory_has_no_owner_or_state():
    assert make_dispatch_controller(None, config(mode="off")) is None
    with pytest.raises(ValueError, match="factory"):
        DispatchController("run", config(mode="off"))


def test_config_frozen_and_p4_modes_closed():
    c = config()
    with pytest.raises(FrozenInstanceError):
        c.mode = "joint"
    for mode in ("interference", "dependency_only", "joint"):
        with pytest.raises(ValueError):
            config(mode=mode)


@pytest.mark.parametrize("changes", [
    {"epoch_ns": 2_000_000}, {"byte_quantum": True},
    {"sample_max_age_ns": 0}, {"max_wait_ns": 0},
    {"max_waiting_keys": 65}, {"max_records": 97},
    {"max_accepted_parents": 0}, {"max_waiting_keys": False},
    {"cumulative": [Amount(1, 16)] * 4}, {"inflight": (Amount(1, 16),)},
    {"shared_copy_cumulative_bytes": 17}, {"shared_ssd_inflight": Amount(1, 17)},
    {"reserve_staging_bytes": 16},
])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_bind_does_not_claim_owner_and_is_exclusive():
    c = DispatchController("run", config())
    c.bind()
    assert c.owner is None
    with pytest.raises(ValueError):
        c.bind()
    c.refresh_epoch(10)
    assert c.owner is not None


def test_unbound_and_foreign_owner_rejected():
    c = DispatchController("run", config())
    with pytest.raises(RuntimeError):
        c.refresh_epoch(10)
    c.bind()
    c.refresh_epoch(10)
    errors = []
    def other():
        try:
            c.refresh_epoch(11)
        except Exception as exc:
            errors.append(exc)
    t = Thread(target=other)
    t.start()
    t.join()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert c.epoch_refreshes == 1


def test_wrong_run_does_not_publish_or_consume():
    c = controller()
    with pytest.raises(ValueError, match="identity"):
        decide(c, sample=state(run_id="other"))
    assert c.grant is None and c.used == list(ZERO)
    assert c.owner is None


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_charged_at_api_return(stage):
    c = controller()
    d = decide(c, stage=stage, size=32)
    assert d.action == "issue"
    assert c.used == list(ZERO)
    c.accepted(d.attempt)
    assert c.used[STAGES.index(stage)] == Amount(1, 32)
    assert c.snapshot()["observed_api_accepted"][stage] == {"ops": 1, "bytes": 32}


def test_same_epoch_never_replenishes_allowance():
    c = controller(cumulative=(Amount(1, 32),) * 4)
    d = decide(c, size=32)
    c.accepted(d.attempt)
    assert not c.refresh_epoch(20)
    d = decide(c, size=16, now=21, key=2)
    assert d.action == "defer" and d.attempt is None
    assert c.used[0] == Amount(1, 32)
    assert c.epoch_refreshes == 1


def test_epoch_refresh_is_owner_clock_independent_of_worker():
    c = controller(cumulative=(Amount(1, 16),) * 4)
    first = decide(c)
    c.accepted(first.attempt)
    later = decide(c, now=3_000_010, key=2)
    assert later.action == "issue" and later.attempt.epoch == 3
    assert c.epoch_refreshes == 2 and c.used == list(ZERO)


def test_clock_regression_never_replenishes():
    c = controller()
    first = decide(c, now=20)
    c.accepted(first.attempt)
    with pytest.raises(ValueError, match="regressed"):
        c.refresh_epoch(19)
    assert c.used[0] == Amount(1, 16) and c.epoch_refreshes == 1


def test_pending_cannot_cross_epoch_and_late_return_charges_old_epoch():
    c = controller()
    d = decide(c)
    with pytest.raises(RuntimeError, match="settle"):
        c.refresh_epoch(1_000_010)
    c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 16)
    assert c.grant.epoch == 0
    assert c.refresh_epoch(1_000_011)
    assert c.used == list(ZERO)


def test_immediate_attempt_cannot_nest_or_settle_twice_or_foreign():
    a, b = controller(), controller()
    da, db = decide(a), decide(b)
    with pytest.raises(RuntimeError):
        decide(a, key=2)
    with pytest.raises(ValueError):
        a.accepted(db.attempt)
    a.accepted(da.attempt)
    with pytest.raises(ValueError):
        a.accepted(da.attempt)
    b.rejected(db.attempt, proved=True)


def test_rejection_requires_proof_and_does_not_charge():
    c = controller()
    d = decide(c)
    with pytest.raises(ValueError, match="proof"):
        c.rejected(d.attempt, "CUDA exception")
    assert c.pending is d.attempt
    c.rejected(d.attempt, "validated pre-acceptance failure", proved=True)
    assert c.used == list(ZERO) and c.rejected_ops == 1


def test_shared_ssd_read_write_allowance_not_two_independent_grants():
    c = controller(shared_ssd_cumulative=Amount(1, 32))
    d = decide(c, size=16)
    c.accepted(d.attempt)
    d = decide(c, "ssd_write", 16, now=11, key=2)
    assert d.action == "defer" and "shared_ssd_epoch_ops" in d.reasons


def test_shared_copy_h2d_d2h_bytes():
    c = controller(shared_copy_cumulative_bytes=32)
    d = decide(c, "h2d", 32)
    c.accepted(d.attempt)
    d = decide(c, "d2h", 16, now=11, key=2)
    assert d.action == "defer" and "shared_copy_epoch_bytes" in d.reasons


def test_fused_launch_is_one_op_with_actual_total_bytes():
    c = controller()
    d = decide(c, "h2d", size=17 + 31)
    c.accepted(d.attempt)
    assert c.used[2] == Amount(1, 48)
    # Byte caps are quantized; actual mapping sizes need not be.
    d = decide(c, "d2h", size=17, now=11, key=2)
    c.accepted(d.attempt)
    assert c.used[3] == Amount(1, 17)


def test_one_shared_physical_read_then_each_destination_copy_charged():
    c = controller()
    for stage, size, key in (("ssd_read", 32, 1), ("h2d", 16, 2), ("h2d", 16, 3)):
        d = decide(c, stage, size, now=10 + key, key=key,
                   progress="continuation" if stage == "h2d" else None)
        c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 32) and c.used[2] == Amount(2, 32)


@pytest.mark.parametrize("stage,index", [(s, i) for i, s in enumerate(STAGES)])
def test_inflight_is_physical_state_not_cumulative_issued(stage, index):
    c = controller(inflight=(Amount(1, 32),) * 4)
    active = list(ZERO)
    active[index] = Amount(1, 16)
    d = decide(c, stage, 16, sample=state(inflight=tuple(active)))
    assert d.action == "defer" and "stage_inflight_ops" in d.reasons
    assert c.used == list(ZERO)
    d = decide(c, stage, 16, now=11, sample=state(11))
    assert d.action == "issue"


def test_shared_ssd_inflight_combines_read_and_write():
    c = controller(shared_ssd_inflight=Amount(1, 64))
    active = (Amount(1, 32), Amount(), Amount(), Amount())
    d = decide(c, "ssd_write", sample=state(inflight=active))
    assert d.action == "defer" and "shared_ssd_inflight_ops" in d.reasons


def test_shared_copy_inflight_combines_directions():
    c = controller(shared_copy_inflight_bytes=32)
    active = (Amount(), Amount(), Amount(1, 32), Amount())
    d = decide(c, "d2h", sample=state(inflight=active))
    assert d.action == "defer" and "shared_copy_inflight_bytes" in d.reasons


@pytest.mark.parametrize("progress", ["mandatory", "mandatory_support", "continuation", "age", "shutdown"])
def test_progress_exceeds_only_performance_limits(progress):
    c = controller(cumulative=ZERO, inflight=ZERO,
                   shared_ssd_cumulative=Amount(), shared_ssd_inflight=Amount(),
                   shared_copy_cumulative_bytes=0, shared_copy_inflight_bytes=0)
    d = decide(c, progress=progress)
    assert d.action == "issue" and d.reasons
    c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 16) and c.performance_override_ops == 1
    d = decide(c, now=11, key=2, sample=state(11, native_issue_safe=False), progress=progress)
    assert d.action == "defer" and d.attempt is None


def test_owner_age_progress_without_scheduler_grant():
    c = controller(cumulative=ZERO)
    d = decide(c, now=10)
    assert d.action == "defer"
    d = decide(c, now=200_010)
    assert d.action == "issue" and d.attempt.progress == "age"
    c.accepted(d.attempt)
    assert c.progress_totals["age"] == 1


def test_pressure_reserve_only_new_staging_not_free_slot_zero_continuation():
    c = controller(mode="pressure", reserve_staging_bytes=32)
    d = decide(c, sample=state(free_staging_bytes=32), staging_bytes_needed=16)
    assert d.action == "defer" and "staging_reserve" in d.reasons
    d = decide(c, now=11, sample=state(11, free_staging_bytes=0),
               staging_bytes_needed=0, progress="continuation")
    assert d.action == "issue"
    c.accepted(d.attempt)
    assert c.snapshot()["resource_release_inferred"] is False


def test_native_capacity_is_never_overridden():
    c = controller(mode="pressure", reserve_staging_bytes=16)
    d = decide(c, sample=state(free_staging_bytes=0),
               staging_bytes_needed=16, progress="mandatory")
    assert d.action == "defer" and "native_capacity_or_dependency" in d.reasons


def test_unknown_parents_are_not_zero_or_stage_admission_limit():
    c = controller(max_accepted_parents=1)
    d = decide(c, sample=state(accepted_parents=None))
    assert d.action == "issue" and "accepted_parents" in d.unknown_fields
    c.accepted(d.attempt)
    d = decide(c, now=11, key=2, sample=state(11, accepted_parents=10_000))
    assert d.action == "issue"
    c.accepted(d.attempt)
    assert c.snapshot()["parent_cap_enforced"] is False


def test_fixed_can_use_native_safety_when_free_bytes_unknown():
    c = controller()
    d = decide(c, sample=state(free_staging_bytes=None), staging_bytes_needed=16)
    assert d.action == "issue" and "free_staging_bytes" in d.unknown_fields
    c.accepted(d.attempt)


def test_pressure_unknown_free_for_new_staging_falls_back_and_acceptance_charges():
    c = controller(mode="pressure", reserve_staging_bytes=16)
    d = decide(c, sample=state(free_staging_bytes=None), staging_bytes_needed=16)
    assert d.action == "native_fallback"
    c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 16)


@pytest.mark.parametrize("sample", [
    state(captured_ns=0), state(captured_ns=200_002),
    state(200_001, inflight=None), state(200_001, native_issue_safe=None),
])
def test_stale_future_or_unknown_required_state_falls_back_without_fake_zero(sample):
    c = controller(cumulative=ZERO)
    d = decide(c, now=200_001, sample=sample)
    assert d.action == "native_fallback" and d.unknown_fields
    c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 16)
    assert c.snapshot()["records"][-1]["unknown_fields"]


def test_missing_state_not_fabricated_as_empty():
    c = controller()
    d = c.decide("ssd_read", 16, None, now_ns=10, work_id=1)
    assert d.action == "native_fallback" and "native_state" in d.unknown_fields
    c.accepted(d.attempt)
    assert c.used[0].ops == 1


def test_shadow_never_denies_and_charges_real_accepts_over_caps():
    c = controller(mode="shadow", cumulative=ZERO)
    d = decide(c)
    assert d.action == "native_fallback" and d.attempt.performance_reasons
    c.accepted(d.attempt)
    assert c.used[0] == Amount(1, 16) and c.shadow_overruns == 1
    assert c.denied_performance == 0


def test_defer_has_no_pending_attempt_and_only_bounded_value_metadata():
    c = controller(cumulative=ZERO, max_waiting_keys=1)
    d = decide(c)
    assert d.action == "defer" and c.pending is None and len(c.waiting) == 1
    d = decide(c, now=11, key=2)
    assert d.action == "native_fallback" and "waiting_metadata_capacity" in d.unknown_fields
    c.accepted(d.attempt)
    assert len(c.waiting) == 1 and c.metadata_fallbacks == 1
    c.retire_work("ssd_read", 1)
    assert not c.waiting


def test_future_or_native_resource_cannot_be_work_identity():
    c = controller()
    for identity in (Future(), object(), (), ("x" * 129,), (1, 2, 3, 4, 5), True):
        with pytest.raises(ValueError, match="work_id"):
            decide(c, key=identity)
    assert c.grant is None and not c.waiting


def test_uncertain_is_sticky_not_rejected_or_zero_executed():
    c = controller()
    d = decide(c, "h2d", 64)
    c.uncertain(d.attempt, "may have partially launched")
    s = c.snapshot()
    assert s["uncertain_ops"] == 1 and s["rejected_ops"] == 0
    assert s["uncertain_requested_bytes"] == 64
    assert s["epoch_totals_are_lower_bounds"] and not s["acceptance_accounting_complete"]
    assert not c.refresh_epoch(2_000_010)
    d = decide(c, "ssd_write", 16, now=2_000_011, key=2, progress="continuation")
    assert d.action == "native_fallback" and d.attempt.epoch is None
    c.accepted(d.attempt)
    s = c.snapshot()
    assert s["accepted_without_epoch_ops"] == 1 and not s["physical_drain_inferred"]


def test_completion_unknown_never_refunds_accepted_work():
    c = controller()
    d = decide(c, "d2h", 64)
    c.accepted(d.attempt)
    c.mark_completion_unknown("d2h", "event.record failed after acceptance")
    assert c.used[3] == Amount(1, 64)
    s = c.snapshot()
    assert s["completion_unknown_ops"] == 1 and not s["completion_accounting_complete"]
    assert not c.refresh_epoch(2_000_010)
    assert c.used[3] == Amount(1, 64)


def test_failure_or_pending_cannot_claim_classification_complete():
    c = controller()
    d = decide(c)
    s = c.snapshot()
    assert not s["acceptance_accounting_complete"] and not s["completion_accounting_complete"]
    c.fail("audit exception")
    assert c.pending is d.attempt
    c.accepted(d.attempt)
    s = c.snapshot()
    assert not s["acceptance_accounting_complete"] and not s["resource_release_inferred"]


def test_record_window_bounded_and_terminal_read_does_not_claim_dma_drain():
    c = controller(max_records=2)
    for i in range(4):
        d = decide(c, now=10 + i, key=i)
        c.accepted(d.attempt)
    s = c.snapshot(native_shutdown=True)
    assert len(s["records"]) == 2 and s["overwritten_records"] == 2
    assert s["complete_flags_scope"].startswith("audit classifications")
    assert not s["physical_drain_inferred"] and not s["resource_release_inferred"]
    for field in ("dependency_policy_enabled", "interference_policy_enabled", "joint_policy_enabled"):
        assert s[field] is False


def test_foreign_owner_diagnostic_failure_preserves_all_issue_state():
    c = controller()
    d = decide(c, "h2d", 32)
    before = (c.owner, c.last_now, c.grant, tuple(c.used), c.pending, c.epoch_refreshes)
    errors = []
    def other():
        try:
            c.fail("foreign-owner optional diagnostic")
        except Exception as exc:
            errors.append(exc)
    t = Thread(target=other)
    t.start()
    t.join()
    assert not errors and c.faulted
    assert (c.owner, c.last_now, c.grant, tuple(c.used), c.pending, c.epoch_refreshes) == before
    assert c.pending is d.attempt
    c.accepted(d.attempt)
    assert c.used[2] == Amount(1, 32)
    assert not c.refresh_epoch(2_000_010)
    s = c.snapshot()
    assert not s["acceptance_accounting_complete"]
    assert not s["resource_release_inferred"] and not s["physical_drain_inferred"]
