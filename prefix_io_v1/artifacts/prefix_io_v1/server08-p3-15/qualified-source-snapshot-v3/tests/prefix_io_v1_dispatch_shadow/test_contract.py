"""CPU-only acceptance contract tests; no native/CUDA API is executed."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from prefix_io_control.dispatch_budget import Amount, DispatchBudget, STAGES, ZERO
from prefix_io_control.dispatch_shadow import DispatchShadow, ShadowAttempt, ShadowState
from prefix_io_control.stage_accounting import StageAccounting


def grant(**changes):
    value = DispatchBudget("r", 0, 0, 100, (Amount(2, 8192),) * 4,
                           (Amount(2, 8192),) * 4, Amount(3, 12288),
                           Amount(3, 12288), 12288, 12288, 4096, 2, 10)
    return replace(value, **changes)


def state(**changes):
    return replace(ShadowState("r", 0, ZERO, 16384, 1, True), **changes)


def audit(g=None, **kwargs):
    x = DispatchShadow("r", mode="shadow", **kwargs)
    x.bind()
    x.publish(g or grant(), now_ns=0)
    return x


def accepted(x, stage="h2d", nbytes=4096, sample=None, **kwargs):
    attempt = x.begin(stage, nbytes, sample or state(), now_ns=kwargs.pop("now_ns", 0), **kwargs)
    assert x.accepted(attempt)
    return attempt


def test_off_does_not_validate_inputs_take_owner_or_create_attempt():
    x = DispatchShadow("r")
    assert x.publish(object(), now_ns=object()) is False
    assert x.begin(object(), object(), object(), now_ns=object()) is None
    assert x.accepted(None) is False
    assert x.mark_completion_unknown(object(), object()) is False
    assert x.owner is None and x.pending is None
    assert x.snapshot()["records"] == []


def test_bind_is_exclusive_and_does_not_take_owner():
    x = DispatchShadow("r", mode="shadow")
    x.bind()
    assert x.bound and x.owner is None
    with pytest.raises(ValueError):
        x.bind()


def test_unbound_shadow_cannot_publish_or_begin():
    x = DispatchShadow("r", mode="shadow")
    with pytest.raises(RuntimeError):
        x.publish(grant(), now_ns=0)
    with pytest.raises(RuntimeError):
        x.begin("h2d", 1, state(), now_ns=0)


def test_would_defer_acceptance_still_consumes_epoch_credit():
    x = audit(grant(cumulative=ZERO))
    a = accepted(x)
    assert a.verdict == "would_defer" and a.permit is not None
    assert "stage_epoch_bytes" in a.performance_reasons
    assert x.snapshot()["used"]["h2d"] == dict(ops=1, bytes=4096)
    assert x.snapshot()["shadow_overruns"] == 1
    b = x.begin("h2d", 4096, state(), now_ns=0)
    assert b.verdict == "would_defer"
    x.accepted(b)
    assert x.snapshot()["used"]["h2d"] == dict(ops=2, bytes=8192)


def test_cumulative_credit_is_not_refunded_by_native_completion_or_new_pump():
    x = audit()
    a = StageAccounting()
    a.bind()
    accepted(x)
    a.accepted("h2d", 1, 4096)
    a.completed("h2d", 1, -5)
    accepted(x)
    assert a.snapshot()["stages"]["h2d"]["inflight_bytes"] == 0
    assert x.publish(grant(), now_ns=0) is False
    ticket = x.begin("h2d", 1, state(), now_ns=0)
    assert ticket.verdict == "would_defer"
    assert "stage_epoch_ops" in ticket.performance_reasons
    x.rejected(ticket, "proved before acceptance")
    assert x.snapshot()["used"]["h2d"]["bytes"] == 8192


def test_new_nonoverlapping_epoch_resets_only_epoch_usage():
    x = audit()
    accepted(x)
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    assert x.snapshot()["used"]["h2d"] == dict(ops=0, bytes=0)
    assert x.snapshot()["observed_api_accepted"]["h2d"] == dict(ops=1, bytes=4096)
    ticket = accepted(x, sample=state(captured_ns=100), now_ns=100)
    assert ticket.epoch == 1


def test_rejected_before_acceptance_does_not_consume_allowance():
    x = audit()
    ticket = x.begin("ssd_write", 4096, state(), now_ns=0)
    assert x.rejected(ticket, "validated before enqueue")
    report = x.snapshot()
    assert report["used"]["ssd_write"] == dict(ops=0, bytes=0)
    assert report["observed_api_accepted"]["ssd_write"]["ops"] == 0
    assert report["rejected_ops"] == 1 and report["uncertain_ops"] == 0


def test_uncertain_api_exception_is_not_rejected_or_zero_execution():
    x = audit()
    ticket = x.begin("d2h", 4096, state(), now_ns=0)
    x.uncertain(ticket, "kernel API raised; partial launch possible")
    report = x.snapshot()
    assert report["rejected_ops"] == 0 and report["uncertain_ops"] == 1
    assert report["uncertain_requested_bytes"] == 4096
    assert report["acceptance_accounting_complete"] is False
    assert report["completion_accounting_complete"] is False
    assert report["epoch_totals_are_lower_bounds"] is True
    assert report["physical_drain_inferred"] is False
    following = x.begin("h2d", 1, state(), now_ns=0)
    assert following.verdict == "unknown"
    assert "prior_acceptance_uncertain" in following.unknown_fields
    x.accepted(following)


def test_new_epoch_does_not_repair_prior_uncertain_execution():
    x = audit()
    x.uncertain(x.begin("d2h", 4096, state(), now_ns=0))
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    ticket = x.begin("h2d", 1, state(captured_ns=100), now_ns=100)
    assert ticket.verdict == "unknown"
    x.accepted(ticket)
    assert x.snapshot()["acceptance_accounting_complete"] is False


def test_event_record_failure_after_api_return_does_not_refund_acceptance():
    x = audit()
    accepted(x, stage="d2h")
    x.mark_completion_unknown("d2h", "end event record failed after API returned")
    report = x.snapshot()
    assert report["used"]["d2h"] == dict(ops=1, bytes=4096)
    assert report["completion_unknown_ops"] == 1
    assert report["completion_accounting_complete"] is False
    assert report["rejected_ops"] == report["uncertain_ops"] == 0
    ticket = x.begin("ssd_write", 4096, state(), now_ns=0)
    assert "prior_completion_unknown" in ticket.unknown_fields
    x.accepted(ticket)


def test_no_grant_records_actual_acceptance_without_fabricating_epoch():
    x = DispatchShadow("r", mode="shadow")
    x.bind()
    ticket = x.begin("ssd_read", 4096, state(), now_ns=0)
    assert ticket.verdict == "unknown" and ticket.permit is None and ticket.epoch is None
    x.accepted(ticket)
    report = x.snapshot()
    assert report["epoch"] is None
    assert report["accepted_without_epoch_ops"] == 1
    assert report["observed_api_accepted"]["ssd_read"]["bytes"] == 4096


def test_expired_grant_does_not_charge_expired_or_future_epoch():
    x = audit()
    ticket = x.begin("ssd_read", 4096, state(captured_ns=100), now_ns=100)
    assert ticket.permit is None and "current_grant" in ticket.unknown_fields
    x.accepted(ticket)
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    assert x.snapshot()["used"]["ssd_read"]["bytes"] == 0
    assert x.snapshot()["accepted_without_epoch_bytes"] == 4096


def test_call_returning_after_expiry_charged_to_issuing_epoch():
    x = audit()
    ticket = x.begin("h2d", 4096, state(captured_ns=99), now_ns=99)
    # A backend may return later; settle has no clock/expiry refund.
    x.accepted(ticket)
    assert x.snapshot()["used"]["h2d"]["bytes"] == 4096
    assert x.snapshot()["records"][-1]["epoch"] == 0
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)


def test_unknown_fields_remain_none_and_do_not_become_zero_capacity():
    x = audit()
    sample = state(free_staging_bytes=None, accepted_parents=None)
    ticket = x.begin("ssd_read", 4096, sample, now_ns=0, staging_bytes_needed=4096)
    assert sample.free_staging_bytes is None and sample.accepted_parents is None
    assert ticket.verdict == "unknown"
    assert set(ticket.unknown_fields) == {"free_staging_bytes", "accepted_parents"}
    assert ticket.native_reasons == ()
    assert "staging_reserve" not in ticket.performance_reasons
    x.accepted(ticket)
    assert x.snapshot()["parent_cap_enforced"] is False


def test_unknown_inflight_still_reports_known_cumulative_overage():
    x = audit(grant(cumulative=ZERO))
    ticket = x.begin("d2h", 4096, state(inflight=None), now_ns=0)
    assert ticket.verdict == "unknown" and "inflight" in ticket.unknown_fields
    assert "stage_epoch_bytes" in ticket.performance_reasons
    assert "stage_inflight_bytes" not in ticket.performance_reasons
    x.accepted(ticket)
    assert x.snapshot()["used"]["d2h"]["bytes"] == 4096


def test_stale_and_future_samples_are_unknown_not_native_permissions():
    for sample in (state(), state(captured_ns=12)):
        x = audit()
        ticket = x.begin("h2d", 4096, sample, now_ns=11)
        assert ticket.verdict == "unknown" and "fresh_native_state" in ticket.unknown_fields
        x.accepted(ticket)
        assert x.snapshot()["observed_api_accepted"]["h2d"]["bytes"] == 4096


def test_wrong_run_grant_and_sample_rejected():
    x = audit()
    with pytest.raises(ValueError):
        x.publish(grant(run_id="wrong"), now_ns=0)
    with pytest.raises(ValueError):
        x.begin("h2d", 1, state(run_id="wrong"), now_ns=0)
    assert x.pending is None


def test_wrong_owner_rejected_but_explicit_shutdown_read_allowed():
    x = audit()
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):
            pool.submit(x.begin, "h2d", 1, state(), now_ns=0).result()
        with pytest.raises(RuntimeError):
            pool.submit(x.snapshot).result()
        report = pool.submit(x.snapshot, native_shutdown=True).result()
    assert report["native_shutdown_read"] is True
    assert report["physical_drain_inferred"] is False


def test_clock_regression_rejected():
    x = audit()
    ticket = accepted(x, sample=state(captured_ns=10), now_ns=10)
    with pytest.raises(ValueError):
        x.begin("h2d", 1, state(), now_ns=9)
    assert ticket.epoch == 0 and x.pending is None


def test_same_epoch_republication_does_not_refill_and_cannot_mutate():
    x = audit()
    accepted(x)
    assert x.publish(grant(), now_ns=0) is False
    for mutation in (grant(cumulative=ZERO), grant(expires_ns=99)):
        with pytest.raises(ValueError):
            x.publish(mutation, now_ns=0)
    assert x.snapshot()["used"]["h2d"]["bytes"] == 4096


def test_overlapping_or_regressing_epochs_and_stale_grants_rejected():
    x = audit()
    for value, now in ((grant(epoch=1, issued_ns=50, expires_ns=150), 50),
                       (grant(epoch=0, issued_ns=100, expires_ns=200), 100),
                       (grant(epoch=1, issued_ns=100, expires_ns=200), 200)):
        with pytest.raises(ValueError):
            x.publish(value, now_ns=now)
    assert x.snapshot()["epoch"] == 0


def test_one_immediate_attempt_and_no_grant_replacement_while_pending():
    x = audit()
    ticket = x.begin("h2d", 4096, state(), now_ns=0)
    with pytest.raises(RuntimeError):
        x.begin("d2h", 4096, state(), now_ns=0)
    with pytest.raises(RuntimeError):
        x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    x.accepted(ticket)
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)


def test_forged_foreign_and_duplicate_attempts_rejected():
    x = audit()
    ticket = x.begin("h2d", 1, state(), now_ns=0)
    with pytest.raises(ValueError):
        x.accepted(replace(ticket))
    other = audit()
    foreign = other.begin("h2d", 1, state(), now_ns=0)
    with pytest.raises(ValueError):
        x.accepted(foreign)
    x.accepted(ticket)
    with pytest.raises(ValueError):
        x.accepted(ticket)
    other.rejected(foreign, "test pre-acceptance cancellation")


def test_fused_h2d_one_operation_charges_all_actual_destinations():
    x = audit()
    accepted(x, stage="ssd_read", nbytes=4096)
    accepted(x, stage="h2d", nbytes=8192)
    report = x.snapshot()
    assert report["used"]["ssd_read"] == dict(ops=1, bytes=4096)
    assert report["used"]["h2d"] == dict(ops=1, bytes=8192)
    ticket = x.begin("h2d", 1, state(), now_ns=0)
    assert "stage_epoch_bytes" in ticket.performance_reasons
    x.accepted(ticket)


def test_shared_read_write_allowance_not_duplicated_by_direction():
    x = audit(grant(shared_ssd_cumulative=Amount(1, 4096)))
    accepted(x, stage="ssd_read")
    ticket = x.begin("ssd_write", 4096, state(), now_ns=0)
    assert ticket.verdict == "would_defer"
    assert "shared_ssd_epoch_ops" in ticket.performance_reasons
    assert "shared_ssd_epoch_bytes" in ticket.performance_reasons
    x.accepted(ticket)
    assert x.snapshot()["used"]["ssd_write"]["bytes"] == 4096


def test_shared_copy_inflight_uses_both_directions_without_native_device_counter():
    x = audit(grant(shared_copy_inflight_bytes=4096))
    sample = state(inflight=(Amount(), Amount(), Amount(1, 4096), Amount()))
    ticket = x.begin("d2h", 1, sample, now_ns=0)
    assert "shared_copy_inflight_bytes" in ticket.performance_reasons
    x.accepted(ticket)
    assert x.snapshot()["used"]["d2h"]["bytes"] == 1


def test_staging_capacity_and_reserve_remain_separate_observations():
    x = audit()
    ticket = x.begin("ssd_read", 4096, state(free_staging_bytes=4096),
                     now_ns=0, staging_bytes_needed=4096)
    assert ticket.native_reasons == () and "staging_reserve" in ticket.performance_reasons
    x.accepted(ticket)
    ticket = x.begin("d2h", 4096, state(free_staging_bytes=0),
                     now_ns=0, staging_bytes_needed=4096)
    assert "native_capacity_or_dependency" in ticket.native_reasons
    x.accepted(ticket)
    assert x.snapshot()["byte_caps_enforced"] is False


def test_parent_count_is_observed_but_intake_is_not_evaluated():
    x = audit()
    ticket = x.begin("ssd_write", 1, state(accepted_parents=1000), now_ns=0)
    assert ticket.verdict == "would_issue"
    assert not any("parent" in reason for reason in ticket.performance_reasons)
    x.accepted(ticket)
    assert x.snapshot()["parent_cap_enforced"] is False


def test_continuation_is_only_factual_tag_and_not_a_quota_override():
    x = audit(grant(cumulative=ZERO))
    ticket = x.begin("ssd_write", 4096, state(), now_ns=0, progress="continuation")
    assert ticket.verdict == "would_defer" and ticket.progress == "continuation"
    x.accepted(ticket)
    report = x.snapshot()
    assert report["shadow_overruns"] == 1
    assert report["progress_override_enforced"] is False


def test_fail_disables_future_observation_without_erasing_accepted_evidence():
    x = audit()
    accepted(x)
    x.fail("optional observer broke " * 20)
    assert x.begin(object(), object(), object(), now_ns=object()) is None
    assert x.publish(object(), now_ns=object()) is False
    report = x.snapshot()
    assert report["faulted"] and len(report["error"]) <= 160
    assert report["used"]["h2d"]["bytes"] == 4096
    assert report["acceptance_accounting_complete"] is False
    assert report["completion_accounting_complete"] is False


def test_fault_during_immediate_call_can_still_settle_existing_attempt():
    x = audit()
    ticket = x.begin("h2d", 4096, state(), now_ns=0)
    x.fail("unrelated audit diagnostics failed")
    x.accepted(ticket)
    assert x.snapshot()["pending_attempt"] is False
    assert x.snapshot()["used"]["h2d"]["bytes"] == 4096


def test_history_and_reason_are_bounded_and_attempt_has_values_only():
    x = audit(max_records=2)
    for _ in range(3):
        ticket = x.begin("ssd_read", 1, state(), now_ns=0)
        x.rejected(ticket, "r" * 500)
    report = x.snapshot()
    assert len(report["records"]) == 2 and report["overwritten_records"] == 1
    assert len(report["records"][-1]["reason"]) == 160
    assert {f.name for f in fields(ShadowAttempt)} == {
        "stage", "nbytes", "started_ns", "permit", "verdict", "performance_reasons",
        "native_reasons", "unknown_fields", "progress"}
    assert not hasattr(x, "complete") and not hasattr(x, "queue")


def test_strict_value_types_and_only_off_shadow_modes():
    for kwargs in (dict(mode="fixed"), dict(mode="pressure"), dict(max_records=True),
                   dict(max_records=0), dict(max_records=97)):
        with pytest.raises(ValueError):
            DispatchShadow("r", **kwargs)
    for mutation in (dict(free_staging_bytes=True), dict(accepted_parents=-1),
                     dict(native_issue_safe=1), dict(inflight=list(ZERO)), dict(captured_ns=True)):
        with pytest.raises(ValueError):
            state(**mutation)
    x = audit()
    for value in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError):
            x.begin("h2d", value, state(), now_ns=0)
    with pytest.raises(ValueError):
        x.begin("unknown", 1, state(), now_ns=0)
    with pytest.raises(ValueError):
        x.begin("h2d", 1, state(), now_ns=0, progress="invented")
    with pytest.raises(FrozenInstanceError):
        state().free_staging_bytes = 0
    ticket = x.begin("h2d", 1, state(), now_ns=0)
    with pytest.raises(ValueError):
        x.settle(ticket, outcome=True)
    with pytest.raises(ValueError):
        x.settle(ticket, outcome="accepted", reason=object())
    x.accepted(ticket)


def test_pending_audit_is_incomplete_and_settle_does_not_prove_dma_drain():
    x = audit()
    ticket = x.begin("h2d", 4096, state(), now_ns=0)
    report = x.snapshot()
    assert report["pending_attempt"]
    assert report["acceptance_accounting_complete"] is False
    assert report["completion_accounting_complete"] is False
    x.accepted(ticket)
    report = x.snapshot()
    assert report["acceptance_accounting_complete"] is True
    assert report["completion_accounting_complete"] is True
    assert report["physical_drain_inferred"] is False
    assert report["resource_release_inferred"] is False
    assert report["complete_flags_scope"] == "audit classifications only; not DMA completion or drain"
    assert report["native_adapter_installed_by_this_module"] is False
