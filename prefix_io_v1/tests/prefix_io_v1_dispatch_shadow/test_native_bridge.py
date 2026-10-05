"""Native acceptance boundary audit with fake CUDA/file APIs; no GPU metrics."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor, TransferCoordinator
from prefix_io_control.dispatch_budget import DispatchBudget, ZERO, Amount, STAGES
from prefix_io_control.dispatch_shadow import DispatchShadow
from prefix_io_control.stage_accounting import StageAccounting
from tests.prefix_io_v1_start_budget._fixture import rig, eventually
from tests.prefix_io_v1_start_budget.test_stage_accounting import attach as attach_accounting


class ZeroAllowance(DispatchShadow):
    """Fixture-only epoch publication on the actual reactor owner thread."""
    def __init__(self, run_id, *, mode="shadow", publish_grant=True):
        super().__init__(run_id, mode=mode)
        self.publish_grant = publish_grant
        self.samples = []

    def begin(self, stage, nbytes, sample=None, *, now_ns, **kwargs):
        if self.publish_grant and self.grant is None:
            grant = DispatchBudget(
                self.run_id, 0, now_ns, now_ns + 1_000_000_000, ZERO, ZERO,
                Amount(), Amount(), 0, 0, 0, 1, 1000
            )
            self.publish(grant, now_ns=now_ns)
        self.samples.append((stage, sample))
        return super().begin(stage, nbytes, sample, now_ns=now_ns, **kwargs)


def attach(x, *, mode="shadow", publish_grant=True):
    accounting = attach_accounting(x)
    shadow = ZeroAllowance("cpu-run", mode=mode, publish_grant=publish_grant)
    shadow.bind()
    x.r._prefix_dispatch_shadow = shadow
    # P3 policies are qualified independently; fixture does not mix start allowances.
    x.r._prefix_start_budget = None
    return accounting, shadow


def snapshot(shadow):
    return shadow.snapshot(native_shutdown=True)


def test_zero_allowance_never_blocks_four_native_physical_stages(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        accounting, shadow = attach(x)
        store = x.submit(files=1)
        x.start()
        assert store.result(timeout=2) == 4096
        load = x.load()
        assert load.result(timeout=2) == 4096
        x.stop()
        s = snapshot(shadow)
        assert not s["faulted"] and s["shadow_overruns"] == 4
        assert not s["pending_attempt"] and s["uncertain_ops"] == 0
        for stage in STAGES:
            assert s["observed_api_accepted"][stage] == dict(ops=1, bytes=4096)
            assert s["used"][stage] == dict(ops=1, bytes=4096)
        assert accounting.valid and not accounting.records
        assert not s["byte_caps_enforced"] and not s["parent_cap_enforced"]
        assert not s["resource_release_inferred"] and not s["physical_drain_inferred"]


def test_samples_keep_unavailable_capacity_unknown_and_use_actual_inflight(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        accounting, shadow = attach(x)
        x.copy_ready.clear()
        future = x.submit(files=2)
        x.start()
        eventually(lambda: accounting.stats["d2h"]["inflight_ops"] == 2)
        d2h = [sample for stage, sample in shadow.samples if stage == "d2h"]
        assert [sample.inflight[3] for sample in d2h] == [Amount(), Amount(1, 4096)]
        assert all(sample.free_staging_bytes is None and sample.accepted_parents is None
                   and sample.native_issue_safe is True for sample in d2h)
        x.copy_ready.set()
        assert future.result(timeout=2) == 8192
        x.stop()
        for record in snapshot(shadow)["records"]:
            assert "free_staging_bytes" in record["unknown_fields"]
            assert "accepted_parents" in record["unknown_fields"]


def test_invalid_passive_accounting_is_unknown_not_fabricated_zero(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        accounting, shadow = attach(x)
        accounting.invalidate("fixture unavailable")
        future = x.submit(files=1)
        x.start()
        assert future.result(timeout=2) == 4096
        x.stop()
        assert not shadow.faulted
        assert all(sample.inflight is None for _, sample in shadow.samples)
        assert all("inflight" in record["unknown_fields"] for record in snapshot(shadow)["records"])


def test_no_current_grant_still_records_native_acceptance(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x, publish_grant=False)
        future = x.submit(files=1)
        x.start()
        assert future.result(timeout=2) == 4096
        x.stop()
        s = snapshot(shadow)
        assert s["epoch"] is None
        assert s["accepted_without_epoch_ops"] == 2
        assert s["accepted_without_epoch_bytes"] == 8192
        assert all("current_grant" in record["unknown_fields"] for record in s["records"])


def test_shared_read_once_and_fused_physical_destinations_charged_together(monkeypatch):
    with rig(monkeypatch, enabled=False, share=True) as x:
        _, shadow = attach(x)
        x.r.ring.auto = False
        x.preload([b"shared"])
        x.start()
        eventually(lambda: x.opens == 1)
        one = x.load(job_id=20, hashes=[b"shared"])
        two = x.load(job_id=21, hashes=[b"shared"])
        eventually(lambda: len(x.r._preload_waiters.get(b"shared", ())) == 2)
        x.r.ring.auto = True
        assert one.result(timeout=2) == two.result(timeout=2) == 4096
        x.stop()
        s = snapshot(shadow)
        assert s["observed_api_accepted"]["ssd_read"] == dict(ops=1, bytes=4096)
        assert s["observed_api_accepted"]["h2d"] == dict(ops=1, bytes=8192)
        assert x.reads == 1 and x.h2d_launches == [dict(entries=2, bytes=8192)]


def test_copy_audit_charges_actual_partial_mapping(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        original = x.r._fill_swap_ptrs
        def partial(*args):
            count = original(*args)
            args[5][:count] = 1024
            return count
        x.r._fill_swap_ptrs = partial
        future = x.load()
        x.start()
        assert future.result(timeout=2) == 4096
        x.stop()
        s = snapshot(shadow)
        assert s["observed_api_accepted"]["ssd_read"]["bytes"] == 4096
        assert s["observed_api_accepted"]["h2d"]["bytes"] == 1024


@pytest.mark.parametrize("stage", ["ssd_read", "ssd_write"])
def test_backend_exception_is_uncertain_and_native_error_unchanged(monkeypatch, stage):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        native_error = IOError("fixture queue failure")
        setattr(x.r.file_store, "queue_read" if stage == "ssd_read" else "queue_write",
                Mock(side_effect=native_error))
        future = x.load() if stage == "ssd_read" else x.submit(files=1)
        x.start()
        with pytest.raises(IOError) as caught:
            future.result(timeout=2)
        assert caught.value is native_error
        x.stop()
        s = snapshot(shadow)
        assert s["uncertain_ops"] == 1 and s["rejected_ops"] == 0
        assert s["uncertain_requested_bytes"] == 4096
        assert s["observed_api_accepted"][stage]["ops"] == 0
        assert not s["acceptance_accounting_complete"] and s["epoch_totals_are_lower_bounds"]
        assert len(x.r.staging_pool.releases) == 1


@pytest.mark.parametrize("stage", ["ssd_read", "ssd_write"])
def test_short_cqe_does_not_refund_successfully_accepted_bytes(monkeypatch, stage):
    with rig(monkeypatch, enabled=False) as x:
        accounting, shadow = attach(x)
        x.r.ring.results[2 if stage == "ssd_read" else 1] = 4095
        future = x.load() if stage == "ssd_read" else x.submit(files=1)
        x.start()
        with pytest.raises(IOError):
            future.result(timeout=2)
        x.stop()
        s = snapshot(shadow)
        assert s["used"][stage] == dict(ops=1, bytes=4096)
        assert s["observed_api_accepted"][stage] == dict(ops=1, bytes=4096)
        assert s["uncertain_ops"] == s["rejected_ops"] == 0
        assert accounting.stats[stage]["failed_ops"] == 1


@pytest.mark.parametrize("is_store", [False, True])
def test_cuda_backend_exception_is_uncertain_not_zero_execution(monkeypatch, is_store):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        native_error = RuntimeError("fixture ambiguous CUDA launch")
        monkeypatch.setattr(mod, "ops", NS(swap_blocks_batch=Mock(side_effect=native_error)))
        future = x.submit(files=1) if is_store else x.load()
        x.start()
        with pytest.raises(RuntimeError) as caught:
            future.result(timeout=2)
        assert caught.value is native_error
        x.stop()
        s = snapshot(shadow)
        stage = "d2h" if is_store else "h2d"
        assert s["observed_api_accepted"][stage]["ops"] == 0
        assert s["uncertain_ops"] == 1 and s["uncertain_requested_bytes"] == 4096
        assert s["rejected_ops"] == 0 and not s["physical_drain_inferred"]


@pytest.mark.parametrize("is_store", [False, True])
def test_api_acceptance_precedes_failed_end_event_and_is_never_refunded(monkeypatch, is_store):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        native_error = RuntimeError("fixture end-event failure")
        original_event = mod.torch.cuda.Event
        count = [0]
        def event(**kwargs):
            count[0] += 1
            obj = original_event(**kwargs)
            # submit_store first records a separate compute event.
            end_number = 3 if is_store else 2
            if count[0] == end_number:
                obj.record = Mock(side_effect=native_error)
            return obj
        monkeypatch.setattr(mod.torch.cuda, "Event", event)
        future = x.submit(files=1) if is_store else x.load()
        x.start()
        with pytest.raises(RuntimeError) as caught:
            future.result(timeout=2)
        assert caught.value is native_error
        x.stop()
        s = snapshot(shadow)
        stage = "d2h" if is_store else "h2d"
        assert s["observed_api_accepted"][stage] == dict(ops=1, bytes=4096)
        assert s["used"][stage] == dict(ops=1, bytes=4096)
        assert s["completion_unknown_ops"] == 1 and stage in s["completion_unknown_stages"]
        assert not s["completion_accounting_complete"]
        assert s["uncertain_ops"] == 0 and s["rejected_ops"] == 0
        assert not s["physical_drain_inferred"] and not s["resource_release_inferred"]


@pytest.mark.parametrize("fault", ["owner", "begin", "settle"])
def test_optional_shadow_fault_preserves_native_parent_completion(monkeypatch, fault):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        if fault == "owner":
            shadow.owner = -1
        elif fault == "begin":
            shadow.begin = Mock(side_effect=ValueError("fixture audit"))
        else:
            shadow.accepted = Mock(side_effect=ValueError("fixture audit settle"))
        future = x.submit(files=2)
        x.start()
        assert future.result(timeout=2) == 8192
        load = x.load()
        assert load.result(timeout=2) == 4096
        x.stop()
        assert shadow.faulted and len(x.r.staging_pool.releases) == 3
        assert not snapshot(shadow)["acceptance_accounting_complete"]


def test_progress_tags_are_facts_and_never_override_zero_allowance(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        _, shadow = attach(x)
        future = x.submit(files=1)
        x.coordinator.request_mandatory([future])
        x.start()
        assert future.result(timeout=2) == 4096
        load = x.load()
        assert load.result(timeout=2) == 4096
        x.stop()
        records = snapshot(shadow)["records"]
        assert records[0]["stage"] == "d2h" and records[0]["progress"] == "mandatory"
        assert records[1]["stage"] == "ssd_write" and records[1]["progress"] == "mandatory"
        assert [record["progress"] for record in records[2:]] == ["continuation", "continuation"]
        assert all(record["performance_reasons"] for record in records)
        assert not snapshot(shadow)["progress_override_enforced"]


def test_duplicate_cqe_and_shutdown_keep_one_native_parent_completion(monkeypatch):
    with rig(monkeypatch, enabled=False) as x:
        accounting, shadow = attach(x)
        x.r.ring.duplicate = True
        one = x.submit(files=2, job_id=7)
        two = x.submit(files=2, job_id=8)
        done = []
        one.add_done_callback(lambda _: done.append(7))
        two.add_done_callback(lambda _: done.append(8))
        x.start()
        x.stop()
        assert one.result() == two.result() == 8192
        assert sorted(done) == [7, 8] and len(x.r.staging_pool.releases) == 4
        s = snapshot(shadow)
        assert s["observed_api_accepted"]["d2h"] == dict(ops=4, bytes=16384)
        assert s["observed_api_accepted"]["ssd_write"] == dict(ops=4, bytes=16384)
        assert accounting.valid and not accounting.records


@pytest.mark.parametrize("mode", [None, "off"])
def test_off_copy_audit_does_not_read_clock_or_sizes(monkeypatch, mode):
    reactor = IoReactor.__new__(IoReactor)
    if mode is not None:
        reactor._prefix_dispatch_shadow = DispatchShadow("r", mode=mode)
    clock = Mock(side_effect=AssertionError("off clock"))
    monkeypatch.setattr(mod.time, "monotonic_ns", clock)
    class Sizes:
        def __getitem__(self, _):
            raise AssertionError("off copy sizes")
    assert reactor._prefix_shadow_copy_begin("h2d", Sizes(), 99) is None
    assert reactor._prefix_shadow_begin("ssd_read", 4096) is None
    clock.assert_not_called()


@pytest.mark.parametrize("mode", [None, "off"])
def test_native_off_completes_without_entering_shadow_module(monkeypatch, mode):
    with rig(monkeypatch, enabled=False) as x:
        if mode is not None:
            attach(x, mode=mode)
        monkeypatch.setattr(DispatchShadow, "begin", Mock(side_effect=AssertionError("off begin")))
        monkeypatch.setattr(mod.time, "monotonic_ns", Mock(side_effect=AssertionError("off clock")))
        future = x.submit(files=1)
        x.start()
        assert future.result(timeout=2) == 4096
        load = x.load()
        assert load.result(timeout=2) == 4096
        x.stop()


@pytest.mark.parametrize("bad", ["type", "no_progress", "identity", "no_accounting",
                                "bad_accounting", "reuse", "start", "order"])
def test_constructor_rejects_before_native_resources(monkeypatch, bad):
    from prefix_io_control.start_budget import StartBudget, StartBudgetConfig
    from prefix_io_control.store_order import MandatoryStoreOrder
    shadow = object() if bad == "type" else DispatchShadow("r", mode="shadow")
    if bad == "reuse":
        shadow.bind()
    kwargs = dict(dispatch_shadow=shadow, progress_run_id="r", stage_accounting=StageAccounting())
    if bad == "no_progress":
        kwargs["progress_run_id"] = None
    if bad == "identity":
        kwargs["progress_run_id"] = "other"
    if bad == "no_accounting":
        kwargs["stage_accounting"] = None
    if bad == "bad_accounting":
        kwargs["stage_accounting"] = object()
    if bad == "start":
        kwargs["start_budget"] = StartBudget(StartBudgetConfig("fixed", 100, 1, 100))
    if bad == "order":
        kwargs["store_order"] = MandatoryStoreOrder()
    resources = Mock(side_effect=AssertionError("native resources"))
    monkeypatch.setattr(mod, "DirectIoFileStore", resources)
    with pytest.raises((TypeError, ValueError)):
        IoReactor(config=None, file_mapper=None, layout=None, **kwargs)
    resources.assert_not_called()


def test_coordinator_forwards_shadow_without_owning_native_work(monkeypatch):
    shadow = DispatchShadow("r", mode="shadow")
    accounting = StageAccounting()
    constructor = Mock(return_value=object())
    monkeypatch.setattr(mod, "IoReactor", constructor)
    coordinator = TransferCoordinator(
        config=None, file_mapper=None, layout=None, progress_run_id="r",
        stage_accounting=accounting, dispatch_shadow=shadow
    )
    assert coordinator.reactor is constructor.return_value
    assert constructor.call_args.kwargs["dispatch_shadow"] is shadow
    assert constructor.call_args.kwargs["stage_accounting"] is accounting
    assert not shadow.bound and shadow.owner is None


def test_native_ast_recovers_seed_after_removing_optional_shadow_hooks():
    root = Path(__file__).resolve().parents[2]
    before = ast.parse((root / "artifacts/prefix_io_v1/server08-p3-13/before/shadow-seed/py_kvcache/reactor.py").read_text())
    after = ast.parse((root / "third_party/work/py-kvcache-p3-shadow-cpu/py_kvcache/reactor.py").read_text())

    def shadow_name(node):
        return any(isinstance(value, ast.Name) and value.id in
                   ("dispatch_shadow", "shadow_attempt", "shadow_accepted") for value in ast.walk(node))

    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name.startswith("_prefix_shadow"):
                return None
            pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                     if arg.arg != "dispatch_shadow"]
            node.args.kwonlyargs = [arg for arg, _ in pairs]
            node.args.kw_defaults = [default for _, default in pairs]
            return self.generic_visit(node)

        def visit_If(self, node):
            if shadow_name(node.test):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any(shadow_name(target) or isinstance(target, ast.Attribute) and
                   target.attr == "_prefix_dispatch_shadow" for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and (
                    node.value.func.attr.startswith("_prefix_shadow") or
                    isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "dispatch_shadow"):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "dispatch_shadow"]
            return self.generic_visit(node)

        def visit_Try(self, node):
            node = self.generic_visit(node)
            # Optional audit wrappers only re-raise the original backend/event error.
            if (len(node.handlers) == 1 and len(node.handlers[0].body) == 1
                    and isinstance(node.handlers[0].body[0], ast.Raise)
                    and not node.orelse and not node.finalbody):
                return node.body
            return node

    assert ast.dump(Strip().visit(after)) == ast.dump(before)
