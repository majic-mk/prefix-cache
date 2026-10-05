"""CPU preparation only: explicit C5 attachment and bounded, passive witnesses.

No receipt factory, GPU event API, queue replacement, dispatch decision, or
completion credit is provided here. A wake is a host invalidation notification.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import queue
import time
import weakref

BLOCKED = "GPU_BLOCKED_NO_C5_RECEIPT_AND_CPU_ENVIRONMENT_QUALIFICATION"
C5_REACTOR_SHA256 = "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"
C5_COLLECTOR_SHA256 = "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"
MAX_WITNESSES = 16


def require(value, reason):
    if not value:
        raise ValueError(reason)


def block_gpu_start(*_args, **_kwargs):
    """Unconditional, no paths/permissions/receipt can lift this preparation gate."""
    raise RuntimeError(BLOCKED)


def blocked_document():
    return dict(status=BLOCKED, gpu_started=False, gpu_qualified=False,
                native_cost_receipt=None, performance_claim=False)


def source_ref(path, expected_sha):
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha, "frozen C5 source hash")
    return dict(path=str(path), bytes=len(content), sha256=expected_sha)


def token_value(token):
    require(type(token) is tuple and len(token) == 4, "value-only wake token")
    require(all(type(x) is str and 0 < len(x) <= 256 for x in token[:2]) and
            all(type(x) is int and x > 0 for x in token[2:]), "bounded wake token fields")
    return list(token)


def _function_source(method, path):
    require(inspect.ismethod(method), "original bound source method")
    require(Path(method.__func__.__code__.co_filename).resolve() == Path(path).resolve(),
            "actual C5 method source path")


def validate_capture_identity(output, capture, *, run_id, request_id):
    require(type(run_id) is str and type(request_id) is str and run_id != request_id,
            "owner run and frontend request namespaces stay distinct")
    require(output["request_id"] == request_id and capture["run_id"] == run_id,
            "capture uses owner run; frontend retains external request identity")
    require(type(output["native_request_id"]) is str and output["native_request_id"],
            "original native request identity retained")


class NotificationAudit:
    """Only weak ownership and at most 16 primitive queue-wait witnesses.

    The real Queue instance and original Queue.get are retained. The transparent
    wrapper calls its original unbound function once with unchanged arguments.
    All observation errors are swallowed and fail the later audit, never native
    work. The wrapper is restored only if it is still our own wrapper.
    """
    def __init__(self, mode, reactor, bridge, capture, run_id, request_id, sources):
        self.mode, self.run_id, self.request_id = mode, run_id, request_id
        self.sources = sources
        self._reactor = weakref.ref(reactor)
        self._capture = weakref.ref(capture)
        self._bridge = weakref.ref(bridge) if bridge is not None else lambda: None
        self._queue = weakref.ref(reactor._incoming)
        self.rows = []
        self.failures = []
        self.overflow = False
        self.get_calls = 0
        self.installed = False
        self.closed = False
        self.restored = False
        self.installation = {}
        self._wrapper = None

    def _failure(self, reason):
        if len(self.failures) < 8:
            self.failures.append(str(reason)[:160])

    def _before_get(self, block, timeout):
        owner, capture = self._reactor(), self._capture()
        if owner is None or capture is None:
            self._failure("borrowed owner disappeared")
            return None
        token = getattr(owner, "_prefix_single_file_wait_armed", None)
        if token is None:
            return None  # The original idle Queue.get is outside this witness.
        self.get_calls += 1
        if len(self.rows) >= MAX_WITNESSES:
            self.overflow = True
            return None
        key, snapshot, _ = owner._prefix_single_file_retry
        age = owner._prefix_p4_bridge.policy.config.sample_max_age_ns
        deadlines = [owner._ready_fds_preload[0].open_start_ns + owner._prefix_p4_bridge.policy.config.max_wait_ns,
                     snapshot.monotonic_ns + age, snapshot.native_state.captured_ns + age, key[0][2] + age]
        registration = capture._wait_registration
        row = dict(token=token_value(token), capture_registration=(None if registration is None else token_value(registration)),
                   original_queue_same=owner._incoming is self._queue(),
                   timeout_ns=round(timeout * 1_000_000_000), block=block,
                   original_deadlines_ns=deadlines, original_deadline_ns=min(deadlines),
                   entered_ns=time.monotonic_ns(), outcome="pending", returned_token=None,
                   ended_ns=None, end_record_before_ns=None, end_record_after_ns=None)
        self.rows.append(row)
        return row

    def _after_get(self, row, item, outcome):
        if row is None:
            return
        row["ended_ns"] = time.monotonic_ns()
        row["outcome"] = outcome
        capture = self._capture()
        if capture is not None:
            active = capture.observer.events.active
            if active is not None and active[0] == row["token"][2]:
                row["end_record_before_ns"] = active[2].record_before_ns
                row["end_record_after_ns"] = active[2].record_after_ns
        # Source-qualified wake class; a similarly named user payload is not proof.
        if outcome == "returned" and type(item).__name__ == "_SingleFileWake":
            # Identity is pinned to the real source method's global class at
            # setup; generated dataclass __init__ may have <string> as source.
            if type(item) is self._wake_type:
                row["returned_token"] = token_value(item.token)
                row["outcome"] = ("matching_wake" if row["returned_token"] == row["token"] else "old_wake")

    def install_queue_observation(self, wake_type):
        queue_obj = self._queue()
        require(type(queue_obj) is queue.Queue and "get" not in vars(queue_obj),
                "unmodified original Queue.get on the one native incoming queue")
        self._wake_type = wake_type
        original_function = queue.Queue.get
        queue_ref, audit_ref = self._queue, weakref.ref(self)
        def observed_get(block=True, timeout=None):
            audit = audit_ref()
            row = None
            if audit is not None:
                try: row = audit._before_get(block, timeout)
                except Exception as exc: audit._failure("before_get: " + type(exc).__name__)
            outcome, item = "raised", None
            try:
                target = queue_ref()
                if target is None:
                    raise RuntimeError("original incoming queue no longer exists")
                item = original_function(target, block=block, timeout=timeout)
                outcome = "returned"
                return item
            except queue.Empty:
                outcome = "original_deadline_timeout"
                raise
            finally:
                if audit is not None:
                    try: audit._after_get(row, item, outcome)
                    except Exception as exc: audit._failure("after_get: " + type(exc).__name__)
        self._wrapper = observed_get
        queue_obj.get = observed_get

    def close(self):
        if self.closed:
            return
        self.closed = True
        queue_obj = self._queue()
        if self._wrapper is None:
            self.restored = True
        elif queue_obj is not None and vars(queue_obj).get("get") is self._wrapper:
            del queue_obj.get
            self.restored = True
        else:
            self._failure("queue observer replaced by another installer; not overwritten")

    def export(self):
        owner, capture = self._reactor(), self._capture()
        require(owner is not None and capture is not None, "export before borrowed owners are released")
        with capture._wait_lock:
            registration_clear = capture._wait_registration is None
            failed = capture._wait_failed
            wait_failures = list(capture.wait_failures)
        return dict(schema="c5_notification_attachment_v1", mode=self.mode,
                    run_id=self.run_id, request_id=self.request_id, installed=self.installed,
                    installation=dict(self.installation), source_refs=self.sources,
                    original_queue_same=owner._incoming is self._queue(),
                    get_calls=self.get_calls, rows=[dict(row) for row in self.rows],
                    overflow=self.overflow, failures=list(self.failures),
                    capture_wait_failed=failed, capture_wait_failures=wait_failures,
                    capture_valid=capture.valid,
                    bridge_fault=getattr(self._bridge(), "fault", None),
                    reactor_wait_faulted=getattr(owner, "_prefix_single_file_wait_faulted", False),
                    registration_clear=registration_clear,
                    reactor_arm_clear=getattr(owner, "_prefix_single_file_wait_armed", None) is None,
                    capture_detached=capture._detached, closed=self.closed, queue_get_restored=self.restored,
                    gpu_completion_proved=False, native_cost_qualified=False, performance_claim=False)


def prepare_notification_capture(mode, reactor, bridge, capture, *, run_id, request_id,
                                 reactor_source, collector_source, wake_type):
    """Shared actual callsite helper; one measured capture per fresh reactor."""
    require(mode in ("off", "shadow", "on"), "finite mode")
    require(type(run_id) is str and type(request_id) is str and run_id != request_id,
            "distinct native run and frontend request ids")
    require(capture.run_id == run_id, "capture must use native owner run_id")
    require(reactor._prefix_p4_bridge is bridge, "same actual reactor bridge")
    require((bridge is None) == (mode == "off"), "mode bridge identity")
    if bridge is not None:
        require(bridge.run_id == run_id and bridge.single_file_shadow == (mode == "shadow"),
                "bridge/capture native run and measured mode")
    require(getattr(reactor, "_prefix_single_file_wait_capture", None) is None,
            "fresh reactor; at most one notification capture")
    sources = dict(reactor=source_ref(reactor_source, C5_REACTOR_SHA256),
                   collector=source_ref(collector_source, C5_COLLECTOR_SHA256))
    for method in (reactor.install_p4_single_file_wait, reactor._prefix_wait_single_file_retry):
        _function_source(method, reactor_source)
    _function_source(reactor._prefix_wake_single_file_locked, reactor_source)
    require(wake_type is reactor._prefix_wake_single_file_locked.__func__.__globals__["_SingleFileWake"],
            "exact actual native source wake class")
    require(Path(inspect.getsourcefile(type(capture))).resolve() == Path(collector_source).resolve(),
            "actual C5 collector class")
    audit = NotificationAudit(mode, reactor, bridge, capture, run_id, request_id, sources)
    if mode != "on":
        return audit
    require(bridge._single_file_capture is not None and bridge._single_file_capture() is capture,
            "same capture already bound by existing original bridge attachment")
    # Validate the four bound lifecycle hooks before touching any attachment.
    hooks = ((capture.observer, "invalidate"), (capture.observer, "detach"),
             (capture.observer.scalar, "_fail"), (capture.observer.scalar, "detach"))
    for target, name in hooks:
        method = getattr(target, name, None)
        require(inspect.ismethod(method) and method.__self__ is target,
                "all four actual observer/scalar invalidation hooks required")
    require(type(reactor._incoming) is queue.Queue and "get" not in vars(reactor._incoming),
            "original Queue instance required before attach")
    capture.attach_single_file_wait(reactor)
    require(reactor._prefix_single_file_wait_capture() is capture and
            capture._wait_reactor_ref() is reactor and capture._wait_bridge_ref() is bridge and
            bridge.fail is capture._wait_fail_wrapper, "installed exact weak owner/capture/bridge identity")
    require(len(capture._wait_observer_bindings) == 4 and all(
        ref() is target and name == expected_name and getattr(target, name) is wrapper
        for (ref, name, _function, wrapper), (target, expected_name) in zip(capture._wait_observer_bindings, hooks)),
        "four installed original invalidation wrappers")
    audit.installed = True
    audit.installation = dict(same_capture=True, same_reactor=True, same_bridge=True,
                              native_run_id=run_id, capture_nonce=capture.wait_nonce,
                              invalidation_hook_count=4, weak_ownership=True)
    audit.install_queue_observation(wake_type)
    return audit


def validate_notification_evidence(document, *, mode, run_id, request_id, require_wait=True):
    """Value evidence alone never grants native/GPU/cost qualification."""
    require(document["schema"] == "c5_notification_attachment_v1" and
            (document["mode"], document["run_id"], document["request_id"]) == (mode, run_id, request_id),
            "notification evidence identity")
    require(document["source_refs"]["reactor"]["sha256"] == C5_REACTOR_SHA256 and
            document["source_refs"]["collector"]["sha256"] == C5_COLLECTOR_SHA256,
            "C5 actual observed source pins")
    require(document["gpu_completion_proved"] is False and document["native_cost_qualified"] is False and
            document["performance_claim"] is False,
            "notification is not GPU completion or native cost qualification")
    require(document["closed"] is True and document["queue_get_restored"] is True and
            document["capture_detached"] is True and document["original_queue_same"] is True and
            document["registration_clear"] is True and document["reactor_arm_clear"] is True,
            "original queue restored and registration cleared after original drain/detach")
    require(document["overflow"] is False and not document["failures"] and
            not document["capture_wait_failed"] and not document["capture_wait_failures"] and
            not document["reactor_wait_faulted"] and document["capture_valid"] is True and
            document["bridge_fault"] is None, "no hidden notification/observation fault")
    rows = document["rows"]
    require(type(rows) is list and len(rows) <= MAX_WITNESSES and
            type(document["get_calls"]) is int and document["get_calls"] == len(rows), "bounded complete queue witnesses")
    if mode != "on":
        require(document["installed"] is False and not rows and not document["installation"],
                "off/shadow do not install or observe new queue waits")
        return dict(status="UNINSTALLED_CONTROL_PATH", notification_exercised=False)
    installation = document["installation"]
    require(document["installed"] is True and installation == dict(same_capture=True, same_reactor=True,
        same_bridge=True, native_run_id=run_id, capture_nonce=installation["capture_nonce"],
        invalidation_hook_count=4, weak_ownership=True), "exact on attachment")
    outcomes = []
    for row in rows:
        token = token_value(tuple(row["token"]))
        require(token[:2] == [run_id, installation["capture_nonce"]] and
                row["capture_registration"] in (None, token) and row["original_queue_same"] is True and row["block"] is True,
                "original queue entered under matching live arm and registration or already-published notification")
        deadlines = row["original_deadlines_ns"]
        require(type(deadlines) is list and len(deadlines) == 4 and all(type(x) is int and x > 0 for x in deadlines)
                and row["original_deadline_ns"] == min(deadlines), "earliest original freshness/arrival deadline")
        require(type(row["timeout_ns"]) is int and 0 < row["timeout_ns"] <= 100000000 and
                type(row["entered_ns"]) is int and type(row["ended_ns"]) is int and row["ended_ns"] >= row["entered_ns"],
                "bounded original timeout and real Queue return")
        outcome = row["outcome"]
        require(outcome in ("matching_wake", "old_wake", "returned", "original_deadline_timeout"), "successful original get outcome")
        if outcome == "matching_wake":
            require(row["returned_token"] == token, "matching value-only wake token")
        elif outcome == "old_wake":
            require(row["returned_token"] != token, "old wake is only a dispatch invalidation")
        elif outcome == "original_deadline_timeout":
            require(row["returned_token"] is None and row["ended_ns"] >= row["original_deadline_ns"], "original deadline actually reached")
        require(row["capture_registration"] is not None or outcome == "matching_wake",
                "registration closed before get needs the matching original queued wake")
        outcomes.append(outcome)
    exercised = any(x in ("matching_wake", "original_deadline_timeout") for x in outcomes)
    require(not require_wait or exercised, "no matching wake/original deadline: notification not demonstrated by deferral alone")
    return dict(status="CPU_WIRING_EVIDENCE_ONLY", notification_exercised=exercised, outcomes=outcomes,
                gpu_qualified=False, performance_claim=False)
