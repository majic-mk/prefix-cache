"""Passive initial connector-only routing around the frozen full-step observer.

No executor is implemented. The original execute method still handles every
connector notification once. Only proven zero model work before the first model
frame bypasses observation; actual model ordinals and strict 128-frame validation
remain unchanged. Importing this module uses the standard library only.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import time
import types
import weakref

ORIGINAL_COLLECTOR_SHA = "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"
MAX_INITIAL_NO_FORWARD = 4096
METADATA_MODULE = "vllm.v1.core.sched.output"


def require(value, reason):
    if not value:
        raise ValueError("INITIAL_NO_FORWARD_OBSERVER_REJECTED: " + reason)


def _checked_source(root, path, refs, expected_sha=None):
    root, path = Path(root).resolve(), Path(path).absolute()
    require(not path.is_symlink(), "source is not a symlink")
    path = path.resolve()
    require(path.is_relative_to(root) and path.is_file(), "source inside actual project")
    name = path.relative_to(root).as_posix()
    row = refs.get(name)
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
            row["path"] == name and not path.is_symlink(), "frozen source reference")
    raw = path.read_bytes()
    require(type(row["bytes"]) is int and len(raw) == row["bytes"] and
            hashlib.sha256(raw).hexdigest() == row["sha256"] and
            (expected_sha is None or row["sha256"] == expected_sha), "actual source bytes")
    return raw, row


def _original_collector(root, refs):
    path = Path(__file__).with_name("bounded_native_full_step_collector.py")
    raw, row = _checked_source(root, path, refs, ORIGINAL_COLLECTOR_SHA)
    name = "_initial_no_forward_frozen_collector_" + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, row


def _metadata_types(root, refs):
    # The original author runner has already imported these classes. Do not
    # import an alternative framework or treat fixtures as native metadata.
    module = sys.modules.get(METADATA_MODULE)
    require(type(module) is types.ModuleType, "already loaded original metadata module")
    scheduler, cached = module.SchedulerOutput, module.CachedRequestData
    require(type(scheduler) is type and type(cached) is type and
            scheduler.__module__ == cached.__module__ == METADATA_MODULE,
            "actual original metadata class identities")
    path = inspect.getsourcefile(scheduler)
    require(path is not None and inspect.getsourcefile(cached) == path and
            Path(path).resolve() == Path(module.__file__).resolve(), "one original metadata source")
    _, row = _checked_source(root, path, refs)
    return (scheduler, cached), row


def _empty(value, kind):
    return type(value) is kind and len(value) == 0


def _strict_no_work_metadata(value, metadata_types):
    """Pure copied-field check; unknown metadata always uses the old observer."""
    scheduler_type, cached_type = metadata_types
    if type(value) is not scheduler_type:
        return False
    fields = vars(value)
    if type(fields.get("total_num_scheduled_tokens")) is not int or fields["total_num_scheduled_tokens"] != 0:
        return False
    for name, kind in (("num_scheduled_tokens", dict), ("scheduled_new_reqs", list),
                       ("scheduled_spec_decode_tokens", dict), ("scheduled_encoder_inputs", dict)):
        if not _empty(fields.get(name), kind):
            return False
    cached = fields.get("scheduled_cached_reqs")
    if type(cached) is not cached_type:
        return False
    cached_fields = vars(cached)
    for name, kind in (("req_ids", list), ("resumed_req_ids", set), ("new_token_ids", list),
                       ("all_token_ids", dict), ("new_block_ids", list),
                       ("num_computed_tokens", list), ("num_output_tokens", list)):
        if not _empty(cached_fields.get(name), kind):
            return False
    if (fields.get("has_structured_output_requests") is not False or
            fields.get("pending_structured_output_tokens") is not False or
            fields.get("ec_connector_metadata", object()) is not None):
        return False
    for name in ("num_invalid_spec_tokens", "new_block_ids_to_zero"):
        item = fields.get(name, object())
        if item is not None and not _empty(item, dict if name == "num_invalid_spec_tokens" else list):
            return False
    # kv_connector_metadata and finished/free notifications are intentionally
    # untouched. The unchanged original no-forward method processes them.
    return True


def _first_call_summary(value, metadata_types, *, pristine, routed):
    """Bounded copied types/counts; no tensors, request IDs or metadata owners."""
    scheduler_type, cached_type = metadata_types
    result = dict(source_class_authenticated=type(value) is scheduler_type,
                  pristine=pristine, routed_initial_no_forward=routed,
                  total_scheduled_tokens=None, total_type=type(None).__name__,
                  work_fields={}, cached_fields={}, KV_metadata_present=False)
    if type(value) is not scheduler_type:
        return result
    fields = vars(value)
    total = fields.get("total_num_scheduled_tokens")
    result["total_type"] = type(total).__name__[:80]
    result["total_scheduled_tokens"] = total if type(total) is int else None
    result["KV_metadata_present"] = fields.get("kv_connector_metadata") is not None
    names = ("num_scheduled_tokens", "scheduled_new_reqs", "scheduled_spec_decode_tokens",
             "scheduled_encoder_inputs", "has_structured_output_requests", "pending_structured_output_tokens",
             "ec_connector_metadata", "num_invalid_spec_tokens", "new_block_ids_to_zero")
    def summary(source, keys):
        copied = {}
        for name in keys:
            item = source.get(name)
            copied[name] = dict(present=name in source, type=type(item).__name__[:80],
                length=len(item) if type(item) in (list, dict, set, tuple) else None,
                scalar=item if type(item) in (bool, int) else None)
        return copied
    result["work_fields"] = summary(fields, names)
    cached = fields.get("scheduled_cached_reqs")
    result["cached_class_authenticated"] = type(cached) is cached_type
    if type(cached) is cached_type:
        result["cached_fields"] = summary(vars(cached), ("req_ids", "resumed_req_ids", "new_token_ids",
            "all_token_ids", "new_block_ids", "num_computed_tokens", "num_output_tokens"))
    return result


class PassiveNoForwardCapture:
    """Thin capture proxy; no model, request, scheduler or native owner retained."""
    def __init__(self, inner, runner, *, metadata_types, original_collector_ref, metadata_source_ref):
        self._inner = inner
        self._runner_ref = weakref.ref(runner)
        self._metadata_types = metadata_types
        self._original_execute = inner.observer.scalar._originals[0]
        require(type(self._original_execute) is weakref.WeakMethod, "original weak bound execute method")
        self._observed_execute = runner.execute_model
        self._forward_started = False
        self._detached = False
        self._no_forward = []
        self._overflow = False
        self._restored = None
        self._first_call = None
        self._original_collector_ref = dict(original_collector_ref)
        self._metadata_source_ref = dict(metadata_source_ref)
        owner_ref = weakref.ref(self)
        observed_execute = self._observed_execute
        def execute(*args, **kwargs):
            owner = owner_ref()
            if owner is None:
                return observed_execute(*args, **kwargs)
            return owner._execute(args, kwargs)
        self._execute_wrapper = execute
        runner.execute_model = execute

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _pristine(self):
        observer, adapter = self._inner.observer, self._inner.observer.scalar._adapter
        events = observer.events
        return (not self._detached and not self._forward_started and not self._overflow and
                self._inner.valid is True and observer.enabled is True and observer.scalar.enabled is True and
                adapter.valid is True and adapter._pending is None and len(adapter.frames) == 0 and
                adapter._first_ordinal is None and events.valid is True and events.active is None and
                len(events.pending) == 0 and events.closed_steps == 0 and events.first_ordinal is None)

    def _execute(self, args, kwargs):
        # The sole original scheduler argument is passed unchanged. Unsupported
        # call shapes cannot opt out of original observation qualification.
        scheduler = args[0] if len(args) in (1, 2) and "scheduler_output" not in kwargs else kwargs.get("scheduler_output") if not args else None
        safe, pristine = False, False
        try:
            pristine = self._pristine()
            safe = pristine and _strict_no_work_metadata(scheduler, self._metadata_types)
        except Exception:
            safe = False
        if self._first_call is None:
            try:
                self._first_call = _first_call_summary(scheduler, self._metadata_types,
                    pristine=pristine, routed=safe and len(self._no_forward) < MAX_INITIAL_NO_FORWARD)
            except Exception as exc:
                self._first_call = dict(summary_unknown=type(exc).__name__[:80],
                                        routed_initial_no_forward=False, production_qualified=False)
        if not safe or len(self._no_forward) >= MAX_INITIAL_NO_FORWARD:
            self._forward_started = True
            if safe:
                self._overflow = True
            return self._observed_execute(*args, **kwargs)
        original = self._original_execute()
        require(original is not None, "live unchanged original execute")
        ordinal = getattr(original.__self__, "_profile_step", None)
        if type(ordinal) is not int or ordinal < 0:
            self._forward_started = True
            return self._observed_execute(*args, **kwargs)
        row = dict(kind="NO_FORWARD_DIAGNOSTIC", original_native_ordinal=ordinal,
                   total_scheduled_tokens=0, model_frame=False, cuda_event_created=False,
                   original_call_count=1, original_exception_type=None)
        self._no_forward.append(row)
        try:
            return original(*args, **kwargs)
        except BaseException as exc:
            row["original_exception_type"] = type(exc).__name__
            self._forward_started = True
            raise
        finally:
            row["original_native_ordinal_after"] = getattr(original.__self__, "_profile_step", None)

    def export(self):
        result = self._inner.export()
        result["initial_no_forward_observation"] = dict(
            scope="source_bound_initial_connector_only_observation_routing_v1",
            diagnostics=[dict(row) for row in self._no_forward], overflow=self._overflow,
            first_call_metadata_summary=self._first_call,
            original_collector_ref=self._original_collector_ref,
            metadata_source_ref=self._metadata_source_ref,
            actual_model_ordinals_rewritten=False, omitted_model_frames=False,
            whole_service_stream_cuda_qualified=False, production_qualified=False)
        return result

    def detach(self):
        if self._detached:
            return
        self._detached = True
        runner = self._runner_ref()
        if runner is not None and vars(runner).get("execute_model") is self._execute_wrapper:
            runner.execute_model = self._observed_execute
            self._restored = True
        else:
            self._restored = False
        self._inner.detach()


def install(worker, *, common, root, refs, run_id, event_class, selected_offsets=(16,),
            action=None, origin="native_gpu_recording", max_steps=128):
    """Delegate frozen wiring, then route only proven initial no-forward calls."""
    old, old_ref = _original_collector(root, refs)
    metadata_types, metadata_ref = _metadata_types(root, refs)
    inner = old.install(worker, common=common, root=root, refs=refs, run_id=run_id,
        event_class=event_class, selected_offsets=selected_offsets, action=action,
        origin=origin, max_steps=max_steps)
    try:
        return PassiveNoForwardCapture(inner, worker.model_runner, metadata_types=metadata_types,
            original_collector_ref=old_ref, metadata_source_ref=metadata_ref)
    except BaseException:
        inner.detach()
        raise
