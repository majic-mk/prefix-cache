"""Small original host metadata methods; never wrap GPU execute/sample/step."""
from __future__ import annotations
import inspect
from pathlib import Path
import types
import weakref


class HostBoundaryBinding:
    def __init__(self, observer, *, llm, worker, capture, root, refs, driver, finite_binding=None):
        self.observer, self._runner = observer, weakref.ref(worker.model_runner)
        self.bindings = []
        self.controller_id = "frontend_progress_bookkeeping"
        self.failure = None
        try:
            scheduler = llm.llm_engine.engine_core.engine_core.scheduler
            self.attach(scheduler, "schedule", "scheduler", "original_scheduler_schedule", False, root, refs, driver)
            self.attach(worker.model_runner.input_batch, "refresh_metadata", "sampling", "original_sampling_refresh_metadata", True, root, refs, driver)
            self.attach(llm.llm_engine.output_processor, "process_outputs", "output", "original_output_process_outputs", True, root, refs, driver)
            if finite_binding is not None:
                driver.require(finite_binding.host_observer is observer,
                               "actual qualified finite controller host hook installed only by original native owner")
        except BaseException:
            self.detach()
            raise

    def ordinal(self, after_execute):
        runner = self._runner()
        value = None if runner is None else getattr(runner, "_profile_step", None)
        if type(value) is not int or value < int(after_execute):
            # HostControlObserver records a bounded invalid-ordinal failure
            # and still calls the original method exactly once.
            return None
        return value - int(after_execute)

    def record_invalid_ordinal(self, reason):
        # This is optional scalar diagnostics only. Even a broken diagnostics
        # object cannot prevent the unchanged original method from running.
        try:
            with self.observer._lock:
                self.observer._fail("host original ordinal unavailable: " + str(reason)[:120])
        except Exception:
            self.failure = "host ordinal unavailable"

    def attach(self, target, name, category, boundary, after_execute, root, refs, driver,
               allow_existing_instance_method=False):
        original = getattr(target, name)
        driver.require(callable(original) and (name not in vars(target) or allow_existing_instance_method), "actual original small host boundary")
        function = original.__func__ if inspect.ismethod(original) else original
        relative = Path(function.__code__.co_filename).resolve().relative_to(root).as_posix()
        driver.require(relative in refs, "actual host boundary source frozen")
        driver.check_ref(root, refs[relative])
        absolute = dict(refs[relative], path=str(root / relative))
        kinds = dict(scheduler="scheduler_metadata", sampling="sampling_metadata_preparation",
                     output="output_processing", controller="controller_bookkeeping")
        self.observer.bind(boundary, category, original, source_ref=absolute, boundary_kind=kinds[category])
        existed = name in vars(target)
        previous = vars(target).get(name)
        def wrapper(this, *args, **kwargs):
            try:
                ordinal = self.ordinal(after_execute)
                if ordinal is None:
                    raise ValueError("missing or invalid original ordinal")
            except Exception as exc:
                self.record_invalid_ordinal(exc)
                return original(*args, **kwargs)
            return self.observer.call(category, ordinal, boundary, original, *args, **kwargs)
        value = types.MethodType(wrapper, target)
        setattr(target, name, value)
        self.bindings.append((weakref.ref(target), name, original, value, existed, previous))

    def bind_frontend_progress(self, original, *, root, refs, driver):
        relative = Path(original.__code__.co_filename).resolve().relative_to(root).as_posix()
        driver.require(relative in refs, "same frontend bookkeeping source frozen")
        driver.check_ref(root, refs[relative])
        self.observer.bind(self.controller_id, "controller", original,
            source_ref=dict(refs[relative], path=str(root / relative)), boundary_kind="controller_bookkeeping")
        def wrapper(rows, step):
            try:
                ordinal = self.ordinal(True)
                if ordinal is None:
                    raise ValueError("missing or invalid original ordinal")
            except Exception as exc:
                self.record_invalid_ordinal(exc)
                return original(rows, step)
            return self.observer.call("controller", ordinal, self.controller_id, original, rows, step)
        return wrapper

    def detach(self):
        okay = True
        for reference, name, original, wrapper, existed, previous in reversed(self.bindings):
            target = reference()
            if target is None:
                continue
            if vars(target).get(name) is wrapper:
                if existed:
                    setattr(target, name, previous)
                else:
                    delattr(target, name)
            else:
                okay = False
        self.bindings.clear()
        return okay
