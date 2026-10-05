"""Reversible, source-bound dictionary conversion of original transfer stats.

The original producer is called once. Only its newly appended record is changed;
the original reducer, Prometheus observer, aggregation and reset remain intact.
This module is stdlib-only and never grants GPU, cost or effect qualification.
"""

import dataclasses
import hashlib
import math
from pathlib import Path
import sys
import types


SOURCE_BYTES = 6195
SOURCE_SHA256 = "5ab3c65af3c0f78b0696bcc60c68ecfe51332c6d5f6684bb2cfa0e0ea9677d73"
MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
_MARKER = object()


class StatsCompatibilityError(ValueError):
    """Source, installation or record shape is outside the fixed contract."""


def _require(value, reason):
    if not value:
        raise StatsCompatibilityError(reason)


def _checked_source(source_path):
    path = Path(source_path)
    _require(not path.is_symlink() and path.is_file(), "regular original metrics source required")
    _require(path.stat().st_size == SOURCE_BYTES, "original metrics source size drift")
    raw = path.read_bytes()
    _require(len(raw) == SOURCE_BYTES and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256,
             "original metrics source SHA drift")
    return path.resolve(), raw


def _class_code(module_code, class_name):
    matches = [c for c in module_code.co_consts
               if type(c) is types.CodeType and c.co_name == class_name]
    _require(len(matches) == 1, "unique original metrics class source required")
    return matches[0]


def _method_code(class_code, method_name):
    matches = [c for c in class_code.co_consts
               if type(c) is types.CodeType and c.co_name == method_name]
    _require(len(matches) == 1, "unique original metrics method source required")
    return matches[0]


def _checked_method(cls, method_name, expected, namespace, path):
    fn = vars(cls).get(method_name)
    _require(type(fn) is types.FunctionType and fn.__globals__ is namespace,
             "original metrics method/globals required")
    _require(Path(fn.__code__.co_filename).resolve() == path and fn.__code__ == expected,
             "original metrics method code/source drift")
    defaults = fn.__defaults__
    defaults_match = (type(defaults) is tuple and len(defaults) == 1
                      and type(defaults[0]) is int and defaults[0] == 0) if method_name == "observe" else defaults is None
    _require(fn.__closure__ is None and defaults_match and fn.__kwdefaults__ is None,
             "unmodified original metrics method binding required")
    return fn


def _checked_module(module, source_path):
    path, raw = _checked_source(source_path)
    _require(type(module) is types.ModuleType and vars(module).get("__name__") == MODULE_NAME
             and sys.modules.get(MODULE_NAME) is module, "exact registered original metrics module required")
    namespace = vars(module)
    file_name = namespace.get("__file__")
    _require(type(file_name) is str and Path(file_name).resolve() == path,
             "original metrics module file required")
    stats = namespace.get("OffloadingConnectorStats")
    operation = namespace.get("OffloadingOperationMetrics")
    prom = namespace.get("OffloadPromMetrics")
    _require(all(type(c) is type and c.__module__ == MODULE_NAME
                 for c in (stats, operation, prom)), "original metrics classes required")
    _require(dataclasses.is_dataclass(operation) and
             tuple(f.name for f in dataclasses.fields(operation)) == ("op_size", "op_time"),
             "original operation dataclass fields required")
    annotations = vars(operation).get("__annotations__")
    _require(type(annotations) is dict and set(annotations) == {"op_size", "op_time"}
             and annotations["op_size"] is int and annotations["op_time"] is float,
             "original operation scalar annotations required")
    compiled = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
    stats_code = _class_code(compiled, "OffloadingConnectorStats")
    methods = {name: _checked_method(stats, name, _method_code(stats_code, name), namespace, path)
               for name in ("record_transfer", "reset", "aggregate", "reduce", "is_empty")}
    _checked_method(prom, "observe", _method_code(_class_code(compiled, "OffloadPromMetrics"), "observe"),
                    namespace, path)
    return stats, operation, methods, path


def _scalar_record(record):
    _require(type(record) is dict and len(record) == 2
             and all(type(key) is str for key in record)
             and set(record) == {"op_size", "op_time"}, "exact two-field transfer record required")
    size, elapsed = record["op_size"], record["op_time"]
    _require(type(size) is int and size >= 0 and type(elapsed) in (int, float)
             and math.isfinite(elapsed) and elapsed >= 0, "finite nonnegative original scalar metrics required")


def _before(data):
    _require(type(data) is dict, "original stats dictionary required")
    snapshot = {}
    for key, ops in data.items():
        _require(type(key) is str and type(ops) is list, "original transfer key/list required")
        for op in ops:
            _scalar_record(op)
        # Local references are discarded after this one original call.
        snapshot[key] = (ops, tuple(ops))
    return snapshot


class StatsDictCompatibility:
    def __init__(self, stats, operation, methods, path):
        self._stats = stats
        self._operation = operation
        self._original = methods["record_transfer"]
        self._path = path
        self._active = True
        original = self._original
        operation_class = operation

        def record_transfer(instance, num_bytes, time, transfer_type):
            snapshot = None
            capture_failed = False
            data = None
            try:
                _require(type(instance) is stats, "exact original stats instance required")
                data = instance.data
                snapshot = _before(data)
            except Exception:
                # Do not replace an exception from the original producer.
                capture_failed = True
            result = original(instance, num_bytes, time, transfer_type)
            _require(not capture_failed, "unexpected pre-existing stats schema")
            _require(type(transfer_type) is tuple and len(transfer_type) == 2
                     and all(type(v) is str for v in transfer_type), "original transfer type scalar pair required")
            key = transfer_type[0] + "_to_" + transfer_type[1]
            _require(instance.data is data and type(data) is dict,
                     "original producer replaced stats dictionary")
            _require(set(data) == set(snapshot) | {key}, "original producer changed unrelated transfer keys")
            for old_key, (old_list, old_records) in snapshot.items():
                now = data[old_key]
                expected_length = len(old_records) + (old_key == key)
                _require(now is old_list and len(now) == expected_length
                         and all(now[i] is record for i, record in enumerate(old_records)),
                         "original producer changed previous transfer observations")
            appended = data[key]
            _require(type(appended) is list and len(appended) == (len(snapshot[key][1]) + 1 if key in snapshot else 1),
                     "exactly one original transfer append required")
            op = appended[-1]
            _require(type(op) is operation_class and original.__globals__.get("OffloadingOperationMetrics") is operation_class,
                     "new record is not the original operation dataclass")
            values = vars(op)
            _scalar_record(values)
            # Copy only the two unchanged numeric values. No serialization,
            # replay, second producer call, metric deletion or transfer occurs.
            appended[-1] = {"op_size": values["op_size"], "op_time": values["op_time"]}
            return result

        record_transfer.__name__ = "record_transfer"
        record_transfer.__wrapped__ = original
        record_transfer._stats_dict_compat_marker = _MARKER
        self._wrapper = record_transfer
        stats.record_transfer = record_transfer

    def witness(self):
        return dict(source_bytes=SOURCE_BYTES, source_sha256=SOURCE_SHA256,
                    source_path=str(self._path), installed=self._active,
                    original_record_calls_per_invocation=1,
                    original_consumers_unchanged=True, GPU_operations=0,
                    production_qualified=False, GPU_qualified=False,
                    release_qualified=False, cost_qualified=False, effect_qualified=False)

    def detach(self):
        current = vars(self._stats).get("record_transfer")
        if not self._active:
            _require(current is self._original, "foreign record method after detach")
            return self.witness()
        _require(current is self._wrapper, "foreign record method override; preserved without overwrite")
        self._stats.record_transfer = self._original
        self._active = False
        return self.witness()


def install_stats_dict_compat(module, source_path):
    """Validate the pinned original classes, then install only the producer wrapper.

    The caller installs this inside its separately authorized guarded process and
    must detach it in finally. Duplicate installation and source drift fail before
    mutation. The witness describes this adapter, not native/GPU qualification.
    """
    stats, operation, methods, path = _checked_module(module, source_path)
    return StatsDictCompatibility(stats, operation, methods, path)
