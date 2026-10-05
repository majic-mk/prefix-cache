"""One actual public receipt per independently guarded diagnostic process.

This caches the typed receipt object, never an authorization or PASS flag.
Default controller APIs retain complete public replay. Only this explicit,
source-bound, PID/session/reservation-bound object enables adjacent internal
entry checks to reuse the object. Every reuse checks actual live metadata and
original guard; independent whole-source before/after proofs remain required.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import time

_TOKEN = object()
_CANONICAL_NAME = 'prefix_io_control.p4_single_file_receipt'
_D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'


def require(condition, message):
    if not condition:
        raise ValueError('PROCESS_ENTRY_REJECTED: ' + message)


def process_identity():
    require(os.name == 'posix', 'actual original Linux process required')
    return dict(pid=os.getpid(), sid=os.getsid(0), process_group=os.getpgid(0))


def validate_identity(expected, observed):
    require(type(expected) is dict and type(observed) is dict and
        set(expected) == set(observed) == {'pid', 'sid', 'process_group'} and
        all(type(observed[key]) is int and observed[key] > 0 and type(expected[key]) is int and
            observed[key] == expected[key] for key in expected), 'actual PID/session/process group changed')


def guard_identity(guard):
    keys = ('id', 'session_id', 'process_group', 'runner_pid', 'label', 'gpu_uuid', 'permissions')
    require(type(guard) is dict and all(key in guard for key in keys), 'complete original guard identity')
    require(type(guard['id']) is str and guard['id'] and
        all(type(guard[key]) is int and guard[key] > 0 for key in ('session_id', 'process_group', 'runner_pid')),
        'actual original reservation/process identity')
    return {key: deepcopy(guard[key]) for key in keys}


def same_guard(expected, actual):
    observed = guard_identity(actual)
    require(all(type(observed[key]) is type(expected[key]) and observed[key] == expected[key]
        for key in expected), 'current actual reservation/permission/source changed')


class ProcessEntryContext:
    def __init__(self, *, token=None, controller=None, root=None, config=None, refs=None,
                 authority=None, receipt=None, guard=None, timing=None):
        require(token is _TOKEN, 'use the actual source-verified create API; no fixture context promotion')
        self._controller = controller
        self._root = controller.project(root)
        self._config = deepcopy(config)
        self._refs = deepcopy(refs)
        self._authority = deepcopy(authority)
        self._receipt = receipt
        self._identity = process_identity()
        self._guard_identity = guard_identity(guard)
        self._source_lock_ref = controller.ref(controller.LOCK, root)
        self._closure_digest = controller.closure_digest(refs)
        self._canonical = sys.modules[_CANONICAL_NAME]
        self._receipt_projection = self._projection()
        self._public_loader_calls = 1
        self._validation_count = 0
        self._timing = deepcopy(timing)

    @classmethod
    def create(cls, controller, root, config_file):
        require(getattr(controller, 'D', None) == _D and controller.__name__ in sys.modules and
            sys.modules[controller.__name__] is controller and
            Path(controller.__file__).resolve() == controller.safe(_D + '/control_p4_single_file.py', root).resolve(),
            'one actual independently frozen diagnostic controller')
        require(not any(name == item or name.startswith(item + '.') for name in sys.modules
            for item in ('torch', 'vllm', 'py_kvcache')), 'validate actual entry before framework/model imports')
        start = time.perf_counter_ns()
        config, refs, authority = controller.verify_metadata_configuration(root, config_file)
        controller.verify_reference(refs[_D + '/control_p4_single_file.py'], root)
        controller.verify_reference(refs[_D + '/process_entry_context.py'], root)
        require(Path(cls.create.__func__.__code__.co_filename).resolve() ==
            controller.safe(_D + '/process_entry_context.py', root).resolve(), 'actual compiled context source')
        guard = controller._verify_active_guard_metadata(root, config, authority)
        public_start = time.perf_counter_ns()
        receipt = controller.load_receipt(root)
        public_end = time.perf_counter_ns()
        canonical = controller.canonical_module(root, refs)
        require(type(receipt) is canonical.ExactSingleFileReceipt and
            receipt.binding_ref.mapping() == refs[controller.BINDING] and receipt.signature[1] == config['gpu_uuid'],
            'one actual public typed receipt for the same source/device')
        after_config, after_refs, after_authority = controller.verify_metadata_configuration(root, config_file)
        require(after_config == config and after_refs == refs and after_authority == authority,
            'actual metadata changed during full public replay')
        after_guard = controller._verify_active_guard_metadata(root, config, authority)
        same_guard(guard_identity(guard), after_guard)
        end = time.perf_counter_ns()
        return cls(token=_TOKEN, controller=controller, root=root, config=config, refs=refs,
            authority=authority, receipt=receipt, guard=after_guard,
            timing=dict(full_entry_start_ns=start, full_entry_end_ns=end,
                public_loader_start_ns=public_start, public_loader_end_ns=public_end))

    def _projection(self):
        value = self._receipt
        return (type(value), value.signature, value.cost_upper_ns, value.step_budget_ns,
            value.binding_ref.mapping(), value.calibration_source_lock_sha256,
            tuple(row.mapping() for row in value.runtime_common_refs),
            tuple(row.mapping() for row in value.runtime_overlay_refs),
            tuple(row.mapping() for row in value.common_source_refs), value.cuda_event_source_sha256,
            value.calibration_native_source_sha256, value.condition_only, value.production_qualified)

    def configuration(self, root, config_file, *, controller):
        require(controller is self._controller and controller.project(root) == self._root,
            'process context cannot cross controller/root')
        validate_identity(self._identity, process_identity())
        require(controller.ref(controller.LOCK, root) == self._source_lock_ref,
            'immutable source-lock bytes changed inside original process')
        config, refs, authority = controller.verify_metadata_configuration(root, config_file)
        require(config == self._config and refs == self._refs and authority == self._authority and
            controller.closure_digest(refs) == self._closure_digest, 'source/config/authority context changed')
        guard = controller._verify_active_guard_metadata(root, config, authority)
        self.validate_guard(guard)
        require(sys.modules.get(_CANONICAL_NAME) is self._canonical and
            type(self._receipt) is self._canonical.ExactSingleFileReceipt and
            self._projection() == self._receipt_projection, 'actual public class/object/finite values changed')
        self._validation_count += 1
        return deepcopy(config), deepcopy(refs), deepcopy(authority)

    def validate_guard(self, actual):
        validate_identity(self._identity, process_identity())
        same_guard(self._guard_identity, actual)

    def receipt(self, root, config_file, *, controller):
        self.configuration(root, config_file, controller=controller)
        return self._receipt

    def evidence(self):
        return dict(schema='c5_source_bound_process_entry_context_v1', diagnostic_index=self._config['diagnostic_index'],
            source_lock_ref=deepcopy(self._source_lock_ref), config_ref=self._controller.ref(
                self._controller.config_path(self._config['diagnostic_index']), self._root),
            authority_ref=self._controller.ref(self._controller.authority_path(self._config['diagnostic_index']), self._root),
            binding_ref=self._receipt.binding_ref.mapping(), actual_identity=deepcopy(self._identity),
            actual_guard_identity=deepcopy(self._guard_identity), public_loader_calls=self._public_loader_calls,
            internal_metadata_revalidations=self._validation_count, same_actual_public_type_and_object=True,
            whole_source_before_after_required=True, internal_whole_asset_rehash=False,
            source_and_authority_rechecked=True, cached_PASS_or_qualification=False,
            private_receipt_issuer_used=False, cost_gate_changed=False, strategy_effect_verified=False,
            startup_timing_scope='host startup metadata only; not per-step or strategy cost',
            startup_stage_times=deepcopy(self._timing))
