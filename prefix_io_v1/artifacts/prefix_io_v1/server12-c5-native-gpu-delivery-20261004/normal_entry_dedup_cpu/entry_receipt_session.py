"""Completed-record CPU prototype; never a GPU entry or scheduling permission.

One public canonical loader issues one actual ExactSingleFileReceipt. Each
subsequent get fully rehashes the original closure and independently rechecks
the completed guard, current authority, SDK, ledger, and process identity.
No callable, module global, original file, numerical gate, or receipt issuer
is replaced. Active-job integration deliberately remains unimplemented.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
CONTROL = dict(path=D + '/control_p4_single_file.py', bytes=55526,
    sha256='1bdc4428d8eaf9bb57d38b31717a08de2daea98edec1d0b74de634f30c2f0510')
COMMON_SHA = 'aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d'
CANONICAL_NAME = 'prefix_io_control.p4_single_file_receipt'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
IMPORT_ATTEMPTS = []


def require(condition, message):
    if not condition:
        raise ValueError('DEDUP_CPU_REJECTED: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'explicit bounded relative path')
    root = Path(root).resolve(strict=True)
    cursor = root
    for part in relative.split('/'):
        cursor /= part
        require(not cursor.is_symlink(), 'symlink refused')
    require(cursor.resolve().is_relative_to(root), 'outside project')
    return cursor


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'regular file required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest)


def verify_ref(root, reference):
    require(type(reference) is dict and set(reference) == {'path', 'bytes', 'sha256'}
        and type(reference['bytes']) is int and reference['bytes'] >= 0
        and type(reference['sha256']) is str and len(reference['sha256']) == 64
        and all(c in '0123456789abcdef' for c in reference['sha256']), 'exact reference schema')
    require(file_ref(root, reference['path']) == reference, 'immutable bytes changed: ' + reference['path'])
    return reference


def verify_integrity(root, references):
    """Generic CPU byte proof, not a native receipt or authorization API."""
    require(type(references) is dict and references, 'nonempty fixed closure')
    for key, reference in references.items():
        require(key == reference.get('path'), 'row-key path mismatch')
        verify_ref(root, reference)
    return hashlib.sha256(canonical([references[key] for key in sorted(references)]).encode()).hexdigest()


def process_identity():
    require(os.name == 'posix', 'actual Linux process identity required; no active GPU integration')
    return dict(pid=os.getpid(), sid=os.getsid(0), process_group=os.getpgid(0))


def validate_process_identity(expected, observed):
    require(type(expected) is dict and type(observed) is dict and
        set(expected) == set(observed) == {'pid', 'sid', 'process_group'} and
        all(type(observed[key]) is int and observed[key] > 0 and
            type(expected[key]) is int and observed[key] == expected[key] for key in expected),
        'receipt reuse cannot cross PID/session/process-group')
    return True


def load_pinned(root, reference, name):
    verify_ref(root, reference)
    require(name not in sys.modules, 'fresh pinned CPU module required')
    path = safe(root, reference['path'])
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        verify_ref(root, reference)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


class CPUOnlyImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('model/GPU import refused: ' + fullname)
        return None


class CompletedReceiptSession:
    """Only completed off01 metadata; cached value is a receipt, never PASS.

    The constructor has no receipt/loader/clock/guard injection parameter.
    A failed migration report remains failed. This object cannot authorize
    shadow/on or validate an active guard and has no launch method.
    """
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        require(self.root == ROOT and not IMPORT_ATTEMPTS and not any(
            name == base or name.startswith(base + '.') for name in sys.modules for base in FORBIDDEN),
            'actual fixed project in clean CPU-only process')
        self.identity = process_identity()
        self.C = load_pinned(self.root, CONTROL, '_completed_entry_dedup_original_control')
        C = self.C
        require(C.D == D and C.ref(C.LOCK, self.root)['sha256'] == COMMON_SHA,
            'actual frozen controller/common closure required')
        self.lock_ref = C.ref(C.LOCK, self.root)
        _, self.refs = C.common_rows(self.root, verify_bytes=True)
        require(self.refs[CONTROL['path']] == CONTROL, 'controller inside exact original source closure')
        self.config = C.read(C.config_path('off'), self.root)
        C.validate_configuration_document(self.root, self.config, C.config_path('off'))
        self.authority = C.load_authority(self.root, self.config)
        self.history = load_pinned(self.root, self.refs[C.HISTORY], '_completed_entry_dedup_original_history')
        self.sdk = load_pinned(self.root, self.refs[C.SITE_SDK], '_completed_entry_dedup_original_sdk')
        self.report_relative = C.result_path('off')
        self.report_ref = C.ref(self.report_relative, self.root)
        self.report = C.read(self.report_relative, self.root)
        require(self.report.get('scope') == C.SCOPE and self.report.get('mode') == 'off'
            and self.report.get('native_execution_verified') is True
            and self.report.get('runtime_condition_qualified') is False
            and self.report.get('frozen_cost_migration_pass') is False
            and self.report.get('permits_next_mode') is None
            and self.report.get('source_lock_ref') == self.lock_ref,
            'only the actual failed off01 migration, without changing its gate')
        evidence = self.report.get('evidence_refs')
        require(type(evidence) is dict and set(evidence) ==
            {'config', 'raw_result', 'actual_guard', 'source_before', 'source_after'}, 'complete completed off evidence')
        require(evidence['config']['path'] == C.config_path('off') and
            evidence['source_before']['path'] == D + '/SOURCE_off_BEFORE.json' and
            evidence['source_after']['path'] == D + '/SOURCE_off_AFTER.json' and
            evidence['actual_guard']['path'] == 'experiments/prefix_io_v1/runs/' + C.label('off') + '/result.json' and
            evidence['raw_result']['path'] == self.config['out'] + '/p4-single-file-runtime-result.json',
            'same actual completed raw/config/guard paths')
        extras = [self.lock_ref, self.report_ref, *evidence.values(),
            C.ref(C.authority_path('off'), self.root)]
        extras += [self.authority[key] for key in ('context_ref', 'human_grant_ref', 'effective_permission_ref',
            'base_permission_ref', 'guard_source_ref')]
        extras += [C.ref(D + '/' + name, self.root) for name in
            ('SCOPE_off.json', 'SOURCE_off_LAUNCH.json', 'LAUNCH_INTENT_off.json', 'LEDGER_BEFORE_off.json')]
        self.extras = {row['path']: deepcopy(row) for row in extras}
        self._verify_metadata()
        self.canonical = load_pinned(self.root, self.refs[C.RECEIPT], CANONICAL_NAME)
        require(self.canonical.ExactSingleFileReceipt.__module__ == CANONICAL_NAME and
            self.refs[C.RECEIPT]['sha256'] == C.CANONICAL_SHA, 'one actual pinned public receipt class')
        start = time.perf_counter_ns(); cpu_start = time.process_time_ns()
        self.receipt = self.canonical.load_verified_single_file(self.root, C.BINDING)
        self.public_loader_cpu_ns = time.process_time_ns() - cpu_start
        self.public_loader_wall_ns = time.perf_counter_ns() - start
        self.public_loader_calls = 1
        require(type(self.receipt) is self.canonical.ExactSingleFileReceipt
            and self.receipt.condition_only is True and self.receipt.production_qualified is False
            and self.receipt.binding_ref.mapping() == self.refs[C.BINDING]
            and self.receipt.signature[1] == self.config['gpu_uuid'], 'actual public typed receipt and same domain')
        self.receipt_projection = self._projection()
        self.get_calls = 0
        self.timings = []
        self._verify_metadata()

    def _projection(self):
        receipt = self.receipt
        return canonical(dict(signature=receipt.signature, cost_upper_ns=receipt.cost_upper_ns,
            step_budget_ns=receipt.step_budget_ns, binding_ref=receipt.binding_ref.mapping(),
            calibration_source_lock_sha256=receipt.calibration_source_lock_sha256,
            runtime_common_refs=[row.mapping() for row in receipt.runtime_common_refs],
            runtime_overlay_refs=[row.mapping() for row in receipt.runtime_overlay_refs],
            common_source_refs=[row.mapping() for row in receipt.common_source_refs],
            cuda_event_source_sha256=receipt.cuda_event_source_sha256,
            calibration_native_source_sha256=receipt.calibration_native_source_sha256))

    def _verify_metadata(self):
        C = self.C; root = self.root
        validate_process_identity(self.identity, process_identity())
        require(not IMPORT_ATTEMPTS and not any(name == base or name.startswith(base + '.')
            for name in sys.modules for base in FORBIDDEN), 'clean CPU process on each entry')
        verify_integrity(root, self.extras)
        require(C.read(C.config_path('off'), root) == self.config and
            C.load_authority(root, self.config) == self.authority, 'per-entry actual config and authority drift')
        for phase in ('before', 'launch', 'after'):
            C.verify_source_proof(root, self.config, self.authority, phase, self.refs)
        intent = C.verify_launch_intent(root, self.config, self.authority, self.refs)
        guard_relative = self.report['evidence_refs']['actual_guard']['path']
        event = C.read(guard_relative, root)
        expected = dict(label=self.config['label'], gpu_uuid=self.config['gpu_uuid'],
            command=C.native_command('off'), permissions=self.authority['effective_permission_ref'],
            evidence=str(C.safe(guard_relative.rsplit('/', 1)[0], root)), exit=0, child_exit=0,
            timed_out=False, interrupted_signal=None, error=None, gpu_job_attempted=True,
            session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[])
        require(all(C.exact(event.get(key), value) for key, value in expected.items()),
            'completed original normal guard fields/session drain')
        require(type(event.get('reservation_id')) is str and event['reservation_id'] and
            type(event.get('session_id')) is int and event['session_id'] > 0 and
            C.number(event.get('elapsed_seconds'), 'actual elapsed', 0.000001) <= 320,
            'actual completed original reservation/session/budget')
        ledger = C.read(C.LEDGER, root)
        ledger_ref = C.ref(C.LEDGER, root)
        require(ledger.get('active_reservation') is None, 'completed CPU prototype refuses any active guard')
        before = C.read(intent['ledger_before_ref']['path'], root)
        prefix = before['events']; index = len(prefix)
        require(type(ledger.get('events')) is list and ledger['events'][:index] == prefix
            and len(ledger['events']) > index and ledger['events'][index] == event and
            [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']] == [event],
            'actual unchanged prefix and first appended unique completed event')
        completed = dict(before, events=prefix + [event], active_reservation=None,
            gpu_wall_seconds=before['gpu_wall_seconds'] + event['elapsed_seconds'])
        self.history.verify_ledger_extension(completed, ledger, event,
            ledger_before_seconds=before['gpu_wall_seconds'], allow_active=False)
        # Public historical adapter reruns the original calibration guard source
        # prefix and the actual current append-only ledger relation on each get.
        cal_config, _cal_refs, cal_binding = self.history.verify_configuration(root,
            C.CALIBRATION + '/NATIVE_COST_CONFIG.json')
        self.history.verify_completed_guard(root, cal_config, cal_binding)
        _helper, assets, _rows = self.sdk.load_site_assets(root, self.refs)
        require(type(assets.get('driver')) is dict and all(C.exact(assets['driver'].get(key), value)
            for key, value in C.HOST_DRIVER_REF.items()), 'actual original complete SDK/current driver audit')
        verify_integrity(root, self.extras)
        require(C.ref(C.LEDGER, root) == ledger_ref, 'live ledger changed during completed CPU entry')
        return event

    def get(self):
        start = time.perf_counter_ns(); cpu_start = time.process_time_ns()
        validate_process_identity(self.identity, process_identity())
        C = self.C
        require(C.ref(C.LOCK, self.root) == self.lock_ref, 'common source-lock bytes changed')
        _, current = C.common_rows(self.root, verify_bytes=True)
        require(current == self.refs, 'whole original source closure changed')
        closure_digest = verify_integrity(self.root, {key: row for key, row in self.refs.items()
            if key in (C.RECEIPT, C.HISTORY, C.SITE_SDK, C.GUARD, C.BINDING, CONTROL['path'])})
        self._verify_metadata()
        require(type(self.receipt) is self.canonical.ExactSingleFileReceipt and
            self._projection() == self.receipt_projection, 'typed receipt/class/value changed')
        self.get_calls += 1
        self.timings.append(dict(entry=self.get_calls, wall_ns=time.perf_counter_ns()-start,
            cpu_ns=time.process_time_ns()-cpu_start, full_source_files_verified=len(current),
            core_ref_digest=closure_digest, public_loader_reexecuted=False))
        return self.receipt

    def verify_active_guard(self, *_args, **_kwargs):
        raise ValueError('DEDUP_CPU_REJECTED: active GPU integration is not implemented or qualified')


def benchmark(root, requests):
    require(type(requests) is int and 2 <= requests <= 8, 'bounded 2..8 completed CPU entries')
    root = Path(root).resolve(strict=True)
    ledger_before_ref = file_ref(root, 'experiments/prefix_io_v1/gpu-budget-ledger.json')
    start = time.perf_counter_ns(); cpu_start = time.process_time_ns()
    session = CompletedReceiptSession(root)
    observations = [session.get() for _ in range(requests)]
    require(all(value is session.receipt for value in observations), 'one same actual public typed object')
    ledger_after_ref = file_ref(root, 'experiments/prefix_io_v1/gpu-budget-ledger.json')
    require(ledger_before_ref == ledger_after_ref, 'CPU prototype must leave original ledger unchanged')
    return dict(status='PASS_COMPLETED_CPU_RECEIPT_REUSE_PROTOTYPE', source_lock_ref=session.lock_ref,
        binding_ref=session.receipt.binding_ref.mapping(), public_loader_calls=session.public_loader_calls,
        entry_get_calls=session.get_calls, same_public_type_and_object=True,
        public_loader_wall_ns=session.public_loader_wall_ns, public_loader_cpu_ns=session.public_loader_cpu_ns,
        entry_timings=session.timings, total_cpu_ns=time.process_time_ns()-cpu_start,
        total_wall_ns=time.perf_counter_ns()-start, GPU_operations=0, actual_GPU_jobs_started=0,
        GPU_startup_fix_integrated=False, active_guard_integration_qualified=False,
        migration_gate_changed=False, original_failed_off_qualification_retained=True,
        runtime_condition_qualified=False, permits_next_mode=None, strategy_effect_verified=False,
        full_runtime_cost_qualified=False, production_qualified=False, P4_completed=False,
        cached_PASS_or_qualification=False, direct_private_issuer_calls=0,
        ledger_before_ref=ledger_before_ref, ledger_after_ref=ledger_after_ref,
        original_gpu_ledger_unchanged=True,
        forbidden_import_attempts=IMPORT_ATTEMPTS, benchmark_scope='completed CPU metadata; no original-seven replay timing comparison')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    sys.meta_path.insert(0, CPUOnlyImports())
    try:
        result = benchmark(args.root, args.requests)
        output = safe(args.root, args.output)
        require(not output.exists(), 'append-only CPU result')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='COMPLETED_CPU_DEDUP_REJECTED', error=str(exc),
            GPU_operations=0, actual_GPU_jobs_started=0, GPU_startup_fix_integrated=False,
            runtime_condition_qualified=False, permits_next_mode=None, strategy_effect_verified=False,
            forbidden_import_attempts=IMPORT_ATTEMPTS), sort_keys=True, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
