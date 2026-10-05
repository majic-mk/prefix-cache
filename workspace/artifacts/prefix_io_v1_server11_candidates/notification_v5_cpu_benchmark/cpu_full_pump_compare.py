"""Frozen, synthetic CPU comparison of complete native reactor control methods.

No torch/vLLM/CUDA imports. The CPU backend is deliberately NOT native I/O.
Each trial is a fresh interpreter to keep arm-specific module identity separate.
"""
from __future__ import annotations
import argparse
import ast
import collections
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import weakref

REL = 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
PROTOCOL_SHA256 = '6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218'
REQUIRED = ('_run', '_pump_once', '_schedule_work', '_poll_ring_completions',
            '_drain_cuda_copies', '_flush_copy_batch', '_finish_jobs', '_intake',
            '_drain_incoming', '_drain_ready_preload_fds', '_drain_ready_load_fds',
            '_file_terminal', '_retire_accepted_parent', 'submit_job', 'request_mandatory',
            '_prefix_single_file_retry_key')
SCENARIOS = ('step_end_40ms', 'step_end_20ms', 'step_end_80ms',
             'mandatory_20ms', 'stop_20ms', 'original_deadline_100ms')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def cgroup():
    base = Path('/sys/fs/cgroup')
    result = {'platform': sys.platform, 'quota': None, 'stat': None}
    try:
        words = (base / 'cpu.max').read_text().split()
        result['quota'] = words
        result['limited'] = words[0] != 'max'
        result['stat'] = {line.split()[0]: int(line.split()[1])
                          for line in (base / 'cpu.stat').read_text().splitlines()}
    except (OSError, ValueError):
        result['limited'] = None if sys.platform.startswith('linux') else False
    return result


def load_support(candidate):
    spec = importlib.util.spec_from_file_location('_full_pump_support', candidate / 'test_single_file_retry_observation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.NSOURCE['time'] = time  # Prior CPU fixture clock is never used for scored runs.
    return module


def extract(candidate):
    path = candidate / REL
    source = path.read_bytes()
    tree = ast.parse(source.decode('utf-8-sig'))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    cls = next(n for n in classes if n.name == 'IoReactor')
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert all(name in methods for name in REQUIRED)
    mod = ModuleType('_full_pump_native_ast')
    sys.modules[mod.__name__] = mod
    g = mod.__dict__
    g.update(dataclass=dataclass, field=field, collections=collections, Future=Future,
             queue=queue, threading=threading, time=time, weakref=weakref, Any=object,
             logger=logging.getLogger('CPU_SYNTHETIC_REACTOR'), now_ns=time.monotonic_ns,
             add_event=lambda *a, **kw: None, emit_transfer_events=lambda *a, **kw: None)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + classes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), g)
    identity = {'path': str(path.resolve()), 'sha256': sha(source), 'bytes': len(source),
                'method_AST_SHA256': {n: sha(ast.dump(methods[n], include_attributes=False).encode()) for n in REQUIRED},
                'method_lines': {n: [methods[n].lineno, methods[n].end_lineno] for n in REQUIRED}}
    return g, identity


class RecordedQueue(queue.Queue):
    """Actual Queue semantics; only timestamps/counters added symmetrically."""
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.get_block_count = 0
        self.put_records = []
        self.get_records = []

    def put(self, item, block=True, timeout=None):
        self.put_records.append({'kind': self.owner.kind(item), 'at_ns': time.monotonic_ns()})
        return super().put(item, block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        if block:
            self.get_block_count += 1
        item = super().get(block=block, timeout=timeout)
        self.get_records.append({'kind': self.owner.kind(item), 'at_ns': time.monotonic_ns()})
        return item


class SyntheticRawEvent:
    def __init__(self):
        self.records = self.queries = 0
    def record(self):
        self.records += 1
        return 'CPU_SYNTHETIC_RECORD'
    def query(self):
        self.queries += 1
        return True


class SyntheticRing:
    """Read payload/latency fixture, not Linux AIO or a CUDA completion."""
    def __init__(self, owner):
        self.owner = owner
        self.pending = []
        self.poll_calls = self.submit_calls = self.completed = 0
    def poll_all(self):
        self.poll_calls += 1
        return []  # Native poll loop runs; completion is the explicit fixture below.
    def submit_pending(self):
        self.submit_calls += 1
        now = time.monotonic_ns()
        for record in list(self.pending):
            if now < record['due_ns']:
                continue
            self.pending.remove(record)
            owner = self.owner
            owner._inflight.pop(record['key'])
            owner._data_inflight -= 1
            owner._preload_inflight_remove(record['hash'])
            waiters = list(owner._preload_waiters.pop(record['hash'], ()))
            for job, file_index in waiters:
                owner._file_terminal(job, ok=True)
            owner.staging_pool.release(record['slot'])
            self.completed += 1
            owner.completed_ns = time.monotonic_ns()
            owner.completed_payload_sha256 = sha(b'CPU_SIMULATED_PREFIX_PAYLOAD_V1')
            owner.completed_event.set()
    def close(self):
        assert not self.pending


def construct(candidate, arm):
    support = load_support(candidate)
    g, identity = extract(candidate)
    Native = g['IoReactor']

    class Owner(support.Owner):
        pass

    # Bind exact original class descriptors, preserving static methods/properties.
    for name, value in Native.__dict__.items():
        if name not in ('__init__', '__dict__', '__weakref__', '__doc__', '__module__'):
            setattr(Owner, name, value)

    owner = Owner(max_wait=100000000)
    # Replace fixture dataclasses with the classes used by the extracted native methods.
    owner.ready = g['_ReadyFd'](fd=7, job=None, file_index=-1, preload_hash=support.HASH,
        open_start_ns=1, preload_info=g['_PreloadInfo'](support.HASH, 'preload', 'request', 'cpu-fixture', 0, 1), sequence=1)
    owner._ready_fds_preload = collections.deque([owner.ready])
    owner._incoming = RecordedQueue(owner)
    owner._closed = False
    owner._prefix_progress = g['_ProgressState']('r')
    owner._parent_cv = threading.Condition(owner._submit_lock)
    owner._admission_token = object()
    owner._parent_count_valid = True
    owner._accepted_parent_count = owner._parent_sequence = owner._parent_peak = 0
    owner._parent_retired_count = owner._parent_waiters = owner._parent_backpressure_waits = 0
    owner._admission_drain_pending = False
    owner._owner_snapshot_pending = 0
    owner._observation_sink = None
    owner._observation_failures = 0
    owner._prefix_start_budget = owner._prefix_store_order = owner._prefix_dispatch_shadow = None
    owner._share_preload = False
    owner._break_even = NS(enabled=False)
    owner._max_preload_slots = 16
    owner._max_open_inflight = 4
    owner._preload_waiter_limit = 64
    owner._preload_pending_count = collections.Counter()
    owner._preload_pending_cancel = collections.Counter()
    owner._preload_owned = {}
    owner._preload_blocked_on_write = {}
    owner._preload_refcount = {}
    owner._store_inflight = {}
    owner._known_missing = collections.OrderedDict()
    owner._streams = []
    owner._copy_stream = None
    owner.completed_event = threading.Event()
    owner.completed_ns = owner.accepted_ns = None
    owner.completed_payload_sha256 = None
    owner.counts = collections.Counter()
    owner.policy_preview_reasons = collections.Counter()
    owner.failures = []
    owner.mandatory_job = None
    owner.ring = SyntheticRing(owner)
    owner.bridge.policy.config = support.support.P4Config('interference', 100000000, 100000000, 100)
    # Ring accounting is deliberately a scalar synthetic fixture; identity/zero
    # checks still execute in the original policy path, no production receipt.
    owner._prefix_stage_accounting._owner = lambda: None
    def fail_everything(exc):
        owner.failures.append(type(exc).__name__ + ': ' + str(exc))
        owner._stop = True
        owner.completed_event.set()
    owner._fail_everything = fail_everything

    def submit_read(ready, slot_index, **kwargs):
        owner.counts['physical_fixture_submit'] += 1
        assert owner.accepted_ns is None, 'duplicate fixture submit'
        owner.accepted_ns = time.monotonic_ns()
        owner.submitted.append((ready.sequence, slot_index))
        key = 99001
        owner._inflight[key] = NS(job=None, op_kind='read', preload_hash=support.HASH)
        owner._data_inflight += 1
        owner.ring.pending.append({'key': key, 'hash': support.HASH, 'slot': slot_index,
                                   'due_ns': owner.accepted_ns + 1000000})
    owner._submit_read_from_ready = submit_read

    # Transparent call counters, not alternative control logic. Costs are included.
    counted = ('_pump_once', '_schedule_work', '_drain_cuda_copies', '_poll_ring_completions',
               '_flush_copy_batch', '_finish_jobs', '_intake', '_prefix_p4_collect',
               '_drain_incoming', '_drain_ready_preload_fds', '_drain_ready_load_fds',
               '_prefix_ready_read_decision', '_prefix_stage_decide', '_prefix_capacity_components',
               '_prefix_single_file_retry_key',
               '_file_terminal', '_retire_accepted_parent')
    for name in counted:
        original = getattr(owner, name)
        def wrapped(*a, _n=name, _f=original, **kw):
            owner.counts[_n] += 1
            return _f(*a, **kw)
        setattr(owner, name, wrapped)
    original_preview = owner.bridge.preview_issue
    def preview(*a, **kw):
        owner.counts['bridge_preview'] += 1
        result = original_preview(*a, **kw)
        owner.policy_preview_reasons[result.reason] += 1
        return result
    owner.bridge.preview_issue = preview
    from prefix_io_control.p4_types import SystemSnapshot
    original_fresh = SystemSnapshot.fresh
    def freshness_check(frame, *a, **kw):
        owner.counts['SystemSnapshot.fresh'] += 1
        return original_fresh(frame, *a, **kw)
    SystemSnapshot.fresh = freshness_check

    collector = support.support.collector
    capture = collector.FullStepCapture(run_id='r', origin='native_gpu_recording', selected_offsets=(1,), action=None)
    start = collector.EventProxy(SyntheticRawEvent(), time.monotonic_ns)
    end = collector.EventProxy(SyntheticRawEvent(), time.monotonic_ns)
    frame = NS(step_kind='decode', batch=1, active_decode=1, prefill_tokens=0, context_length=144)
    pending = {'phase': 'prepared', 'ordinal': 5, 'frame': frame}
    capture.observer = NS(enabled=True, scalar=NS(enabled=True, _adapter=NS(_pending=pending)),
                          events=NS(active=(5, start, end)), frames=[], detach=lambda: None)
    spec = importlib.util.spec_from_file_location('_full_pump_identity', candidate / 'single_file_runtime_binding.py')
    binding = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = binding
    spec.loader.exec_module(binding)
    class SyntheticRunner:
        pass
    runner = SyntheticRunner()
    capture.runner_ref = weakref.ref(runner)
    runtime_identity = object.__new__(binding.RuntimeSingleFileIdentity)
    object.__setattr__(runtime_identity, 'signature_prefix', owner.bridge.policy.single_file.signature[:4])
    object.__setattr__(runtime_identity, '_runner_ref', weakref.ref(runner))
    owner.bridge._single_file_capture = weakref.ref(capture)
    owner.bridge._single_file_runtime = runtime_identity
    del owner.bridge._single_file_state  # Restore exact weakref/runner identity borrow.
    original_live = owner.bridge._single_file_state
    def live():
        owner.counts['live_borrow'] += 1
        return original_live()
    owner.bridge._single_file_state = live
    if arm == 'B':
        capture.attach_single_file_wait(owner)
    owner.capture = capture
    owner.capture_start = start
    owner.capture_end = end
    owner.synthetic_runner_owner = runner
    owner.native_g = g
    owner.ready_hash = support.HASH
    owner.kind = lambda item: ('STOP' if item is owner._STOP else type(item).__name__)
    identity['physical_fixtures'] = {
        '_submit_read_from_ready': 'scheduled CPU payload with 1ms completion; no file/AIO/H2D',
        'ring.poll_all': 'empty native CQE list; original poll method still executes',
        'ring.submit_pending': 'CPU fixture completion invokes original _file_terminal and _preload_inflight_remove',
        'staging_pool': 'retry fixture scalar reserves/releases, no pinned memory',
        'CUDA_events': 'SyntheticRawEvent; native_gpu_recording is only the existing accessor branch selector',
        'stage_accounting': 'synthetic zero-inflight scalar records; no production qualification',
        'add_event_and_emit_transfer_events': 'no-op profiling sinks, equal both arms',
        '_fail_everything': 'fixture failure capture, not native exceptional drain qualification'}
    identity['transparent_count_wrappers'] = list(counted) + ['bridge.preview_issue', 'bridge._single_file_state', 'SystemSnapshot.fresh']
    identity['freshness_counter_scope'] = 'Calls of original SystemSnapshot.fresh and original retry-key method; inline age predicates are not separately counted.'
    identity['unexecuted_physical_branches'] = ['real ring CQE completion', 'real CUDA copy construction/query', 'actual disk/pinned/GPU payload']
    identity['actual_collector'] = {'path': str(Path(collector.__file__).resolve()),
        'sha256': sha(Path(collector.__file__).read_bytes()),
        'accessor_code_filename': str(Path(collector.FullStepCapture.current_single_file_step.__code__.co_filename).resolve())}
    assert identity['actual_collector']['path'] == str((candidate / 'native_full_step_collector.py').resolve())
    assert identity['actual_collector']['accessor_code_filename'] == identity['actual_collector']['path']
    identity['actual_original_method_code_files'] = {name: str(Path(getattr(Native, name).__code__.co_filename).resolve()) for name in REQUIRED}
    assert set(identity['actual_original_method_code_files'].values()) == {str((candidate / REL).resolve())}
    identity['fixture_helpers'] = {name: sha((candidate / name).read_bytes()) for name in
                                  ('test_single_file_retry_observation.py', 'test_single_file_policy.py')}
    return owner, identity


def run_trial(args):
    assert not any(n == 'torch' or n == 'vllm' or n.startswith(('torch.', 'vllm.')) for n in sys.modules)
    owner, identity = construct(Path(args.candidate_root).resolve(), args.arm)
    producer_data = {}
    worker_data = {}
    launch = threading.Barrier(3)
    t0_box = []
    completed_once = []
    original_complete_set = owner.completed_event.set
    def completed_set():
        completed_once.append(time.monotonic_ns())
        original_complete_set()
    owner.completed_event.set = completed_set

    def until_ns(deadline):
        while True:
            delta = deadline - time.monotonic_ns()
            if delta <= 0:
                return
            time.sleep(delta / 1e9)

    def worker():
        launch.wait()
        cpu0 = time.thread_time_ns()
        wall0 = time.monotonic_ns()
        owner._run()
        worker_data.update(cpu_ns=time.thread_time_ns() - cpu0, wall_start_ns=wall0, wall_end_ns=time.monotonic_ns())

    def producer():
        cpu0 = time.thread_time_ns()
        try:
            # The one shared startup barrier publishes preparation. Per-step
            # registration runs on (and is charged to) this producer thread.
            prepare_cpu0 = time.thread_time_ns()
            if hasattr(owner.capture, 'after_prepare'):
                owner.capture.after_prepare()
            producer_data['prepare_callback_cpu_ns'] = time.thread_time_ns() - prepare_cpu0
            producer_data['prepare_callback_included_in_producer_cpu'] = True
            launch.wait()
            t0 = t0_box[0]
            scenario = args.scenario
            if scenario.startswith('step_end_'):
                ns = int(scenario.split('_')[-1].removesuffix('ms')) * 1000000
                deadline = t0 + ns
                until_ns(deadline)
                producer_data.update(event_planned_ns=deadline, event_actual_ns=time.monotonic_ns())
                owner.capture_end.record()
            elif scenario in ('mandatory_20ms', 'stop_20ms'):
                deadline = t0 + 20000000
                until_ns(deadline)
                producer_data.update(event_planned_ns=deadline, event_actual_ns=time.monotonic_ns())
                if scenario == 'mandatory_20ms':
                    job = owner.native_g['_ReactorJob'](job_id=1, is_store=False, block_hashes=[owner.ready_hash],
                        block_chunks=[(0,)], profile=NS(req_id='mandatory-cpu', profile_tid='cpu', start_ns=time.monotonic_ns()),
                        future=Future(), total_files=1, transfer_size=917504, num_blocks=1)
                    owner.mandatory_job = job
                    owner.submit_job(job)
                    assert owner.request_mandatory([job.future])
                else:
                    owner.shutdown(wait=False)
                until_ns(t0 + 80000000)
                owner.capture_end.record()
            else:
                assert scenario == 'original_deadline_100ms'
                producer_data.update(event_planned_ns=t0 + 100000000, event_actual_ns=None)
                until_ns(t0 + 120000000)
                owner.capture_end.record()
            if not owner.completed_event.wait(1.0):
                raise RuntimeError('synthetic completion missing')
            owner.shutdown(wait=False)
        except BaseException as exc:
            producer_data['error'] = type(exc).__name__ + ': ' + str(exc)
            launch.abort()
            owner._incoming.put(owner._STOP)
        finally:
            producer_data['cpu_ns'] = time.thread_time_ns() - cpu0
            producer_data['finished_ns'] = time.monotonic_ns()

    owner._worker = threading.Thread(target=worker, name='native-reactor-CPU', daemon=True)
    prod = threading.Thread(target=producer, name='external-step-CPU', daemon=True)
    # Fixed premeasurement settling interval: setup must not consume the quota
    # period used by the measured threads. It is not a reactor policy sleep.
    time.sleep(0.2)
    before = cgroup()
    # Shared initial state published before threads and the only startup barrier.
    t0 = time.monotonic_ns()
    t0_box.append(t0)
    owner.ready.open_start_ns = t0
    owner.capture_start.record_before_ns = t0 - 30
    owner.capture_start.record_after_ns = t0 - 20
    owner.capture_start.completed_query_ns = t0 - 10
    owner._worker.start()
    prod.start()
    launch.wait()
    owner._worker.join(2.0)
    prod.join(max(0, 2.0 - (time.monotonic_ns() - t0) / 1e9))
    end_ns = time.monotonic_ns()
    after = cgroup()
    failures = list(owner.failures)
    if owner._worker.is_alive() or prod.is_alive():
        failures.append('2-second watchdog/thread not drained')
    if producer_data.get('error'):
        failures.append(producer_data['error'])
    if owner.ring.completed != 1 or len(owner.submitted) != 1 or len(completed_once) != 1:
        failures.append('simulated read/complete not exactly once')
    if owner.staging_pool.free_count != 16 or owner.staging_pool.reserves != owner.staging_pool.releases:
        failures.append('synthetic staging accounting not balanced')
    if owner._has_work() or owner._inflight or owner._accepted_parent_count or not owner._incoming.empty():
        failures.append('native ownership/queue did not drain')
    if owner.bridge.fault:
        failures.append('bridge fault: ' + owner.bridge.fault)
    if owner.mandatory_job is not None:
        job = owner.mandatory_job
        if not job.future.done() or job.future.exception() or job.future.result() != 917504 or not job.accepted_parent_retired:
            failures.append('mandatory original Future/retirement incomplete')
    if owner.bridge.single_file_blocked_attempts < 1:
        failures.append('scenario did not exercise active deferral')
    if owner.accepted_ns is not None:
        earliest = (t0 + 100000000 if args.scenario == 'original_deadline_100ms' else producer_data.get('event_actual_ns'))
        if earliest is not None and owner.accepted_ns < earliest:
            failures.append('submitted before external progress/deadline')
    missing = [n for n in ('_pump_once', '_schedule_work', '_drain_cuda_copies', '_poll_ring_completions',
                          '_flush_copy_batch', '_finish_jobs', '_intake', '_drain_incoming',
                          '_drain_ready_preload_fds', '_drain_ready_load_fds') if owner.counts[n] <= 0]
    if missing:
        failures.append('required native stages not called: ' + repr(missing))
    # Gather module source bytes before interpreter exit, outside measured threads.
    modules = {}
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and (name.startswith('prefix_io_control') or name in ('_cpu_only_candidate_collector', '_full_pump_support', '_retry_cpu_policy_support', '_full_pump_identity')):
            path = Path(file)
            modules[name] = {'path': str(path.resolve()), 'sha256': sha(path.read_bytes())}
    identity['loaded_modules'] = modules
    forbidden = sorted(n for n in sys.modules if n == 'torch' or n == 'vllm' or n.startswith(('torch.', 'vllm.')))
    if forbidden:
        failures.append('forbidden GPU runtime imported')
    put = owner._incoming.put_records
    get = owner._incoming.get_records
    primary_kind = ('_MandatoryWait' if args.scenario == 'mandatory_20ms' else
                    'STOP' if args.scenario == 'stop_20ms' else None)
    enqueue = next((e['at_ns'] for e in put if e['kind'] == primary_kind), None)
    intake = next((e['at_ns'] for e in get if e['kind'] == primary_kind), None)
    progress_signed = None if enqueue is None or owner.accepted_ns is None else owner.accepted_ns - enqueue
    already_progressing = progress_signed is not None and progress_signed < 0
    stat0, stat1 = before.get('stat'), after.get('stat')
    throttle = None if stat0 is None or stat1 is None else stat1.get('nr_throttled', 0) - stat0.get('nr_throttled', 0)
    result = {'schema_version': 1, 'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
        'scope': 'CPU_SYNTHETIC_FULL_NATIVE_CONTROL_FLOW_NOT_GPU_IO', 'arm': args.arm, 'scenario': args.scenario,
        'source_identity': identity, 'cuda_initialized': False, 'GPU_operations': 0, 'forbidden_imports': forbidden,
        'worker_cpu_ns': worker_data.get('cpu_ns'), 'producer_cpu_ns': producer_data.get('cpu_ns'),
        'worker_plus_producer_cpu_ns': worker_data.get('cpu_ns', 0) + producer_data.get('cpu_ns', 0),
        'wall_ns': end_ns - t0, 'worker': worker_data, 'producer': producer_data,
        'ready_arrived_ns': t0, 'ready_arrival_unchanged': owner.ready.open_start_ns == t0,
        'accepted_ns': owner.accepted_ns, 'completed_ns': owner.completed_ns,
        'enqueue_to_intake_ns': None if enqueue is None or intake is None else intake - enqueue,
        'enqueue_to_progress_signed_ns': progress_signed,
        'already_in_progress_at_control_enqueue': already_progressing,
        'enqueue_to_progress_eligible': progress_signed is not None and not already_progressing,
        'enqueue_to_progress_ns': None if already_progressing else progress_signed,
        'STOP_to_join_ns': None if not any(x['kind'] == 'STOP' for x in put) else worker_data.get('wall_end_ns', end_ns) - next(x['at_ns'] for x in put if x['kind'] == 'STOP'),
        'deadline_lateness_ns': None if args.scenario != 'original_deadline_100ms' or owner.accepted_ns is None else owner.accepted_ns - (t0 + 100000000),
        'counts': dict(owner.counts), 'preview_reasons': dict(owner.policy_preview_reasons),
        'bridge': {'preview': owner.bridge.single_file_previews, 'defer': owner.bridge.single_file_blocked_attempts,
                   'observation_reuse': owner.bridge.single_file_observation_reuses},
        'ring': {'poll': owner.ring.poll_calls, 'submit_pending': owner.ring.submit_calls, 'simulated_completed': owner.ring.completed},
        'pool': {'reserves': owner.staging_pool.reserves, 'releases': owner.staging_pool.releases, 'free': owner.staging_pool.free_count},
        'queue': {'put': put, 'get': get, 'blocking_get_calls': owner._incoming.get_block_count},
        'payload_sha256': owner.completed_payload_sha256, 'cgroup_before': before, 'cgroup_after': after,
        'nr_throttled_delta': throttle, 'physical_GPU_release_credit': False,
        'native_final_state': {'active': len(owner._active), 'inflight': len(owner._inflight),
            'pending_copies': len(owner._pending_copies), 'copy_ready': len(owner._copy_ready),
            'ready_load': len(owner._ready_fds_load), 'ready_preload': len(owner._ready_fds_preload),
            'queue_size': owner._incoming.qsize(), 'accepted_parents': owner._accepted_parent_count,
            'parent_retired_count': owner._parent_retired_count, 'original_STOP_observed': owner._stop,
            'worker_alive': owner._worker.is_alive(), 'producer_alive': prod.is_alive(),
            'mandatory_future_done': None if owner.mandatory_job is None else owner.mandatory_job.future.done(),
            'mandatory_future_result': None if owner.mandatory_job is None or not owner.mandatory_job.future.done()
                                        or owner.mandatory_job.future.exception() else owner.mandatory_job.future.result()}}
    write_json(Path(args.output), result)
    print(json.dumps({'status': result['status'], 'arm': args.arm, 'scenario': args.scenario, 'failures': failures}))
    return 0 if not failures else 1


def suite(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(args.protocol).resolve()
    protocol_bytes = protocol_path.read_bytes()
    assert sha(protocol_bytes) == PROTOCOL_SHA256, 'fixed preregistration bytes changed'
    with (out / 'PROTOCOL.json').open('xb') as f:
        f.write(protocol_bytes)
    protocol = json.loads(protocol_bytes)
    assert protocol['original_max_wait_ns'] == protocol['original_sample_max_age_ns'] == 100000000
    candidates = {'A': Path(args.baseline_root).resolve(), 'B': Path(args.candidate_root).resolve()}
    manifest = {arm: {'root': str(p), 'reactor': sha((p / REL).read_bytes()),
                      'collector': sha((p / 'native_full_step_collector.py').read_bytes())} for arm, p in candidates.items()}
    plan = {'schema_version': 1, 'mode': args.mode, 'protocol_path': str(protocol_path),
            'protocol_sha256': sha(protocol_bytes), 'protocol': protocol, 'candidate_manifest': manifest,
            'script_sha256': sha(Path(__file__).read_bytes()), 'python': sys.version, 'argv': sys.argv,
            'GPU_runs': 0, 'scored': args.mode == 'score', 'premeasurement_settle_ns': 200000000,
            'synthetic_backend_completion_delay_ns': 1000000}
    write_json(out / 'PLAN.json', plan)
    fixtures = [protocol['main']] + protocol['sensitivity'] + protocol['functional_scenarios']
    all_results = []
    for scenario in fixtures:
        if args.mode == 'smoke':
            schedule = [('smoke', 0, arm) for arm in ('A', 'B')]
        else:
            schedule = [('warmup', i, arm) for i, arm in enumerate(protocol['warmup_order_per_scenario'])]
            schedule += [('score', pair, arm) for pair in range(scenario['pairs']) for arm in protocol['pair_order'][pair]]
        for phase, number, arm in schedule:
            label = '%s-%s-%02d-%s' % (scenario['name'], phase, number, arm)
            target = out / (label + '.json')
            command = [sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve()), '--trial',
                '--candidate-root', str(candidates[arm]), '--arm', arm, '--scenario', scenario['name'], '--output', str(target)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            if args.author_source_root:
                env['SERVER11_AUTHOR_SOURCE_ROOT'] = str(Path(args.author_source_root).resolve())
            begin = time.monotonic_ns()
            try:
                completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired as exc:
                def decoded(value):
                    return value.decode('utf-8', 'replace') if isinstance(value, bytes) else (value or '')
                completed = NS(returncode=124, stdout=decoded(exc.stdout), stderr=decoded(exc.stderr) + '\nouter 15-second fixture watchdog')
            record = {'label': label, 'phase': phase, 'pair_or_warmup': number, 'arm': arm,
                      'scenario': scenario['name'], 'command': command, 'exit': completed.returncode,
                      'wall_ns_including_setup': time.monotonic_ns() - begin,
                      'stdout': completed.stdout, 'stderr': completed.stderr,
                      'result_file': target.name, 'result_sha256': sha(target.read_bytes()) if target.exists() else None}
            write_json(out / (label + '-command.json'), record)
            all_results.append(record)
            if completed.returncode:
                write_json(out / 'SUITE.json', {'status': 'STOPPED_ON_FAILURE', 'mode': args.mode, 'records': all_results})
                print(json.dumps({'status': 'STOPPED_ON_FAILURE', 'label': label, 'stderr': completed.stderr[-3000:], 'stdout': completed.stdout[-1500:]}))
                return 1
    for arm, candidate in candidates.items():
        assert sha((candidate / REL).read_bytes()) == manifest[arm]['reactor']
        assert sha((candidate / 'native_full_step_collector.py').read_bytes()) == manifest[arm]['collector']
    write_json(out / 'SUITE.json', {'status': 'COMPLETE', 'mode': args.mode, 'records': all_results})
    print(json.dumps({'status': 'COMPLETE', 'mode': args.mode, 'trials': len(all_results), 'output': str(out)}))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial', action='store_true')
    p.add_argument('--mode', choices=['smoke', 'score'], default='smoke')
    p.add_argument('--baseline-root')
    p.add_argument('--candidate-root', required=True)
    p.add_argument('--author-source-root')
    p.add_argument('--protocol')
    p.add_argument('--output-dir')
    p.add_argument('--arm', choices=['A', 'B'])
    p.add_argument('--scenario', choices=SCENARIOS)
    p.add_argument('--output')
    args = p.parse_args()
    return run_trial(args) if args.trial else suite(args)


if __name__ == '__main__':
    raise SystemExit(main())
