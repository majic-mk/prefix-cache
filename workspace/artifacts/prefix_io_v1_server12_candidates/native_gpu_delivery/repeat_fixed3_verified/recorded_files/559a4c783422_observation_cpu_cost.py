"""Bounded gross CPU evidence over the complete existing observation lifecycle.

The span also contains original model execution and native I/O. It is not an
isolated observer cost estimator or a new upper bound. Thread values are
diagnostics inside process CPU, never additional costs to sum. No GPU imports,
profilers, model hooks, native hooks, queues or scheduling changes are installed.
"""
from __future__ import annotations
import os
from pathlib import Path
from math import ceil
import threading
import time
import weakref

SCHEMA = 'c5_normal_complete_path_gross_cpu_v1'
SCOPE = 'collector_install_through_capture_detach_original_drain_notification_export'
MODES = ('off', 'shadow', 'on')


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name)
    return value


def reference(row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
            and type(row['path']) is str and row['path'] and '\\' not in row['path']
            and ':' not in row['path'] and not row['path'].startswith('/')
            and all(part not in ('', '.', '..') for part in row['path'].split('/'))
            and type(row['bytes']) is int and row['bytes'] > 0
            and type(row['sha256']) is str and len(row['sha256']) == 64
            and all(part in '0123456789abcdef' for part in row['sha256']),
            'actual project-relative source reference')
    return dict(row)


def native_task_cpu(native_tid):
    """Read one fixed Linux TID safely; terminated tasks become unavailable.

    Foreign pthread CPU APIs can have undefined behavior after thread exit.
    /proc provides bounded counters plus task start time for TID reuse checks.
    """
    integer(native_tid, 'actual Linux native TID', 1)
    raw = (Path('/proc/self/task') / str(native_tid) / 'stat').read_text(encoding='utf-8')
    require(int(raw.split(' ', 1)[0]) == native_tid, 'actual task stat TID')
    fields = raw[raw.rfind(')') + 2:].split()
    require(len(fields) >= 20, 'complete task stat CPU fields')
    ticks = integer(int(fields[11]), 'task user ticks') + integer(int(fields[12]), 'task system ticks')
    start = integer(int(fields[19]), 'task starttime identity')
    hz = integer(os.sysconf('SC_CLK_TCK'), 'actual kernel CPU clock ticks', 1)
    return dict(native_tid=native_tid, start_ticks=start, cpu_ns=ticks * 1_000_000_000 // hz,
                resolution_ns=(1_000_000_000 + hz - 1) // hz, clock_source='linux_proc_task_stat')


class CompletePathCPU:
    """Observe one finite span, borrowing the three original threads weakly."""
    def __init__(self, *, mode, run_id, request_id, source_refs, origin='native_gpu_recording',
                 wall_clock=time.monotonic_ns, process_clock=time.process_time_ns,
                 thread_clock=time.thread_time_ns, native_reader=None):
        require(mode in MODES and origin in ('native_gpu_recording', 'cpu_fixture'), 'explicit finite origin/mode')
        require(type(run_id) is str and type(request_id) is str and run_id and request_id and run_id != request_id,
                'distinct actual run/request identities')
        require(type(source_refs) is dict and set(source_refs) == {'runner', 'cost_meter', 'collector', 'reactor', 'notification_adapter'},
                'actual complete-path sources')
        if origin == 'native_gpu_recording':
            require(wall_clock is time.monotonic_ns and process_clock is time.process_time_ns
                    and thread_clock is time.thread_time_ns and native_reader is None,
                    'native cost uses actual builtin clocks')
        self.mode, self.run_id, self.request_id, self.origin = mode, run_id, request_id, origin
        self.sources = {key: reference(value) for key, value in source_refs.items()}
        self.wall_clock, self.process_clock, self.thread_clock = wall_clock, process_clock, thread_clock
        self.native_reader = native_task_cpu if native_reader is None else native_reader
        self.started = self.stopped = False
        self._threads = []
        self.failures = []

    def start(self, reactor):
        require(not self.started, 'one CPU meter start')
        self.wall_start_ns = integer(self.wall_clock(), 'wall start', 1)
        self.process_start_ns = integer(self.process_clock(), 'process CPU start')
        original_threads = [('main', threading.current_thread()), ('native_owner', getattr(reactor, '_worker', None)),
                            ('native_aio', getattr(getattr(reactor, 'ring', None), '_worker', None))]
        for role, thread in original_threads:
            row = dict(role=role, python_ident=None, native_tid=None, before_cpu_ns=None, after_cpu_ns=None,
                       cpu_ns=None, status='UNAVAILABLE', reason=None, clock_source=None,
                       resolution_ns=None, task_start_ticks=None)
            slot = dict(row=row, thread=None, main=(role == 'main'))
            try:
                require(thread is not None and thread.is_alive(), 'original thread not live')
                ident, tid = thread.ident, getattr(thread, 'native_id', None)
                integer(ident, 'actual Python thread identity', 1); integer(tid, 'actual native TID', 1)
                row.update(python_ident=ident, native_tid=tid)
                slot['thread'] = weakref.ref(thread)
                if role == 'main':
                    value = self.thread_clock()
                    row.update(clock_source='builtin_thread_time_ns' if self.origin == 'native_gpu_recording' else 'explicit_cpu_fixture_thread_clock',
                               resolution_ns=max(1, ceil(time.get_clock_info('thread_time').resolution * 1_000_000_000)))
                else:
                    task = self.native_reader(tid)
                    require(task['native_tid'] == tid, 'actual fixed native TID snapshot')
                    row.update(clock_source=task['clock_source'], resolution_ns=integer(task['resolution_ns'], 'task clock resolution', 1),
                               task_start_ticks=integer(task['start_ticks'], 'task lifetime identity'))
                    value = task['cpu_ns']
                row.update(before_cpu_ns=integer(value, 'actual thread CPU start'), status='STARTED')
            except Exception as exc:
                row['reason'] = type(exc).__name__ + ': ' + str(exc)[:120]
            self._threads.append(slot)
        self.started = True

    def stop(self, *, full_frame_count, action_count, notification_evidence):
        require(self.started and not self.stopped, 'CPU meter stops once after actual lifecycle')
        for slot in self._threads:
            row = slot['row']
            if row['status'] != 'STARTED':
                continue
            try:
                thread = slot['thread']()
                require(thread is not None and thread.is_alive() and thread.ident == row['python_ident']
                        and getattr(thread, 'native_id', None) == row['native_tid'], 'thread ended or identity reused')
                if slot['main']:
                    value = self.thread_clock()
                else:
                    task = self.native_reader(row['native_tid'])
                    require(task['native_tid'] == row['native_tid'] and task['start_ticks'] == row['task_start_ticks']
                            and task['clock_source'] == row['clock_source'] and task['resolution_ns'] == row['resolution_ns'],
                            'native task vanished or TID lifetime reused')
                    value = task['cpu_ns']
                integer(value, 'actual thread CPU end')
                require(value >= row['before_cpu_ns'], 'thread CPU counter moved backward')
                row.update(after_cpu_ns=value, cpu_ns=value-row['before_cpu_ns'], status='RECORDED')
            except Exception as exc:
                row.update(status='UNAVAILABLE_AT_END', cpu_ns=None,
                           reason=type(exc).__name__ + ': ' + str(exc)[:120])
        # Enclose every diagnostic read inside the primary process/wall span.
        self.process_end_ns = integer(self.process_clock(), 'process CPU end')
        self.wall_end_ns = integer(self.wall_clock(), 'wall end', 1)
        require(self.process_end_ns >= self.process_start_ns and self.wall_end_ns >= self.wall_start_ns,
                'monotonic actual process/wall CPU span')
        self.full_frame_count = integer(full_frame_count, 'actual full frame count')
        self.action_count = integer(action_count, 'actual action count')
        require(type(notification_evidence) is dict and notification_evidence.get('mode') == self.mode
                and notification_evidence.get('run_id') == self.run_id
                and notification_evidence.get('request_id') == self.request_id,
                'actual same-mode notification evidence')
        rows = notification_evidence.get('rows')
        require(type(rows) is list and len(rows) <= 16, 'bounded actual notification witnesses')
        self.wait_exercised = any(row.get('outcome') in ('matching_wake', 'original_deadline_timeout') for row in rows)
        self.wait_status = ('UNINSTALLED_CONTROL_PATH' if self.mode != 'on' else
                            'EXERCISED' if self.wait_exercised else 'NOT_EXERCISED')
        self.notification_rows = len(rows)
        self.stopped = True

    def export(self):
        require(self.started and self.stopped, 'CPU span completed before export')
        rows = [dict(slot['row']) for slot in self._threads]
        identifiers = [row['native_tid'] for row in rows if row['native_tid'] is not None]
        return dict(schema=SCHEMA, status='GROSS_COMPLETE_PATH_CPU_RECORDED', origin=self.origin,
            mode=self.mode, run_id=self.run_id, request_id=self.request_id, measurement_scope=SCOPE,
            source_refs=self.sources, wall_start_ns=self.wall_start_ns, wall_end_ns=self.wall_end_ns,
            process_cpu_start_ns=self.process_start_ns, process_cpu_end_ns=self.process_end_ns,
            wall_ns=self.wall_end_ns-self.wall_start_ns,
            process_cpu_ns=self.process_end_ns-self.process_start_ns,
            thread_cpu_diagnostics=rows, aliased_native_tids=len(set(identifiers)) != len(identifiers),
            thread_diagnostics_included_in_process_cpu=True, thread_values_are_not_additive=True,
            full_frame_count=self.full_frame_count, action_count=self.action_count,
            notification_rows=self.notification_rows, notification_wait_status=self.wait_status,
            measurement_includes_original_model_and_native_io=True,
            isolated_observation_cost_measured=False, net_observation_cost_qualified=False,
            full_runtime_cost_qualified=False, on_observation_cost_qualified=False,
            performance_effect_verified=False, valid_native_cost_upper_ns=None,
            valid_native_step_budget_ns=None)


def validate_gross_cost(document, *, mode, run_id, request_id, source_refs,
                        full_frame_count, action_count, notification_evidence):
    require(type(document) is dict and document.get('schema') == SCHEMA
            and document.get('status') == 'GROSS_COMPLETE_PATH_CPU_RECORDED'
            and document.get('origin') == 'native_gpu_recording'
            and (document.get('mode'), document.get('run_id'), document.get('request_id')) == (mode, run_id, request_id)
            and document.get('measurement_scope') == SCOPE and document.get('source_refs') == source_refs,
            'actual complete gross CPU origin/identity/source')
    start = integer(document.get('wall_start_ns'), 'gross wall start', 1)
    end = integer(document.get('wall_end_ns'), 'gross wall end', start)
    before = integer(document.get('process_cpu_start_ns'), 'gross process CPU start')
    after = integer(document.get('process_cpu_end_ns'), 'gross process CPU end', before)
    require(document.get('wall_ns') == end-start and type(document.get('wall_ns')) is int
            and document.get('process_cpu_ns') == after-before and type(document.get('process_cpu_ns')) is int,
            'gross clock subtraction')
    require(type(document.get('full_frame_count')) is int and document['full_frame_count'] == full_frame_count
            and type(document.get('action_count')) is int and document['action_count'] == action_count
            and document.get('thread_values_are_not_additive') is True
            and document.get('thread_diagnostics_included_in_process_cpu') is True
            and document.get('measurement_includes_original_model_and_native_io') is True,
            'actual path coverage and overlapping CPU scope')
    for key in ('isolated_observation_cost_measured', 'net_observation_cost_qualified',
                'full_runtime_cost_qualified', 'on_observation_cost_qualified', 'performance_effect_verified'):
        require(document.get(key) is False, 'gross CPU evidence cannot grant ' + key)
    require(document.get('valid_native_cost_upper_ns') is None and document.get('valid_native_step_budget_ns') is None,
            'meter cannot refit native calibration or budget')
    rows = document.get('thread_cpu_diagnostics')
    require(type(rows) is list and [row.get('role') for row in rows] == ['main', 'native_owner', 'native_aio'],
            'bounded three original thread diagnostics')
    for row in rows:
        require(row.get('status') in ('RECORDED', 'UNAVAILABLE', 'UNAVAILABLE_AT_END'), 'actual thread clock status')
        if row['status'] == 'RECORDED':
            integer(row.get('python_ident'), 'actual diagnostic pthread', 1)
            integer(row.get('native_tid'), 'actual diagnostic TID', 1)
            first = integer(row.get('before_cpu_ns'), 'thread CPU before')
            last = integer(row.get('after_cpu_ns'), 'thread CPU after', first)
            require(type(row.get('cpu_ns')) is int and row['cpu_ns'] == last-first, 'actual thread delta')
            integer(row.get('resolution_ns'), 'actual diagnostic clock resolution', 1)
            if row['role'] == 'main':
                require(row.get('clock_source') == 'builtin_thread_time_ns' and row.get('task_start_ticks') is None,
                        'actual primary thread CPU clock')
            else:
                require(row.get('clock_source') == 'linux_proc_task_stat', 'actual safe fixed native TID clock')
                integer(row.get('task_start_ticks'), 'actual diagnostic task lifetime')
        else:
            require(row.get('cpu_ns') is None and type(row.get('reason')) is str and row['reason'],
                    'unavailable/reused thread is unknown, not zero')
    notification_rows = notification_evidence['rows']
    require(type(notification_rows) is list and len(notification_rows) <= 16
            and notification_evidence.get('mode') == mode
            and notification_evidence.get('run_id') == run_id
            and notification_evidence.get('request_id') == request_id,
            'bounded same-path native notification cost witness')
    exercised = any(row.get('outcome') in ('matching_wake', 'original_deadline_timeout') for row in notification_rows)
    status = 'UNINSTALLED_CONTROL_PATH' if mode != 'on' else 'EXERCISED' if exercised else 'NOT_EXERCISED'
    require(type(document.get('notification_rows')) is int
            and document['notification_rows'] == len(notification_rows)
            and document.get('notification_wait_status') == status, 'actual wait cost path exercised flag')
    ids = [row.get('native_tid') for row in rows if row.get('native_tid') is not None]
    require(document.get('aliased_native_tids') is (len(set(ids)) != len(ids)),
            'thread aliases cannot be presented as independent additional cost')
    return dict(status='VALIDATED_NATIVE_GROSS_CPU_ONLY', process_cpu_ns=after-before,
                wall_ns=end-start, notification_wait_status=status,
                full_runtime_cost_qualified=False, isolated_observation_cost_measured=False,
                net_observation_cost_qualified=False, performance_effect_verified=False)


def descriptive_net(current, baseline):
    require(current.get('measurement_scope') == baseline.get('measurement_scope') == SCOPE
            and current.get('source_refs') == baseline.get('source_refs'), 'same gross complete CPU source/scope')
    current_ns = integer(current.get('process_cpu_ns'), 'current actual process CPU')
    baseline_ns = integer(baseline.get('process_cpu_ns'), 'baseline actual process CPU')
    return dict(scope='matched_output_and_work_whole_path_descriptive_net_cpu',
                current_mode=current['mode'], baseline_mode=baseline['mode'],
                process_cpu_difference_ns=current_ns-baseline_ns,
                measurement_includes_original_model_and_native_io=True,
                isolated_observation_cost_measured=False, net_observation_cost_qualified=False,
                full_runtime_cost_qualified=False, performance_effect_verified=False,
                sample_pairs=1)
