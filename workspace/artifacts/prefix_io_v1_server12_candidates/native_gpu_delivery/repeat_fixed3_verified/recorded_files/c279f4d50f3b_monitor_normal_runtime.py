"""One read-only compact snapshot of the original normal guard and raw result.

No framework imports, subprocesses, signals, polling loop or file writes. This
monitor never validates native qualification; the frozen verifier owns that.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

PROJECT = '/root/autodl-tmp/prefix-io-v1-handoff/project'
ENTRY = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
GUARD_SOURCE = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
GUARD_SHA = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'


def safe(root, relative):
    if ':' in relative or '\\' in relative or relative.startswith('/') or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError('bounded project-relative path required')
    path = root
    for part in relative.split('/'):
        path /= part
        if path.is_symlink():
            raise ValueError('symlink monitor path refused')
    if not path.resolve().is_relative_to(root):
        raise ValueError('monitor path outside project')
    return path


def clean(value, limit=160):
    if type(value) is str:
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)
        text = re.sub(r'(?i)(password|passwd|api[_-]?key|secret|access[_-]?token|authorization)\s*[:=].*', r'\1=[redacted]', text)
        return text[-limit:]
    return value if type(value) in (int, float, bool) or value is None else '<non-scalar>'


def scalars(document, names):
    return {name: clean(document.get(name)) for name in names}


def count(value):
    return len(value) if type(value) in (list, dict) else None


def read(root, relative):
    path = safe(root, relative)
    if not path.exists():
        return None, dict(state='ABSENT', path=relative)
    before = path.stat()
    if not path.is_file() or before.st_size > 32 * 1024**2:
        return None, dict(state='NOT_BOUNDED_REGULAR_FILE', path=relative)
    raw = path.read_bytes()
    after = path.stat()
    metadata = dict(path=relative, bytes=len(raw), stable_read=(before.st_size, before.st_mtime_ns) ==
        (after.st_size, after.st_mtime_ns), sha256=hashlib.sha256(raw).hexdigest())
    try:
        def pairs(items):
            out = {}
            for key, value in items:
                if key in out:
                    raise ValueError('duplicate key')
                out[key] = value
            return out
        document = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        if type(document) is not dict:
            raise ValueError('non-object JSON')
        return document, dict(metadata, state='READABLE_JSON')
    except (ValueError, UnicodeDecodeError) as exc:
        return None, dict(metadata, state='PARTIAL_OR_INVALID_JSON', reason=clean(str(exc)))


def process(pid):
    result = dict(pid=pid, observed=False)
    if type(pid) is not int or pid <= 0 or os.name != 'posix':
        return result
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        result.update(observed=True, state=fields[0], parent_pid=int(fields[1]),
            process_group=int(fields[2]), session_id=int(fields[3]))
    except (OSError, ValueError, IndexError):
        result['state'] = 'NO_LONGER_OBSERVABLE'
    return result


def tail(root, relative, limit):
    path = safe(root, relative)
    if not path.is_file():
        return dict(path=relative, state='ABSENT', lines=[])
    size = path.stat().st_size
    with path.open('rb') as stream:
        stream.seek(max(0, size - 16384))
        raw = stream.read(16384)
    lines = raw.decode('utf-8', errors='replace').splitlines()[-48:]
    return dict(path=relative, state='READ_ONLY_TAIL', bytes=size,
        tail_lines_examined=len(lines), lines=[clean(line, 180) for line in (lines[-limit:] if limit else [])])


def snapshot(root, mode, tail_lines):
    if root.as_posix() != PROJECT:
        raise ValueError('actual fixed project required')
    label = 'server12-c5-native-normal-' + mode + '01'
    run = 'experiments/prefix_io_v1/runs/' + label
    budget, budget_ref = read(root, LEDGER)
    guard, guard_ref = read(root, run + '/result.json')
    raw, raw_ref = read(root, run + '/details/p4-single-file-runtime-result.json')
    qualification, qualification_ref = read(root, ENTRY + '/QUALIFICATION_' + mode + '.json')
    active = budget.get('active_reservation') if budget is not None else None
    active_summary = None
    if type(active) is dict:
        active_summary = scalars(active, ('id', 'label', 'gpu_uuid', 'seconds_limit', 'reserved_seconds',
            'started_unix', 'session_id', 'process_group', 'runner_pid', 'state', 'accounted_seconds'))
        active_summary['same_mode'] = active.get('label') == label
        active_summary['runner_process'] = process(active.get('runner_pid'))
        active_summary['child_session_process'] = process(active.get('session_id'))
        started = active.get('started_unix')
        if type(started) in (int, float):
            active_summary['observed_wall_age_seconds_not_budget_charge'] = round(max(0, time.time()-started), 3)
    used = budget.get('gpu_wall_seconds') if budget is not None else None
    budget_summary = dict(source=budget_ref, gpu_wall_seconds=clean(used),
        original_limit_seconds=28800,
        remaining_seconds=(28800-used if type(used) in (int,float) else None),
        events_count=count(budget.get('events')) if budget is not None else None,
        active=active_summary)
    guard_summary = dict(source=guard_ref)
    if guard is not None:
        guard_summary.update(scalars(guard, ('label', 'gpu_uuid', 'reservation_id', 'session_id',
            'elapsed_seconds', 'exit', 'child_exit', 'timed_out', 'interrupted_signal', 'error',
            'gpu_job_attempted', 'session_drained')))
        guard_summary['session_members_before_cleanup_count'] = count(guard.get('session_members_before_cleanup'))
        guard_summary['session_members_after_cleanup_count'] = count(guard.get('session_members_after_cleanup'))
    raw_summary = dict(source=raw_ref, raw_exists_only_after_runner_finally=True)
    if raw is not None:
        raw_summary.update(scalars(raw, ('status', 'mode', 'label', 'gpu_uuid', 'original_engine_shutdown_returned',
            'fresh_original_process', 'subprocess_pid', 'subprocess_sid', 'cache_reset_count',
            'baseline_policy_changed', 'error_type', 'error_message')))
        windows = raw.get('windows')
        raw_summary['window_count'] = count(windows)
        raw_summary['warmup_count'] = count(raw.get('warmups'))
        journal = raw.get('native_journal')
        raw_summary['journal'] = dict(valid=journal.get('valid'), event_count=count(journal.get('events')),
            frame_count=count(journal.get('frames')), lost=journal.get('lost')) if type(journal) is dict else None
        raw_summary['native_tail'] = scalars(raw.get('native_tail_assertions', {}),
            ('worker_alive', 'aio_worker_alive', 'handler_shutdown', 'reactor_closed'))
        selected = []
        for row in windows[:2] if type(windows) is list else []:
            capture = row.get('capture', {})
            output = row.get('frontend', {}).get('output', {})
            notification = row.get('notification_evidence', {})
            meter = row.get('observation_cpu_cost', {})
            selected.append(dict(condition=clean(row.get('condition')), output_count=count(output.get('output_token_ids')),
                cached_tokens=clean(output.get('num_cached_tokens')), capture_valid=clean(capture.get('valid')),
                frame_count=count(capture.get('frames')), witness_count=count(capture.get('event_witnesses')),
                pending_event_pairs=clean(capture.get('pending_event_pairs')), open_event_pair=clean(capture.get('open_event_pair')),
                capture_failure_count=count(capture.get('failures')), action_count=count(capture.get('actions')),
                notification=scalars(notification, ('mode', 'installed', 'capture_valid', 'capture_wait_failed',
                    'reactor_wait_faulted', 'reactor_arm_clear')),
                gross_cpu=scalars(meter, ('status', 'wall_ns', 'process_cpu_ns'))))
        raw_summary['windows'] = selected
    qual_summary = dict(source=qualification_ref)
    if qualification is not None:
        qual_summary.update(scalars(qualification, ('mode', 'native_execution_verified', 'runtime_condition_qualified',
            'qualification_passed', 'permits_next_mode', 'notification_wait_status', 'selected_gpu_elapsed_ns',
            'frozen_cost_upper_ns', 'frozen_a_only_budget_ns', 'full_output_tokens', 'full_frames',
            'strategy_effect_verified')))
    guard_path = safe(root, GUARD_SOURCE)
    guard_pin = dict(path=GUARD_SOURCE, observed_sha256=hashlib.sha256(guard_path.read_bytes()).hexdigest()) if guard_path.is_file() else None
    if guard_pin is not None:
        guard_pin['matches_original_guard_source'] = guard_pin['observed_sha256'] == GUARD_SHA
    result = dict(status='READ_ONLY_NORMAL_MONITOR_SNAPSHOT', mode=mode, label=label,
        budget=budget_summary, guard=guard_summary, raw=raw_summary, qualification=qual_summary,
        original_guard_source=guard_pin, child_output_log=tail(root, run + '/process.log', tail_lines),
        qualification_performed_by_monitor=False, GPU_operations_this_action=0, writes=0)
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(text) > 5000:
        result['child_output_log']['lines'] = result['child_output_log']['lines'][-3:]
        result['child_output_log']['display_limited_for_5000_character_budget'] = True
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(text) > 5000:
        result['child_output_log']['lines'] = []
        result['raw'].pop('windows', None)
        result['compact_display_omitted_details'] = True
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(text) > 5000:
        raise ValueError('compact snapshot exceeded display budget')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('off', 'shadow', 'on'))
    parser.add_argument('--tail-lines', type=int, default=8, choices=range(0,49))
    args = parser.parse_args(argv)
    try:
        print(snapshot(args.project.resolve(strict=True), args.mode, args.tail_lines))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='READ_ONLY_MONITOR_UNAVAILABLE', error_type=type(exc).__name__,
            reason=clean(str(exc), 240), qualification_performed_by_monitor=False,
            GPU_operations_this_action=0, writes=0), ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
