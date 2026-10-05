"""Append an actual formal peer's parent only after original guard/raw/OS closure.

This CPU command never starts a model, GPU guard, cache directory or backend.
The parent itself is checked in memory before its first write and is added to a
later source lock by the caller. There is no draft, self-reference or new GPU
authority. The unchanged original budget and its existing lock are read only.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

CLOSURE_SCHEMA = 'guard_closed_formal_native_peer_parent_v1'
ROLES = {('development', 'off', 'U'): 'development',
         ('effect', 'off', 'U'): 'evaluation', ('effect', 'on', 'I'): 'evaluation'}


def require(value, reason):
    if not value:
        raise ValueError('FORMAL_PEER_CLOSE_REJECTED: ' + reason)


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and not relative.startswith('/') and
            ':' not in relative and '\\' not in relative and '\0' not in relative and
            all(part not in ('', '.', '..') for part in relative.split('/')), 'project-relative POSIX path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink source/output')
    require(path.resolve().is_relative_to(root), 'project containment')
    return path


def actual_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual regular file required: ' + relative)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    require(path.stat().st_size == size, 'file changed while hashing')
    return dict(path=relative, bytes=size, sha256=digest.hexdigest())


def bounded_json(path, limit=32 * 1024**2):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded actual JSON')
    return parse(path.read_bytes())


def cpu_modules_only():
    require(not any(name == 'torch' or name.startswith(('torch.', 'vllm', 'py_kvcache'))
                    for name in sys.modules), 'CPU closure never imports framework/model/native backend')


def proc_session_fields(raw):
    """Read the actual stat session after the final command-name delimiter."""
    require(type(raw) is str, 'actual proc stat text')
    cut = raw.rfind(')')
    fields = raw[cut + 1:].split()
    require(cut >= 0 and len(fields) >= 4 and fields[3].isdecimal(), 'actual proc session stat shape')
    return fields[0], int(fields[3])


def session_members(session_id, *, proc_root=Path('/proc')):
    """Read Linux process session scalars only. No kill, polling or CUDA query."""
    require(os.name == 'posix' and type(session_id) is int and session_id > 0 and proc_root.is_dir(),
            'actual Linux original guard session required')
    rows = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / 'stat').read_text(encoding='utf-8')
        except (FileNotFoundError, ProcessLookupError):
            continue
        state, actual_session = proc_session_fields(raw)
        if actual_session == session_id:
            rows.append(dict(pid=int(entry.name), state=state, session_id=session_id))
    return sorted(rows, key=lambda row: row['pid'])


@contextmanager
def original_budget_read_lock(root, *, driver):
    """Use the existing guard's lock read only; fail instead of blocking a run."""
    require(os.name == 'posix', 'actual server Linux budget-lock check')
    import fcntl
    ledger = driver.safe(root, driver.LEDGER)
    lock = ledger.with_suffix('.lock')
    require(lock.is_file() and not lock.is_symlink(), 'existing original guard budget lock required; no lock creation')
    with lock.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield ledger
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _closed_leaf(root, row, refs, *, driver):
    require(type(row) is dict and refs.get(row.get('path')) == row, 'actual input inside current frozen source closure')
    return driver.read(driver.check_ref(root, row))


def produce_parent(root, *, source_lock_ref, child_ref, guard_ref, config_ref,
                   output_relative, driver, peer_checker,
                   session_probe=session_members, budget_lock=original_budget_read_lock):
    """Close real evidence once. Tests inject CPU-only mocks, never authority."""
    cpu_modules_only()
    with budget_lock(root, driver=driver) as ledger_path:
        before = ledger_path.read_bytes()
        ledger = parse(before)
        require(type(ledger) is dict and 'active_reservation' in ledger and ledger['active_reservation'] is None,
                'original GPU budget must be idle before parent closure')
        used = ledger.get('gpu_wall_seconds')
        require(type(used) in (int, float) and math.isfinite(used) and 0 <= used <= driver.MAX_GPU_SECONDS,
                'actual unchanged cumulative GPU budget')
        refs = driver.source_rows(root, source_lock_ref, full=True)
        child = _closed_leaf(root, child_ref, refs, driver=driver)
        guard = _closed_leaf(root, guard_ref, refs, driver=driver)
        config = _closed_leaf(root, config_ref, refs, driver=driver)
        require(type(child) is type(guard) is type(config) is dict, 'actual child/guard/config JSON objects')
        role = config.get('phase'), config.get('mode'), config.get('arm')
        require(role in ROLES and role == (child.get('phase'), child.get('mode'), child.get('arm')),
                'actual bounded development Uoff or effect Uoff/Ion peer role')
        require(child.get('os_session_drained') is False and
                all(name not in child for name in ('completed_guard_ref', 'child_result_ref', 'formal_parent_closure_schema')),
                'actual unclosed raw child; no already-closed parent or closure fields')
        require(child.get('actual_run_config_ref') == config_ref, 'actual raw child used this frozen config')
        require(type(guard.get('exit')) is int and guard['exit'] == 0 and
                type(guard.get('child_exit')) is int and guard['child_exit'] == 0 and
                guard.get('gpu_job_attempted') is True and guard.get('timed_out') is False and
                guard.get('interrupted_signal') is None and guard.get('error') is None and
                guard.get('session_drained') is True and guard.get('session_members_before_cleanup') == [] and
                guard.get('session_members_after_cleanup') == [], 'actual original guard ended naturally and drained')
        reservation = guard.get('reservation_id')
        require(type(reservation) is str and reservation and reservation == child.get('guard_reservation_id'),
                'actual original guard reservation matches raw child')
        events = ledger.get('events')
        require(type(events) is list, 'actual original GPU budget events')
        matches = [row for row in events if type(row) is dict and row.get('reservation_id') == reservation]
        require(len(matches) == 1 and matches[0] == guard,
                'unique actual completed guard event is already recorded in unchanged original budget')
        sid = guard.get('session_id')
        require(type(sid) is int and sid > 0 and session_probe(sid) == [],
                'original guard OS session still has processes; no fabricated drain')
        pair_document = _closed_leaf(root, config.get('pair_config_ref'), refs, driver=driver)
        require(pair_document.get('schema') == 'strong_native_u_i_cpu_configuration_v1', 'actual original strong pair source')
        pair = pair_document['configurations']
        manifest = _closed_leaf(root, config.get('workload_ref'), refs, driver=driver)
        partition = ROLES[role]
        arm = config['arm']
        storage = pair[arm]['engine']['kv_transfer_config']['kv_connector_extra_config']['shared_storage_path']
        require(Path(storage).is_absolute(), 'actual absolute retained peer namespace')
        storage_relative = Path(storage).relative_to(Path(root).resolve(strict=True)).as_posix()
        require(storage_relative.startswith('experiments/prefix_io_v1/runs/') and
                driver.safe(root, storage_relative).is_dir(), 'retained actual peer namespace required; never create it')
        target = driver.safe(root, output_relative)
        require(output_relative.endswith('.json') and
                output_relative.startswith(('artifacts/prefix_io_v1/', 'experiments/prefix_io_v1/runs/')) and
                target.parent.is_dir() and not target.exists() and output_relative not in refs,
                'fresh bounded parent output with existing directory; no draft/ref cycle or overwrite')
        parent = deepcopy(child)
        parent['os_session_drained'] = True
        parent.update(completed_guard_ref=deepcopy(guard_ref), child_result_ref=deepcopy(child_ref),
                      formal_parent_closure_schema=CLOSURE_SCHEMA)
        unchanged_parent = deepcopy(parent)
        checked = peer_checker(parent, root, config, refs, pair, partition=partition,
                               peer_arm=arm, peer_storage=storage, manifest=manifest, driver=driver)
        require(checked == unchanged_parent and parent == unchanged_parent,
                'original pure peer checker returned this unchanged complete parent')
        cpu_modules_only()
        require(session_probe(sid) == [], 'original guard session ceased to be empty during CPU verification')
        require(ledger_path.read_bytes() == before, 'original budget bytes changed; never write a stale parent')
        for row in (source_lock_ref, child_ref, guard_ref, config_ref):
            driver.check_ref(root, row)
        driver.new_json(target, parent)
        output_ref = driver.ref(root, output_relative)
        require(ledger_path.read_bytes() == before, 'original budget unexpectedly changed after parent append')
        return dict(schema='formal_native_peer_parent_CPU_delivery_v1', status='PASS_ACTUAL_GUARD_CLOSED_FORMAL_PARENT',
                    parent_ref=output_ref, child_ref=child_ref, guard_ref=guard_ref, config_ref=config_ref,
                    source_lock_ref=source_lock_ref, original_budget_sha256=hashlib.sha256(before).hexdigest(),
                    original_budget_bytes_unchanged=True, original_OS_session_reverified_empty=True,
                    actual_GPU_operations=0, formal_effect_qualified=False, gpu_eligible=False,
                    output_must_enter_later_source_lock_before_consumption=True)


def bootstrap_driver(root, row):
    """Load the actual frozen standard-library driver without invoking its CLI."""
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
            Path(row['path']).name == 'strong_trace_runner_v3.py' and actual_ref(root, row['path']) == row,
            'actual original thin V3 CPU driver source bytes')
    path = safe(root, row['path'])
    require(row['bytes'] <= 1024**2, 'bounded CPU driver source')
    tree = ast.parse(path.read_bytes())
    allowed = {'__future__', 'argparse', 'ast', 'copy', 'hashlib', 'importlib.util', 'json', 'math',
               'os', 'pathlib', 're', 'shutil', 'sys', 'time'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            require(all(alias.name in allowed for alias in node.names), 'standard-library driver module imports')
        elif isinstance(node, ast.ImportFrom):
            require(node.module in allowed, 'standard-library driver module imports')
    name = '_actual_formal_parent_CPU_driver_' + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[name] = driver
    spec.loader.exec_module(driver)
    require(actual_ref(root, row['path']) == row, 'driver source changed during import')
    cpu_modules_only()
    return driver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--source-lock', required=True, help='actual project-relative current complete source lock')
    parser.add_argument('--child', required=True, help='actual raw formal native child file, in current lock')
    parser.add_argument('--guard', required=True, help='actual completed original guard file, in current lock')
    parser.add_argument('--config', required=True, help='actual peer run config file, in current lock')
    parser.add_argument('--output-relative', required=True)
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        config_ref = actual_ref(root, args.config)
        config = bounded_json(safe(root, args.config))
        require(type(config) is dict, 'actual peer config JSON object')
        driver = bootstrap_driver(root, config.get('runner_ref'))
        lock_ref = actual_ref(root, args.source_lock)
        # Self and bridge must exist in the current immutable closure. This
        # check does not add the future parent output to its own input lock.
        rows = driver.source_rows(root, lock_ref, full=False)
        self_relative = Path(__file__).resolve().relative_to(root).as_posix()
        require(rows.get(self_relative) == actual_ref(root, self_relative), 'actual CPU producer source is frozen')
        require(rows.get(config_ref['path']) == config_ref and rows.get(config['runner_ref']['path']) == config['runner_ref'],
                'actual driver and peer config inherited unchanged in current source closure')
        bridge = driver.formal_namespace_module(root, config, rows)
        result = produce_parent(root, source_lock_ref=lock_ref,
            child_ref=actual_ref(root, args.child), guard_ref=actual_ref(root, args.guard), config_ref=config_ref,
            output_relative=args.output_relative, driver=driver, peer_checker=bridge.verify_closed_peer_document)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='BLOCKED_CPU_FORMAL_PARENT_NOT_QUALIFIED', reason=str(exc),
                              no_output_claim_made_if_append_itself_failed=True,
                              actual_GPU_operations=0, formal_effect_qualified=False, gpu_eligible=False)))
        return 78


if __name__ == '__main__':
    raise SystemExit(main())
