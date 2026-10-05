"""Freeze actual existing source/model/SDK bytes; never import a GPU backend."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

REL = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ANCESTOR = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/PRERENT_SOURCE_LOCK_V10.json'
ANCESTOR_SHA = '069f96134f950e2d1100fdc751a2449145f7d534a9e625030503462dfdd1273a'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def safe(root, name):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name and ':' not in name,
            'relative project path')
    require(all(x not in ('', '.', '..') for x in name.split('/')), 'no traversal')
    p = root
    for part in name.split('/'):
        p /= part
        require(not p.is_symlink(), 'no source symlink')
    require(p.resolve(strict=True).is_relative_to(root), 'contained source')
    return p


def file_ref(root, name):
    p = safe(root, name)
    require(p.is_file(), 'regular source')
    before = p.stat()
    with p.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = p.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'source drift while hashing')
    return dict(path=name, bytes=before.st_size, sha256=sha)


def write(p, doc):
    with p.open('x') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only actual invocation')
    root = args.project_root.resolve(strict=True)
    directory = safe(root, REL)
    ledger_raw = safe(root, LEDGER).read_bytes()
    require(hashlib.sha256(ledger_raw).hexdigest() == LEDGER_SHA and
            json.loads(ledger_raw)['active_reservation'] is None, 'original unchanged idle budget')
    ancestor_raw = safe(root, ANCESTOR).read_bytes()
    require(hashlib.sha256(ancestor_raw).hexdigest() == ANCESTOR_SHA, 'frozen original ancestry')
    ancestor = json.loads(ancestor_raw)
    require(len(ancestor['files']) == 4893, 'complete prior source/model/SDK closure')
    rows = {}
    started = time.monotonic()
    for expected in ancestor['files']:
        actual = file_ref(root, expected['path'])
        require(actual == expected, 'original actual source byte mismatch: ' + expected['path'])
        require(expected['path'] not in rows, 'duplicate inherited source')
        rows[expected['path']] = actual
    rows[ANCESTOR] = file_ref(root, ANCESTOR)
    source_sets = ['source_inputs', 'runner', 'protocol', 'activation', 'raw_device_binding', 'raw_relative_config', 'raw_cache_semantics', 'raw_capture_diagnostics', 'raw_no_forward_observation', 'raw_no_forward_binding', 'raw_initial_execution_contract','raw_runtime_collector_binding','formal_trace_binding','formal_runtime_bridge']
    included = [directory / 'freeze_prerental_sources_v11b.py', directory / 'entry_control_v4.py', directory / 'audit_project_sdk_cpu.py', root / 'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/prepare_actual_exact_attempt_v1.py', root / 'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/supervise_actual_raw_original_guard_v3.py']
    for folder in source_sets:
        base = directory / folder
        if base.is_dir():
            for p in base.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.yaml'):
                    if any(s in p.name for s in ('RESULT', 'STDOUT', 'STDERR', 'MANIFEST', 'DELIVERY', 'CPU_CONTRACT')):
                        continue
                    included.append(p)
    stage=root/'artifacts/prefix_io_v1/server12-strategy-cpu-bridge-20261005'
    for folder in ['formal_runner','formal_runtime','formal_close','input_audit','semantic_review']:
        base=stage/folder
        if base.is_dir():
            included.extend(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                            p.suffix in ('.py','.json') and not any(x in p.name for x in ('RESULT','STDOUT','STDERR','MANIFEST')))
    included.extend([stage/'ORIGINAL_AUTHOR_NATURAL_PARSER_SOURCE_REFS.json',stage/'P4_NEXT_INPUT_REQUIREMENTS.json',stage/'build_cpu_bridge_delivery.py',
                     root/'third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py',
                     root/'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py'])
    for p in included:
        require(p.is_file(), 'required CPU entry source exists: ' + str(p))
        name = p.relative_to(root).as_posix()
        rows[name] = file_ref(root, name)
    # Metadata lists point at original server files. Bind those originals too,
    # including current engine APIs, P3 inputs and historical planner curves.
    for p in (directory / 'source_inputs').glob('*.json'):
        obj = json.loads(p.read_bytes())
        references = obj.get('files', []) if isinstance(obj, dict) else []
        if p.name == 'ORIGINAL_PLANNER_CURVES_INPUT.json':
            references = [obj]
        for row in references:
            if type(row) is not dict or set(row) != {'path', 'bytes', 'sha256'}:
                continue
            path = Path(row['path'])
            if path.is_absolute():
                require(path.is_relative_to(root), 'original actual input inside project')
                row = dict(row, path=path.relative_to(root).as_posix())
            if row['path'] in rows:
                require(rows[row['path']] == row, 'conflicting actual source row')
            else:
                require(file_ref(root, row['path']) == row, 'original actual input reference')
                rows[row['path']] = row
    # Complete the inherited thin executor's explicit execution/lifecycle
    # contract; ancestry size alone cannot establish a complete closure.
    runner_path = directory / 'runner/strong_trace_runner_v3.py'
    spec = importlib.util.spec_from_file_location('_prerent_v3_source_contract', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name in runner.REQUIRED:
        actual = file_ref(root, name)
        if name in rows:
            require(rows[name] == actual, 'conflicting required execution source')
        rows[name] = actual
    require(set(runner.REQUIRED) <= rows.keys(), 'complete original execution/lifecycle/guard/table closure')
    # Bind the actual completed off and the existing device-specific permission.
    # No source count or CPU flag substitutes for the original GPU guard.
    extra_sources = [
        REL+'/live-off01/EFFECTIVE_GPU_PERMISSION.json',
        REL+'/PRERENT_SOURCE_PROOF_V10.json',
        'experiments/prefix_io_v1/runs/server12-strong-u-qual-off01/result.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/OFF_GUARD_CLOSED_QUALIFICATION.json',
        'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/PRERENT_SOURCE_PROOF_V9.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/BOUND_PLAN.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/PRELAUNCH_INTENT.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/ACTUAL_RAW_PARENT_CLOSED.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/ACTUAL_COST_ISSUER_REPORT.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/SOURCE_BEFORE.json',
        'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/SOURCE_AFTER.json',
        'experiments/prefix_io_v1/runs/server12-strong-exact-cal05/result.json']
    for name in extra_sources:
        actual = file_ref(root, name)
        if name in rows:
            require(rows[name] == actual, 'conflicting actual completed off/permission source')
        rows[name] = actual
    require(safe(root, LEDGER).read_bytes() == ledger_raw, 'original ledger unchanged during full CPU freeze')
    lock_path = directory / 'PRERENT_SOURCE_LOCK_V11.json'
    lock = dict(schema='strong_trace_source_lock_v1', files=[rows[n] for n in sorted(rows)],
                ancestry_ref=file_ref(root, ANCESTOR), GPU_launch_allowed=False,
                production_qualified=False, strategy_effect_verified=False,
                model_and_SDK_loaded=False, observation_scope='this_CPU_source_freeze_action_only')
    write(lock_path, lock)
    proof = dict(schema='strong_trace_source_proof_v1', status='PASS_FULL_CPU_SOURCE_BYTES',
                 source_lock_ref=file_ref(root, lock_path.relative_to(root).as_posix()),
                 source_count=len(rows), actual_gpu_runs=0,
                 total_bytes_verified=sum(row['bytes'] for row in rows.values()),
                 elapsed_seconds=time.monotonic()-started,
                 original_idle_ledger_sha256=LEDGER_SHA,
                 source_ancestry_is_not_GPU_authority=True,
                 GPU_qualification=False, effect_verified=False)
    write(directory / 'PRERENT_SOURCE_PROOF_V11.json', proof)
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()
