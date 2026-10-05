"""CPU preparation and one bounded, source-locked strong U qualification job.

The launch verb is explicit. CPU preflight never probes a GPU, loads a shared
library, reserves GPU time or changes the original permission/usage ledger.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REL = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
PREVIOUS = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
BASE = 'experiments/prefix_io_v1/configs/permissions.yaml'
LABEL = 'server12-strong-u-qual-off01'
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
MAX = 28800
SECONDS = 300
RESERVE = 20


def require(value, reason):
    if not value:
        raise ValueError('PRERENT_ENTRY_REJECTED: ' + reason)


def safe(root, name, existing=True):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name and ':' not in name,
            'relative project POSIX path')
    require(all(p not in ('', '.', '..') for p in name.split('/')), 'no path traversal')
    cursor = root
    for part in name.split('/'):
        cursor /= part
        require(not cursor.is_symlink(), 'no source or output symlink')
    require(cursor.resolve(strict=existing).is_relative_to(root), 'project-contained path')
    return cursor


def document(path):
    require(path.stat().st_size <= 4 * 1024**2, 'bounded JSON metadata')
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite metadata'))


def file_ref(root, name):
    path = safe(root, name)
    require(path.is_file(), 'regular source file')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(path.stat().st_size == before.st_size, 'file drift')
    return dict(path=name, bytes=before.st_size, sha256=sha)


def module(root, row, name):
    require(file_ref(root, row['path']) == row, 'actual helper source pin')
    spec = importlib.util.spec_from_file_location(name, safe(root, row['path']))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def budget(ledger):
    used = ledger.get('gpu_wall_seconds')
    require(type(used) in (int, float) and used == used and 0 <= used <= MAX, 'finite original cumulative GPU time')
    require(ledger.get('active_reservation') is None, 'no unresolved prior GPU session')
    require(used + SECONDS + RESERVE <= MAX, 'original budget including cleanup reserve')
    return MAX - used


def effective_permission(base, gpu_uuid):
    require(type(base) is dict and base.get('allow_gpu_runs') is True and
            type(base.get('max_gpu_hours')) in (int, float) and base['max_gpu_hours'] == 8,
            'original project GPU grant and eight-hour ceiling')
    require(type(gpu_uuid) is str and UUID.fullmatch(gpu_uuid), 'one actually observed GPU UUID')
    out = dict(base)
    out['approved_gpu_ids'] = [gpu_uuid]
    for key in ('allow_model_downloads', 'allow_driver_or_system_changes', 'allow_shared_data_deletion',
                'allow_payment', 'allow_new_cloud_rental', 'allow_remote_push'):
        out[key] = False
    out.pop('approved_auxiliary_storage', None)
    return out


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def checked_sources(root, full):
    lock_name = REL + '/PRERENT_SOURCE_LOCK.json'
    proof_name = REL + '/PRERENT_SOURCE_PROOF.json'
    lock = document(safe(root, lock_name))
    proof = document(safe(root, proof_name))
    require(lock.get('schema') == 'strong_trace_source_lock_v1' and
            proof.get('schema') == 'strong_trace_source_proof_v1' and
            proof.get('status') == 'PASS_FULL_CPU_SOURCE_BYTES' and
            proof.get('source_lock_ref') == file_ref(root, lock_name) and
            proof.get('actual_gpu_runs') == 0 and type(proof.get('actual_gpu_runs')) is int,
            'actual full CPU source closure/proof')
    rows = lock.get('files')
    require(type(rows) is list and len(rows) == proof.get('source_count') and len(rows) >= 4749,
            'complete inherited source/model/SDK closure')
    refs = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
                type(row['bytes']) is int and row['bytes'] >= 0 and
                type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'strict source reference')
        require(row['path'] not in refs, 'no duplicate source row')
        path = safe(root, row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'], 'current existing source size')
        if full or path.suffix in ('.py', '.json', '.yaml'):
            require(file_ref(root, row['path']) == row, 'current source byte drift')
        refs[row['path']] = row
    return lock, proof, refs


def preflight(root, full=False):
    lock, proof, refs = checked_sources(root, full)
    remaining = budget(document(safe(root, LEDGER)))
    auth_name = PREVIOUS + '/calibration_v2/standing_gpu_authorization.py'
    auth = module(root, refs[auth_name], '_prerent_standing_authorization')
    _, grant_ref = auth.standing_grant(root)
    for relative in ('runner/strong_trace_runner.py', 'runner/strong_native_runtime.py',
                     'protocol/CONTROLLED_P3_OFF_QUALIFICATION_WORKLOAD.json', 'runner/PRIVATE_U_I_CONFIG.json'):
        require(REL + '/' + relative in refs, 'actual CPU-prepared input: ' + relative)
    require(shutil.disk_usage(root).free >= (8 * 1024**3) + (128 * 1024**2), 'PRIMARY floor and bounded reservation')
    require(not safe(root, 'experiments/prefix_io_v1/runs/' + LABEL, existing=False).exists(), 'single fresh qualification job')
    return dict(schema='actual_GPU_prerental_CPU_entry_preflight_v1',
                status='PASS_CPU_BOUNDED_STRONG_U_QUALIFICATION_ENTRY',
                actual_GPU_runs=0, source_count=len(refs), full_byte_recheck_this_command=full,
                source_lock_ref=file_ref(root, REL+'/PRERENT_SOURCE_LOCK.json'),
                source_proof_ref=file_ref(root, REL+'/PRERENT_SOURCE_PROOF.json'),
                standing_authorization_ref=grant_ref, original_remaining_seconds=remaining,
                label=LABEL, seconds_limit=SECONDS, cleanup_seconds=RESERVE,
                normal_source_and_config_GPU_qualified=False, formal_effect_qualified=False,
                current_device_nodes_visible=bool(list(Path('/dev').glob('nvidia*'))) if os.name == 'posix' else False,
                live_device_UUID_and_resource_recheck_required=True,
                source_ancestry_is_not_authority=True, natural_service_trace_bound=False,
                historical_planner_curves_qualify_current_cost=False)


def launch(root):
    # Absence is checked before running any GPU query. It does not imply that
    # a visible GPU is free or compatible; the inherited resource probe checks that.
    require(os.name == 'posix' and list(Path('/dev').glob('nvidia*')), 'GPU launch blocked: no actual NVIDIA device nodes')
    report = preflight(root, full=True)
    lock, _, refs = checked_sources(root, full=False)
    probe_name = PREVIOUS + '/calibration_v2/gpu_entry_binding.py'
    probe = module(root, refs[probe_name], '_prerent_original_resource_probe')
    resource = probe.probe_resources(root)
    gpu = probe._resource_shape(resource)
    # The original fixed base contract is checked by the existing human grant.
    # YAML is imported only in the explicit launch branch, never by CPU preflight.
    import yaml
    base = yaml.safe_load(safe(root, BASE).read_text())
    permission = effective_permission(base, gpu)
    require(permission['approved_dependency_root'] == str(root) and
            permission['approved_experiment_root'] == str(root/'experiments/prefix_io_v1/runs'), 'original approved roots')
    bound = safe(root, REL + '/live-off01', existing=False)
    bound.mkdir(exist_ok=False)
    permission_name = REL + '/live-off01/EFFECTIVE_GPU_PERMISSION.json'
    write(safe(root, permission_name, existing=False), permission)
    permission_ref = file_ref(root, permission_name)
    derived_rows = list(lock['files']) + [permission_ref]
    derived_name = REL + '/live-off01/BOUND_SOURCE_LOCK.json'
    write(safe(root, derived_name, existing=False), dict(lock, files=sorted(derived_rows, key=lambda r:r['path']),
          parent_source_lock_ref=report['source_lock_ref'], binding_origin='actual_device_specific_metadata_only'))
    derived_ref = file_ref(root, derived_name)
    proof_name = REL + '/live-off01/BOUND_SOURCE_PROOF.json'
    write(safe(root, proof_name, existing=False), dict(schema='strong_trace_source_proof_v1',
          status='PASS_FULL_CPU_SOURCE_BYTES', source_lock_ref=derived_ref, source_count=len(derived_rows),
          actual_gpu_runs=0, parent_full_source_proof_ref=report['source_proof_ref'],
          current_original_source_full_rechecked_before_binding=True,
          added_metadata_ref=permission_ref, GPU_qualification=False))
    runner_name = REL + '/runner/strong_trace_runner.py'
    runtime_name = REL + '/runner/strong_native_runtime.py'
    config_name = REL + '/live-off01/CONFIG.json'
    config = dict(schema='strong_native_trace_run_v1', phase='qualification', arm='U', mode='off', run_id=LABEL,
                  source_lock_ref=derived_ref, source_proof_ref=file_ref(root, proof_name),
                  pair_config_ref=refs[REL+'/runner/PRIVATE_U_I_CONFIG.json'],
                  workload_ref=refs[REL+'/protocol/CONTROLLED_P3_OFF_QUALIFICATION_WORKLOAD.json'],
                  permissions_ref=permission_ref, gpu_uuid=gpu, seconds_limit=SECONDS,
                  storage_reserve_bytes=128*1024**2, storage_floor_bytes=8*1024**3,
                  output_relative='experiments/prefix_io_v1/runs/'+LABEL+'/details',
                  runner_ref=refs[runner_name], runtime_ref=refs[runtime_name], off_qualification_ref=None)
    write(safe(root, config_name, existing=False), config)
    write(bound/'LIVE_CONTEXT.json', dict(resource=resource, standing_authorization_ref=report['standing_authorization_ref'],
          original_source_lock_ref=report['source_lock_ref'], derived_source_lock_ref=derived_ref,
          qualification_only=True, seconds_limit=SECONDS, reserved_seconds=SECONDS+RESERVE,
          origin='derived_from_existing_human_standing_authorization', new_human_approval_needed=False))
    runner = module(root, refs[runner_name], '_prerent_actual_trace_runner')
    runner.verify_configuration(root, safe(root, config_name), full=False)
    command = runner.guard_command(root, config_name, config)
    write(bound/'ORIGINAL_GUARD_COMMAND.json', dict(argv=command, gpu_UUID=gpu,
          reserved_seconds=SECONDS+RESERVE, no_data_deletions=True, no_driver_or_system_changes=True))
    return subprocess.run(command, cwd=root).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'launch'))
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--full-source-check', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    if args.action == 'launch':
        require(args.output is None, 'launch evidence uses fixed append-only namespace')
        return launch(root)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preflight requires empty device environment')
    result = preflight(root, full=args.full_source_check)
    if args.output is not None:
        require(args.output.resolve().is_relative_to(root/REL), 'CPU output in new audit directory')
        write(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
