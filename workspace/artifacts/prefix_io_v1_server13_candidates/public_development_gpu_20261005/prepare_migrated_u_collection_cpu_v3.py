"""Bind only new GPU U/off assets, inheriting the actual immutable V14 proof.

No old CPU suite, model download, whole-model hash, CUDA or inference is run.
The source proof states exactly inherited ancestry plus targeted byte checks.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

A = 'artifacts/prefix_io_v1/server13-public-development-gpu-20261005'
OLD = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
P = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ANCESTOR = OLD + '/PRERENT_SOURCE_LOCK_V14.json'
ANCESTOR_SHA = '427c796365d31a5ff7a725f2bfa52641a74a9c609b67fda918fa23834af343b4'
ANCESTOR_PROOF = OLD + '/PRERENT_SOURCE_PROOF_V14.json'
ANCESTOR_PROOF_SHA = 'e5392891a83f60c436e708aac85ebcca99717a52a185183ef573692c3b2cf8fd'
UUID = 'GPU-bff07e52-6cfb-5828-bb80-9aa47a3a4a9b'


def need(ok, message):
    if not ok:
        raise ValueError('MIGRATED_U_PREPARATION_REJECTED: ' + message)


def write(root, name, doc):
    with (root / name).open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def ref(root, name):
    need(type(name) is str and name and ':' not in name and '\\' not in name and not name.startswith('/') and
         all(part not in ('', '.', '..') for part in name.split('/')), 'safe relative path')
    path = root
    for part in name.split('/'):
        path /= part
        need(not path.is_symlink(), 'symlink source')
    need(path.is_file() and path.resolve(strict=True).is_relative_to(root), 'regular project source')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed during targeted hash')
    return dict(path=name, bytes=after.st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--addition-manifest', required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only metadata preparation')
    root = args.project.resolve(strict=True)
    ledger = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before_ledger = ledger.read_bytes()
    need(json.loads(before_ledger)['active_reservation'] is None, 'idle original budget')
    ancestor_ref, ancestor_proof_ref = ref(root, ANCESTOR), ref(root, ANCESTOR_PROOF)
    need(ancestor_ref['sha256'] == ANCESTOR_SHA and ancestor_proof_ref['sha256'] == ANCESTOR_PROOF_SHA,
         'actual copied immutable V14 ancestry')
    ancestor = json.loads((root / ANCESTOR).read_bytes())
    old_proof = json.loads((root / ANCESTOR_PROOF).read_bytes())
    need(ancestor['schema'] == 'strong_trace_source_lock_v1' and len(ancestor['files']) == 5014 and
         old_proof['source_lock_ref'] == ancestor_ref and old_proof['source_count'] == 5014 and
         old_proof['status'] == 'PASS_FULL_CPU_SOURCE_BYTES', 'actual inherited whole-byte proof')
    manifest = json.loads((root / args.addition_manifest).read_bytes())
    need(manifest['schema'] == 'explicit_migrated_U_source_additions_v1' and
         type(manifest['files']) is list and len(set(manifest['files'])) == len(manifest['files']),
         'explicit unique additions only')
    inherited = {row['path']: row for row in ancestor['files']}
    need(len(inherited) == 5014, 'unique complete inherited rows')
    rows = dict(inherited)
    for name in [ANCESTOR, ANCESTOR_PROOF, args.addition_manifest,
                 A + '/prepare_migrated_u_collection_cpu_v3.py', *manifest['files']]:
        actual = ref(root, name)
        need(name not in rows or rows[name] == actual, 'immutable ancestor drift')
        rows[name] = actual
    driver_name = P + '/runner/strong_trace_runner_v7.py'
    source = (root / driver_name).read_bytes()
    need(ref(root, driver_name) == rows[driver_name], 'actual frozen driver bytes')
    spec = importlib.util.spec_from_file_location('_actual_migrated_u_preparer_driver', root / driver_name)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[driver.__name__] = driver
    exec(compile(source, str(root / driver_name), 'exec', dont_inherit=True), driver.__dict__)
    required = []
    for name in driver.REQUIRED:
        actual = ref(root, name)
        need(rows.get(name) == actual, 'actual required original route source')
        required.append(actual)
    targeted = [rows[name] for name in sorted(set(rows) - set(inherited))]
    lock_name = A + '/MIGRATED_U_SOURCE_LOCK_03.json'
    proof_name = A + '/MIGRATED_U_SOURCE_PROOF_03.json'
    write(root, lock_name, dict(schema='strong_trace_source_lock_v1', files=[rows[name] for name in sorted(rows)],
        ancestry_ref=ancestor_ref, inherited_whole_source_proof_ref=ancestor_proof_ref,
        addition_manifest_ref=ref(root, args.addition_manifest),
        current_host_whole_source_hash_performed=False, full_source_verified=False,
        observation_scope='strict_new_GPU_development_Uoff_collection_only'))
    lock_ref = ref(root, lock_name)
    write(root, proof_name, dict(schema='migrated_u_collection_source_proof_v1',
        status='PASS_INHERITED_V14_PLUS_TARGETED_CPU_BYTES',
        ancestor_source_lock_ref=ancestor_ref, ancestor_source_proof_ref=ancestor_proof_ref,
        source_lock_ref=lock_ref, source_count=len(rows), ancestor_source_count=5014,
        targeted_source_refs=targeted, required_source_refs=required, actual_gpu_runs=0,
        full_source_verified=False, current_host_whole_source_hash_performed=False))
    config = dict(schema=driver.SCHEMA, phase='development', arm='U', mode='off',
        run_id='server13-public-development-uoff03', source_lock_ref=lock_ref, source_proof_ref=ref(root, proof_name),
        pair_config_ref=rows[A + '/MIGRATED_U_I_CONFIG_03.json'],
        workload_ref=rows[OLD + '/ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json'],
        permissions_ref=rows[A + '/EFFECTIVE_STANDING_GPU_PERMISSION_01.json'], gpu_uuid=UUID,
        seconds_limit=300, storage_reserve_bytes=134217728, storage_floor_bytes=8589934592,
        output_relative='experiments/prefix_io_v1/runs/server13-public-development-uoff03/details',
        runner_ref=rows[driver_name], runtime_ref=rows[P + '/runner/native_runtime_v7.py'],
        off_qualification_ref=None, collection_ref=rows[A + '/UNCALIBRATED_U_COLLECTION_DESCRIPTOR_04.json'],
        formal_trace_binding_ref=rows[A + '/MIGRATED_DEVELOPMENT_INPUT_BINDING_03.json'], formal_peer_closed_ref=None)
    config_name = A + '/MIGRATED_UOFF_RUN_CONFIG_03.json'
    write(root, config_name, config)
    need(ledger.read_bytes() == before_ledger, 'GPU ledger unchanged by metadata preparation')
    print(json.dumps(dict(status='PASS_TARGETED_MIGRATION_CONFIG_CREATED_REQUIRES_ORIGINAL_RUNNER_PREFLIGHT',
        config_ref=ref(root, config_name), source_lock_ref=lock_ref, source_proof_ref=ref(root, proof_name),
        source_count=len(rows), inherited_source_count=5014, newly_hashed_source_count=len(targeted),
        required_source_count=len(required), old_CPU_suites_repeated=0, whole_model_hash_repeated=False,
        actual_gpu_runs=0, formal_effect_qualified=False)))


if __name__ == '__main__':
    main()
