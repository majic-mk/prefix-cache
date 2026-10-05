"""CPU-only new-machine permission and exact independent SSD input copy."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DEST = ROOT / 'artifacts/prefix_io_v1/server11-native-cost-v1-20261003'
STORAGE = ROOT / 'experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage'
ORIGINAL = ROOT / 'experiments/prefix_io_v1/runs/server10-kv-byte-02-private-storage'
GPU = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'
PERMISSION = 'experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml'
SOURCE_REFS = [
    dict(path='experiments/prefix_io_v1/runs/server10-kv-byte-populate-02/result.json', bytes=1338,
         sha256='29502f3caee76b434e164c8072b7c2dd4f71caf4a9e5ad02e0c491ddcbfef39d'),
    dict(path='experiments/prefix_io_v1/runs/server10-kv-byte-populate-02/details/storage-publication.json', bytes=8044,
         sha256='f00027a3653cc01c5b1155fd67d73e82d15f1d4ca48867b8da1cb0155181ce83'),
    dict(path='experiments/prefix_io_v1/runs/server10-kv-byte-populate-02/details/kv-capture/native-kv-byte-receipt.json', bytes=95364,
         sha256='eb3b5795281facfd52e96827d93cddbfac1279ddb29a764e8be81f529e765020'),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path):
    assert path.resolve().is_relative_to(ROOT)
    cursor = path
    while cursor != ROOT:
        assert not cursor.is_symlink(), str(cursor)
        cursor = cursor.parent
    return path


def ref(path):
    raw = safe(path).read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=sha(raw))


def put(path, raw):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    assert path.read_bytes() == raw
    return ref(path)


def put_json(path, value):
    return put(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode())


def main():
    assert ROOT.resolve(strict=True) == ROOT and not STORAGE.exists()
    assert os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize > 8*1024**3 + 1024**3
    for row in SOURCE_REFS:
        assert ref(ROOT/row['path']) == row
    guard, publication, capture = [json.loads((ROOT/row['path']).read_bytes()) for row in SOURCE_REFS]
    assert guard['exit'] == guard['child_exit'] == 0 and guard['session_drained'] and not guard['timed_out']
    assert capture['status'] == 'PASS_NATIVE_KV_BYTE_DIAGNOSTIC' and not capture['failures']
    assert publication['storage'] == str(ORIGINAL) and publication['file_count'] == 24
    originals = {Path(row['path']).stem: row for row in publication['files']}
    assert len(originals) == 24 and set(originals) == set(capture['files'])
    ledger_raw = (ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json').read_bytes()
    ledger = json.loads(ledger_raw)
    assert ledger['active_reservation'] is None and ledger['gpu_wall_seconds'] + 620 <= 28800
    assert sha(ledger_raw) == '74bf74b2d90f6da4ba4ed70ad68d500cab1e415400c844820ed665e1729cc607'
    DEST.mkdir(exist_ok=True)
    groups = []
    stores = sorted({row['job_id'] for row in capture['files'].values()}, key=lambda key:int(key.split(':')[1]))
    assert len(stores) == 3
    for index, job in enumerate(stores):
        records = sorted(((key, row) for key, row in capture['files'].items() if row['job_id'] == job),
                         key=lambda pair:pair[1]['file_index'])
        assert [row['file_index'] for key,row in records] == list(range(8))
        files = []
        for key, row in records:
            old = originals[key]
            raw = safe(ORIGINAL/old['path']).read_bytes()
            assert len(raw) == old['bytes'] == row['bytes'] == 917504
            assert sha(raw) == old['sha256'] == row['sha256']
            put(STORAGE/old['path'], raw)
            files.append(dict(path=old['path'], bytes=len(raw), sha256=sha(raw), block_hash=key,
                              producer_file_index=row['file_index']))
        groups.append(dict(pair_index=index, producer_job_id=job, files=files))
    input_manifest = dict(schema_version=1, status='EXACT_COPY_OF_VERIFIED_MODEL_PRODUCED_KV',
                          storage=STORAGE.relative_to(ROOT).as_posix(), groups=groups,
                          source_refs=SOURCE_REFS, file_count=24, total_bytes=22020096,
                          GPU_runs_this_preparation=0, source_data_modified=False,
                          new_machine_GPU_qualified=False, production_qualified=False)
    manifest_ref = put_json(DEST/'SSD_INPUT_MANIFEST.json', input_manifest)
    permission = (ROOT/'experiments/prefix_io_v1/configs/permissions.server10.reference.yaml').read_text()
    permission = permission.replace('# User-designated server10 migration, 2026-10-03; original cumulative budget and restrictions preserved.',
                                    '# User-designated server11, 2026-10-03; original cumulative budget and restrictions preserved.')
    assert permission.count('GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f') == 1
    permission = permission.replace('GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f', GPU)
    permission_ref = put(ROOT/PERMISSION, permission.encode())
    authorization = dict(origin='direct_human_instruction', credentials_omitted=True,
        user_instruction='又为你申请了一个新的服务器，代码全部克隆到新的服务器，在新的服务器完成剩余的操作',
        target=dict(host='connect.westc.seetacloud.com', port=26909, gpu_uuid=GPU),
        bounded_interpretation='One initial native sustained-decode calibration/holdout job, six 128-token windows plus three one-token warmups; later work requires its own concrete bounded scope within this same human instruction and original budget.',
        maximum_initial_jobs=1, initial_job_label='server11-native-cost-six-window-01',
        seconds_limit=600, reserved_seconds=620, PRIMARY_reserve_bytes=1024**3,
        PRIMARY_minimum_free_bytes=8*1024**3, cumulative_gpu_limit_seconds=28800,
        permission_ref=permission_ref, source_ledger_ref=ref(ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'),
        no_downloads=True, no_system_or_driver_changes=True, no_deletions=True,
        formal_SLO_or_performance_claim=False)
    auth_ref = put_json(DEST/'DIRECT_USER_AUTHORIZATION_AND_INITIAL_SCOPE.json', authorization)
    assert (ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json').read_bytes() == ledger_raw
    result = dict(status='PASS_CPU_MIGRATION_INPUT_PREPARATION', input_manifest_ref=manifest_ref,
                  permission_ref=permission_ref, authorization_ref=auth_ref,
                  copied_files=24, copied_bytes=22020096, original_ledger_unchanged=True, GPU_runs=0)
    put_json(DEST/'CPU_MIGRATION_PREPARATION_RESULT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
