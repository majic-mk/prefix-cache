"""Package CPU source/evidence only; preserve the original GPU ledger and archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

STAGE = 'artifacts/prefix_io_v1/server12-natural-input-cpu-20261005'
PREP = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
GPU = 'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
GPU_ARCHIVE_SHA = '5904a4df64fdd01eef51b84372abbd1a1b77aee26666d181e5201f7e285589ad'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def actual(root, name):
    need(name and not name.startswith('/') and all(x not in ('', '.', '..') for x in name.split('/')), 'relative path')
    path = root / name
    need(not any(p.is_symlink() for p in (path,) + tuple(path.parents)), 'no symlink')
    need(path.resolve(strict=True).is_relative_to(root) and path.is_file(), 'contained regular file')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'file drift')
    return dict(path=name, bytes=after.st_size, sha256=digest)


def put(path, doc):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation')
    root = args.project_root.resolve(strict=True)
    ledger = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before = ledger.read_bytes()
    need(hashlib.sha256(before).hexdigest() == LEDGER_SHA and json.loads(before)['active_reservation'] is None, 'unchanged idle budget')
    prior = actual(root, GPU + '/GPU_DELIVERY_SOURCE_EVIDENCE.zip')
    need(prior['sha256'] == GPU_ARCHIVE_SHA, 'original verified GPU archive unchanged')
    names = set()
    excluded = {'NATURAL_CPU_SOURCE_EVIDENCE.zip', 'NATURAL_CPU_ARCHIVE_VERIFICATION.json', 'LOCAL_NATURAL_CPU_ARCHIVE_VERIFICATION.json'}
    for path in (root / STAGE).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in excluded and path.suffix in ('.py', '.json', '.md', '.txt', '.log'):
            names.add(path.relative_to(root).as_posix())
    names.update(PREP + '/' + name for name in (
        'protocol/prerental_protocol.py',
        'PRERENT_SOURCE_LOCK_V12.json', 'PRERENT_SOURCE_PROOF_V12.json',
        'freeze_prerental_sources_v12.py',
    ))
    names.add('models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444/tokenizer_config.json')
    previous_CPU_archive_name = 'artifacts/prefix_io_v1/server12-strategy-cpu-bridge-20261005/CPU_BRIDGE_SOURCE_EVIDENCE.zip'
    previous_CPU_archive_ref = actual(root, previous_CPU_archive_name)
    need(previous_CPU_archive_ref['sha256'] == 'ec681577173d7f89f67227c35bbc8f4d2aa82a4db8705cae40711b56f1aee5be', 'prior CPU bridge archive unchanged')

    names.update((
        'third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py',
        'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py',
    ))
    rows = [actual(root, name) for name in sorted(names)]
    need(len(rows) < 1000 and sum(row['bytes'] for row in rows) < 128 * 1024**2, 'bounded CPU evidence archive')
    scope = dict(schema='natural_raw_CPU_delivery_scope_v1', actual_new_GPU_operations=0,
                 previous_verified_GPU_archive_ref=prior, previous_verified_CPU_bridge_archive_ref=previous_CPU_archive_ref,
                 full_model_SDK_and_private_cache_payloads_included=False, tokenizer_binary_and_full_tokenizer_JSON_included=False,
                 full_instance_backup=False, gpu_eligible=False, strategy_effect_verified=False,
                 source_lock_contains_external_asset_refs=True,
                 instruction='Use the unchanged prior verified GPU evidence archive and retained server assets for raw calibration dependencies.')
    target = root / STAGE / 'NATURAL_CPU_SOURCE_EVIDENCE.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in rows:
            archive.write(root / row['path'], row['path'])
        archive.writestr('FILE_MANIFEST.json', json.dumps(dict(files=rows), sort_keys=True, indent=2) + '\n')
        archive.writestr('SCOPE.json', json.dumps(scope, sort_keys=True, indent=2) + '\n')
    with zipfile.ZipFile(target) as archive:
        need(len(archive.namelist()) == len(rows) + 2 and len(set(archive.namelist())) == len(rows) + 2, 'unique archive names')
        for row in rows:
            raw = archive.read(row['path'])
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'actual archive member bytes and CRC')
            need(actual(root, row['path']) == row, 'source changed while packaging')
        need(json.loads(archive.read('SCOPE.json')) == scope, 'scope bytes')
    need(ledger.read_bytes() == before and actual(root, prior['path']) == prior and actual(root, previous_CPU_archive_ref['path']) == previous_CPU_archive_ref, 'original budget/archives preserved')
    report = dict(status='PASS_ACTUAL_SERVER_NATURAL_CPU_ARCHIVE', archive_ref=actual(root, target.relative_to(root).as_posix()),
                  source_files=len(rows), archive_members=len(rows) + 2,
                  source_bytes=sum(row['bytes'] for row in rows), every_member_CRC_length_and_SHA_verified=True,
                  original_budget_sha256=LEDGER_SHA, previous_GPU_archive_preserved=True,
                  actual_new_GPU_operations=0, strategy_effect_verified=False)
    put(root / STAGE / 'NATURAL_CPU_ARCHIVE_VERIFICATION.json', report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
