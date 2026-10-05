"""Archive actual CPU evidence and public payloads; preserve all prior assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

A = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
P = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
ZIP_NAME = 'PUBLIC_DEVELOPMENT_CPU_SOURCE_EVIDENCE.zip'
VERIFY_NAME = 'PUBLIC_DEVELOPMENT_CPU_ARCHIVE_VERIFICATION.json'
PRIOR = [
 ('artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/GPU_DELIVERY_SOURCE_EVIDENCE.zip',
  '5904a4df64fdd01eef51b84372abbd1a1b77aee26666d181e5201f7e285589ad'),
 ('artifacts/prefix_io_v1/server12-strategy-cpu-bridge-20261005/CPU_BRIDGE_SOURCE_EVIDENCE.zip',
  'ec681577173d7f89f67227c35bbc8f4d2aa82a4db8705cae40711b56f1aee5be'),
 ('artifacts/prefix_io_v1/server12-natural-input-cpu-20261005/NATURAL_CPU_SOURCE_EVIDENCE.zip',
  '6d0451347a893c2176c3d6c1e3147b7241cdc1a263c0c5e65a245662b7f8c647')]


def need(value, message):
    if not value:
        raise ValueError('PUBLIC_CPU_ARCHIVE_REJECTED: ' + message)


def actual(root, relative):
    need(type(relative) is str and relative and not relative.startswith('/') and ':' not in relative and
         '\\' not in relative and all(x not in ('', '.', '..') for x in relative.split('/')), 'contained path')
    path = root
    for part in relative.split('/'):
        path /= part
        need(not path.is_symlink(), 'no symlink')
    need(path.is_file() and path.resolve(strict=True).is_relative_to(root), 'regular project file')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'file drift')
    return dict(path=relative, bytes=after.st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation')
    root = args.project_root.resolve(strict=True)
    ledger = root / LEDGER
    before = ledger.read_bytes()
    need(hashlib.sha256(before).hexdigest() == LEDGER_SHA and
         json.loads(before)['active_reservation'] is None, 'original idle GPU budget')
    prior_refs = []
    for name, expected in PRIOR:
        row = actual(root, name)
        need(row['sha256'] == expected, 'original verified archive unchanged')
        prior_refs.append(row)
    names = set()
    excluded = {ZIP_NAME, VERIFY_NAME, 'LOCAL_PUBLIC_DEVELOPMENT_CPU_ARCHIVE_VERIFICATION.json'}
    for path in (root / A).rglob('*'):
        need(not path.is_symlink(), 'no audit source symlinks')
        if (path.is_file() and '__pycache__' not in path.parts and path.name not in excluded and
                path.suffix in ('.py', '.json', '.md', '.txt', '.log')):
            names.add(path.relative_to(root).as_posix())
    names.update(P + '/' + name for name in (
        'runner/strong_trace_runner_v4.py', 'runner/native_runtime_v4.py',
        'development_runtime_bridge/development_diagnostic_bridge.py',
        'formal_trace_binding/formal_trace_binding.py', 'formal_runtime_bridge/namespace_bridge.py',
        'runner/activation_request.py', 'protocol/prerental_protocol.py'))
    names.update((
        'third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py',
        'third_party/upstream/kvcache-experiments/common/prefix_cache_common.py'))
    rows = [actual(root, name) for name in sorted(names)]
    need(len(rows) < 1000 and sum(row['bytes'] for row in rows) < 128 * 1024**2, 'bounded delivery')
    scope = dict(schema='public_development_CPU_delivery_archive_scope_v1', actual_new_GPU_runs=0,
        source_files_cutoff='actual_packaging_command_start', prior_verified_archives=prior_refs,
        full_official_public_payload_and_transport_and_failed_partials_included=True,
        source_or_result_not_GPU_permission=True, current_CPU_preflight_does_not_prove_method_gain=True,
        model_SDK_tokenizer_binary_and_private_cache_payloads_included=False, full_instance_backup=False,
        source_lock_external_assets_retained_on_server=True)
    target = root / A / ZIP_NAME
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in rows:
            archive.write(root / row['path'], row['path'])
        archive.writestr('FILE_MANIFEST.json', json.dumps(dict(files=rows), indent=2, sort_keys=True) + '\n')
        archive.writestr('SCOPE.json', json.dumps(scope, indent=2, sort_keys=True) + '\n')
    with zipfile.ZipFile(target) as archive:
        need(len(archive.namelist()) == len(rows) + 2 and
             len(set(archive.namelist())) == len(rows) + 2, 'unique complete archive')
        for row in rows:
            raw = archive.read(row['path'])
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                 'actual archived length/SHA/CRC')
            need(actual(root, row['path']) == row, 'source changed during archive')
        need(json.loads(archive.read('SCOPE.json')) == scope, 'actual scope')
    need(ledger.read_bytes() == before and all(actual(root, row['path']) == row for row in prior_refs),
         'original ledger and prior three archives preserved')
    result = dict(status='PASS_ACTUAL_SERVER_PUBLIC_DEVELOPMENT_CPU_ARCHIVE',
        archive_ref=actual(root, A + '/' + ZIP_NAME), source_files=len(rows), archive_members=len(rows) + 2,
        source_bytes=sum(row['bytes'] for row in rows), every_member_CRC_length_and_SHA_verified=True,
        prior_verified_archives_preserved=True, GPU_ledger_unchanged=True, actual_new_GPU_runs=0,
        method_effect_proven=False)
    with (root / A / VERIFY_NAME).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
