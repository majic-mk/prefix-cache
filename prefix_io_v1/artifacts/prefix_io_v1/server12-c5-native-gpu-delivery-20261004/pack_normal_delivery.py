"""Append a byte-verified normal evidence archive; never import or run an engine.

Run only after the original GPU guard is idle. This archives observations and
existing qualification reports, and does not itself qualify a GPU experiment.
The earlier calibration archive remains a separately pinned dependency.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
A = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
BASE = 'experiments/prefix_io_v1/configs/permissions.yaml'
MODES = ('off', 'shadow', 'on')
SCHEMA = 'server12_normal_normal_delivery_manifest_v1'
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 100 * 1024**2
MAX_MEMBERS = 1024
SUFFIXES = {'.py', '.md', '.json', '.log', '.txt', '.yaml'}
EXCLUDED_PARTS = {'__pycache__', 'runtime-cache', 'private-storage', 'fixture-tmp', 'recorded_files'}
EXCLUDED_NAMES = {'GPU_DELIVERY_MANIFEST.json', 'GPU_ARCHIVE_RESULT.json', 'GPU_EVIDENCE.tar.gz'}


def require(value, reason):
    if not value:
        raise ValueError('NORMAL_DELIVERY_REJECTED: ' + reason)


def relative_name(value):
    require(type(value) is str and 0 < len(value) <= 240 and
        all(32 <= ord(c) < 127 for c in value) and ':' not in value and '\\' not in value and
        not value.startswith('/') and all(p not in ('', '.', '..') for p in value.split('/')),
        'bounded plain project-relative path')
    require(not set(value.split('/')).intersection(EXCLUDED_PARTS), 'excluded cache/fixture path')
    return value


def safe(root, relative):
    relative_name(relative)
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink refused: ' + relative)
    require(path.resolve().is_relative_to(root), 'path outside project')
    return path


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def document(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def file_ref(root, relative):
    path = safe(root, relative)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= MAX_FILE,
        'bounded regular input file: ' + relative)
    size = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            size += len(block)
            require(size <= MAX_FILE, 'file grew above limit')
            digest.update(block)
    after = path.stat()
    require(size == before.st_size == after.st_size and
        (before.st_dev, before.st_ino, before.st_mtime_ns) ==
        (after.st_dev, after.st_ino, after.st_mtime_ns), 'input changed while hashing')
    return dict(path=relative, bytes=size, sha256=digest.hexdigest())


def read(root, relative):
    row = file_ref(root, relative)
    raw = safe(root, relative).read_bytes()
    require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
        'input changed while reading')
    return parse(raw), raw, row


def walk(root, relative):
    base = safe(root, relative)
    require(base.is_dir(), 'existing evidence directory required: ' + relative)
    pending = [base]
    while pending:
        path = pending.pop()
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            if child.name in EXCLUDED_PARTS or child.name.startswith('fixture-'):
                continue
            require(not child.is_symlink(), 'evidence symlink refused: ' + str(child))
            mode = child.stat().st_mode
            if stat.S_ISDIR(mode):
                pending.append(child)
            else:
                require(stat.S_ISREG(mode), 'nonregular evidence refused')
                if (child.suffix in SUFFIXES and child.name not in EXCLUDED_NAMES and
                    not child.name.startswith('NORMAL_NORMAL_DELIVERY_')):
                    # Earlier native commands are already archived separately.
                    if (path == safe(root, A) and re.match(r'^\d+_', child.name) and
                        int(child.name.split('_', 1)[0]) < 14):
                        continue
                    yield child.relative_to(root).as_posix()


def checked_ledger(root):
    value, raw, row = read(root, LEDGER)
    seconds = value.get('gpu_wall_seconds')
    require(type(value) is dict and value.get('active_reservation') is None and
        type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 28800 and
        type(value.get('events')) is list and len(value['events']) <= 10000,
        'original ledger must be idle and within its eight-hour budget')
    return value, raw, row


def put_bytes(root, relative, raw):
    path = safe(root, relative)
    require(path.parent.is_dir(), 'existing output directory required')
    with path.open('xb') as stream:
        stream.write(raw)
    return file_ref(root, relative)


def regular_info(name, size):
    relative_name(name)
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info


def verify_tar(root, archive_relative, manifest_relative, manifest_bytes, rows):
    expected = {row['path']: row for row in rows}
    require(len(expected) == len(rows), 'unique expected archive members')
    expected[manifest_relative] = dict(path=manifest_relative, bytes=len(manifest_bytes),
        sha256=hashlib.sha256(manifest_bytes).hexdigest())
    count = 0
    total = 0
    seen = set()
    last = None
    with tarfile.open(safe(root, archive_relative), 'r:gz') as archive:
        for info in archive:
            relative_name(info.name)
            require(info.type == tarfile.REGTYPE and info.name in expected and info.name not in seen and
                not info.linkname and info.pax_headers == {}, 'unexpected/duplicate/nonregular archive header')
            row = expected[info.name]
            require(type(info.size) is int and info.size == row['bytes'] and 0 <= info.size <= MAX_FILE,
                'archive member size differs')
            count += 1
            total += info.size
            require(count <= MAX_MEMBERS and total <= MAX_TOTAL, 'bounded archive contents')
            member = archive.extractfile(info)
            require(member is not None, 'regular member has no bytes')
            digest = hashlib.sha256()
            actual = 0
            with member:
                for block in iter(lambda: member.read(1024**2), b''):
                    actual += len(block)
                    require(actual <= row['bytes'], 'archive member grew')
                    digest.update(block)
            require(actual == row['bytes'] and digest.hexdigest() == row['sha256'], 'archive member byte drift')
            seen.add(info.name)
            last = info.name
    require(seen == set(expected) and last == manifest_relative, 'complete archive with terminal manifest required')
    return count


def pack(root, audit_relative, tag, modes):
    root = Path(root).resolve(strict=True)
    require(root == ROOT and not root.is_symlink(), 'fixed server project only')
    require(audit_relative == A, 'fixed server12 audit/delivery directory')
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', tag) is not None, 'append-only short delivery tag')
    require(type(modes) is tuple and modes in (MODES[:1], MODES[:2], MODES), 'ordered completed mode prefix')
    require(shutil.disk_usage(root).free >= 8 * 1024**3, 'preserve eight-GiB storage floor')
    ledger, ledger_bytes, ledger_ref = checked_ledger(root)
    selected = {}

    def add(relative):
        row = file_ref(root, relative)
        require(relative not in selected or selected[relative] == row, 'conflicting source path')
        selected[relative] = row
        require(len(selected) < MAX_MEMBERS and sum(r['bytes'] for r in selected.values()) <= MAX_TOTAL,
            'bounded complete evidence selection')

    # Freeze only the new normal directory and audit metadata, not weight or KV payloads.
    for base in (D, audit_relative):
        for relative in walk(root, base):
            add(relative)
    for relative in (BASE, 'experiments/prefix_io_v1/scripts/run_gpu_stage.py',
        audit_relative + '/PROJECT_GPU_AUTHORIZATION_AMENDMENT.json'):
        add(relative)
    lock, _, lock_ref = read(root, D + '/COMMON_SOURCE_LOCK.json')
    require(type(lock.get('files')) is list and 1 <= len(lock['files']) <= 10000,
        'existing normal common source lock required')
    locked = {row['path']: row for row in lock['files']}
    require(len(locked) == len(lock['files']), 'unique normal source closure')
    for relative, row in selected.items():
        if relative.startswith(D + '/') and relative.endswith('.py'):
            require(locked.get(relative) == row, 'frozen normal source changed: ' + relative)
    require(D + '/NORMAL_RUNTIME_BINDING.json' in selected and
        locked.get(D + '/NORMAL_RUNTIME_BINDING.json') == selected[D + '/NORMAL_RUNTIME_BINDING.json'],
        'actual frozen normal binding')
    cpu = []
    for relative in selected:
        if relative.endswith('/CPU_RESULT.json'):
            report, _, row = read(root, relative)
            if report.get('schema') == 'c5_normal_native_server_cpu_result_v1':
                require(report.get('status') == 'PASS_NORMAL_NATIVE_CPU_CHECKS' and
                    type(report.get('tests')) is int and report['tests'] > 0 and
                    report.get('passed') == report['tests'] and
                    all(type(report.get(k)) is int and report[k] == 0 for k in ('failed', 'errors', 'skipped')),
                    'actual complete CPU test evidence')
                require(relative.rsplit('/', 1)[0] + '/TEST_STDOUT.log' in selected, 'CPU test stdout required')
                cpu.append(row)
    require(cpu, 'server normal CPU test evidence required')
    completed = []
    for mode in modes:
        label = 'server12-c5-native-normal-' + mode + '01'
        run = 'experiments/prefix_io_v1/runs/' + label
        event, _, guard_ref = read(root, run + '/result.json')
        require(event.get('label') == label and event.get('session_drained') is True and
            event.get('session_members_after_cleanup') == [] and
            type(event.get('reservation_id')) is str and event['reservation_id'] and
            [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']] == [event],
            'actual unique drained original guard completion required')
        for relative in walk(root, run):
            add(relative)
        require(run + '/process.log' in selected, 'original process log required')
        require(run + '/details/p4-single-file-runtime-result.json' in selected, 'actual normal raw result required')
        for name in ('CONFIG_', 'AUTHORITY_', 'SCOPE_', 'HUMAN_GPU_GRANT_', 'EFFECTIVE_GPU_PERMISSION_',
            'LIVE_CONTEXT_', 'LAUNCH_INTENT_', 'LEDGER_BEFORE_'):
            require(D + '/' + name + mode + '.json' in selected, 'complete per-mode authority/launch evidence')
        for phase in ('BEFORE', 'LAUNCH', 'AFTER'):
            require(D + '/SOURCE_' + mode + '_' + phase + '.json' in selected, 'complete per-mode source proof')
        qualification = D + '/QUALIFICATION_' + mode + '.json'
        completed.append(dict(mode=mode, label=label, actual_guard_ref=guard_ref,
            existing_qualification_ref=selected.get(qualification), actual_exit=event.get('exit'),
            actual_child_exit=event.get('child_exit'), actual_elapsed_seconds=event.get('elapsed_seconds')))
    # A reference to the already persisted archive is sufficient; do not duplicate its raw payload.
    prior_result, _, prior_result_ref = read(root, A + '/GPU_ARCHIVE_RESULT.json')
    prior_archive_ref = file_ref(root, A + '/GPU_EVIDENCE.tar.gz')
    prior_manifest_ref = file_ref(root, A + '/GPU_DELIVERY_MANIFEST.json')
    require(prior_result.get('archive') == prior_archive_ref and prior_result.get('manifest') == prior_manifest_ref,
        'previous persisted calibration archive drift')
    stem = audit_relative + '/NORMAL_NORMAL_DELIVERY_' + tag
    snapshot_relative = stem + '_LEDGER_SNAPSHOT.json'
    manifest_relative = stem + '_MANIFEST.json'
    archive_relative = stem + '.tar.gz'
    result_relative = stem + '_RESULT.json'
    require(all(not safe(root, p).exists() for p in (snapshot_relative, manifest_relative, archive_relative, result_relative)),
        'new append-only delivery names required')
    snapshot_ref = put_bytes(root, snapshot_relative, ledger_bytes)
    add(snapshot_relative)
    rows = [selected[k] for k in sorted(selected)]
    payload_bytes = sum(row['bytes'] for row in rows)
    manifest = dict(schema=SCHEMA, files=rows, count=len(rows), payload_bytes=payload_bytes,
        source_lock_ref=lock_ref, original_ledger_snapshot_ref=snapshot_ref, original_ledger_ref_at_pack=ledger_ref,
        completed_modes=completed, actual_cpu_evidence_refs=cpu,
        prior_calibration_archive_ref=prior_archive_ref, prior_calibration_manifest_ref=prior_manifest_ref,
        prior_calibration_archive_result_ref=prior_result_ref,
        archive_scope='normal source/authority/guard/raw/source-proof/CPU evidence bytes; reports retain their actual qualification',
        excluded_payloads=['model_weights', 'SDK_binaries', 'runtime-cache', 'private-storage', 'fixture-tmp',
            'duplicate prior calibration raw archive'], excluded_payloads_preserved_on_server=True,
        manifest_self_excluded=True, GPU_operations_this_action=0, data_deletions_this_action=0,
        experiment_qualification_issued_this_action=False)
    manifest_bytes = document(manifest)
    require(payload_bytes + len(manifest_bytes) <= MAX_TOTAL and len(rows) + 1 <= MAX_MEMBERS,
        'bounded payload including terminal manifest')
    manifest_ref = put_bytes(root, manifest_relative, manifest_bytes)
    with safe(root, archive_relative).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for row in rows:
                    require(file_ref(root, row['path']) == row, 'source drift before archive insertion')
                    data = safe(root, row['path']).read_bytes()
                    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                        'actual archive member bytes')
                    archive.addfile(regular_info(row['path'], len(data)), io.BytesIO(data))
                archive.addfile(regular_info(manifest_relative, len(manifest_bytes)), io.BytesIO(manifest_bytes))
    count = verify_tar(root, archive_relative, manifest_relative, manifest_bytes, rows)
    for row in rows:
        require(file_ref(root, row['path']) == row, 'source drift after complete archive verification')
    require(checked_ledger(root)[1] == ledger_bytes, 'original GPU ledger changed during archive')
    require(shutil.disk_usage(root).free >= 8 * 1024**3, 'storage floor after archive')
    # Archive may legitimately be larger than one individual evidence file.
    archive_path = safe(root, archive_relative)
    archive_size = archive_path.stat().st_size
    require(0 < archive_size <= MAX_TOTAL, 'bounded compressed archive')
    with archive_path.open('rb') as stream:
        archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(status='PASS_NORMAL_NORMAL_DELIVERY_BYTE_ARCHIVE',
        archive=dict(path=archive_relative, bytes=archive_size, sha256=archive_sha), manifest=manifest_ref,
        ledger_snapshot=snapshot_ref, original_gpu_ledger_before=ledger_ref, original_gpu_ledger_after=ledger_ref,
        gpu_ledger_byte_unchanged=True, source_files=len(rows), archive_members=count,
        payload_bytes=payload_bytes, completed_modes=list(modes),
        GPU_operations_this_action=0, data_deletions_this_action=0, experiment_qualification_issued_this_action=False)
    put_bytes(root, result_relative, document(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--audit-relative', default=A)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--modes', required=True, help='off, or off,shadow, or off,shadow,on; completed arms only')
    args = parser.parse_args(argv)
    print(json.dumps(pack(args.project, args.audit_relative, args.tag, tuple(args.modes.split(','))), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
