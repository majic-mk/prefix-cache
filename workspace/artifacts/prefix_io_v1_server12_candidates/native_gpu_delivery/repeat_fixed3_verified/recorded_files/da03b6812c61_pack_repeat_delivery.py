"""Append a byte-verified archive of the fixed three off diagnostics.

This is a standard-library evidence packer, never an experiment verifier or
launcher. The caller first copies the final idle ledger to a new immutable
snapshot. Old calibration/normal archives remain separately pinned dependencies.
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
D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
A = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
LOCK_SHA = '4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
BASE = 'experiments/prefix_io_v1/configs/permissions.yaml'
SCOPE = 'server12_c5_normal_repeatability_v1'
SCHEMA = 'server12_fixed_repeat_delivery_manifest_v1'
MODEL_TARGET = 'models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444'
LABELS = tuple('server12-c5-native-repeat-off' + str(i + 1).zfill(2) for i in range(3))
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 100 * 1024**2
MAX_SOURCE_FILE = 64 * 1024**3
MAX_MEMBERS = 2048
MAX_TAR = MAX_TOTAL + 4 * 1024**2
SUFFIXES = {'.py', '.md', '.json', '.log', '.txt', '.yaml'}
EXCLUDED_PARTS = {'__pycache__', 'runtime-cache', 'private-storage', 'fixture-tmp',
    'recorded_files', 'server_evidence_root', 'normal_off01_verified', '.private-sdk',
    'private-sdk', '.private-ninja', 'private-ninja', 'ninja', 'torch_extensions',
    'compiled', 'compiledso', '.git'}
ARCHIVE_PREFIXES = ('GPU_EVIDENCE', 'NORMAL_NORMAL_DELIVERY_', 'REPEAT_REPEAT_DELIVERY_')


def require(value, reason):
    if not value:
        raise ValueError('REPEAT_DELIVERY_REJECTED: ' + reason)


def relative_name(value):
    require(type(value) is str and 0 < len(value) <= 240 and
        all(32 <= ord(c) < 127 for c in value) and ':' not in value and '\\' not in value and
        not value.startswith('/') and all(p not in ('', '.', '..') for p in value.split('/')),
        'bounded plain project-relative path')
    return value


def payload_name(value):
    relative_name(value)
    require(not set(value.split('/')).intersection(EXCLUDED_PARTS) and
        not value.startswith('models/'), 'excluded cache/model payload path')
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


def reference(row, max_bytes=MAX_FILE):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact file reference')
    relative_name(row['path'])
    require(type(row['bytes']) is int and 0 <= row['bytes'] <= max_bytes and
        type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
        'typed bounded file bytes/SHA')
    return row


def file_ref(root, relative, max_bytes=MAX_FILE):
    path = safe(root, relative)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= max_bytes,
        'bounded regular input file: ' + relative)
    size = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024**2), b''):
            size += len(block)
            require(size <= max_bytes, 'file grew above limit')
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


def checked_ledger(root):
    value, raw, row = read(root, LEDGER)
    require(type(value) is dict, 'original ledger document')
    seconds = value.get('gpu_wall_seconds')
    require(value.get('active_reservation') is None and
        type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 28800 and
        type(value.get('events')) is list and len(value['events']) <= 10000,
        'original ledger must be idle and within eight hours')
    return value, raw, row


def walk(root, relative, aliases, excluded):
    base = safe(root, relative)
    require(base.is_dir(), 'existing evidence directory: ' + relative)
    pending = [base]
    visited = 0
    while pending:
        path = pending.pop()
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            visited += 1
            require(visited <= 10000, 'bounded evidence traversal')
            child_relative = child.relative_to(root).as_posix()
            relative_name(child_relative)
            if child.name in EXCLUDED_PARTS or child.name.startswith('fixture-'):
                excluded.add(child_relative)
                continue
            if child.is_symlink():
                known = {'experiments/prefix_io_v1/runs/' + label +
                    '/details/Qwen/Qwen2.5-7B-Instruct' for label in LABELS}
                require(child_relative in known, 'unknown evidence symlink refused: ' + child_relative)
                target = safe(root, MODEL_TARGET)
                require(target.is_dir() and child.resolve(strict=True) == target.resolve(strict=True),
                    'known excluded model alias changed exact target')
                aliases[child_relative] = dict(path=child_relative, target=MODEL_TARGET,
                    symlink_text=os.readlink(child), target_verified=True, target_payload_read=False)
                continue
            mode = child.stat().st_mode
            if stat.S_ISDIR(mode):
                # Run model directories contain only the permitted symlink, not
                # a second physical model tree. Any physical weight is refused.
                if child.name == 'Qwen2.5-7B-Instruct':
                    require(False, 'model alias must remain the exact known symlink')
                pending.append(child)
            else:
                require(stat.S_ISREG(mode), 'nonregular evidence refused')
                if child.suffix not in SUFFIXES:
                    excluded.add(child_relative)
                    continue
                if child.name.startswith(ARCHIVE_PREFIXES) or child.name in (
                    'GPU_DELIVERY_MANIFEST.json', 'GPU_ARCHIVE_RESULT.json'):
                    continue
                if (path == safe(root, A) and re.match(r'^\d+_', child.name) and
                    int(child.name.split('_', 1)[0]) < 14):
                    continue
                yield payload_name(child_relative)


def verify_source_closure(root):
    value, _, row = read(root, LOCK)
    require(row['sha256'] == LOCK_SHA and type(value) is dict and
        value.get('schema') == 'c5_repeatability_common_source_lock_v1' and value.get('scope') == SCOPE and
        value.get('GPU_operations') == 0 and value.get('gpu_authority_issued') is False and
        value.get('normal_runtime_qualified') is False and type(value.get('files')) is list and
        len(value['files']) == 4816, 'actual already frozen CPU diagnostic source lock')
    refs = {}
    for source in value['files']:
        reference(source, MAX_SOURCE_FILE)
        require(source['path'] not in refs and source['path'] not in (LEDGER, LOCK),
            'unique immutable source closure; mutable ledger excluded')
        require(file_ref(root, source['path'], MAX_SOURCE_FILE) == source,
            'complete source byte drift: ' + source['path'])
        refs[source['path']] = source
    return value, row, refs


def put_bytes(root, relative, raw):
    path = safe(root, relative)
    require(path.parent.is_dir(), 'existing output directory required')
    with path.open('xb') as stream:
        stream.write(raw)
    return file_ref(root, relative)


def regular_info(name, size):
    payload_name(name)
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info


def verify_tar(root, archive_relative, manifest_relative, manifest_bytes, rows):
    """Inspect every literal regular USTAR header, including hidden extensions."""
    expected = {row['path']: row for row in rows}
    require(len(expected) == len(rows), 'unique expected archive members')
    expected[manifest_relative] = dict(path=manifest_relative, bytes=len(manifest_bytes),
        sha256=hashlib.sha256(manifest_bytes).hexdigest())
    blocks = []
    expanded = 0
    with gzip.open(safe(root, archive_relative), 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            expanded += len(block)
            require(expanded <= MAX_TAR, 'bounded tar expansion')
            blocks.append(block)
    raw = b''.join(blocks)
    require(len(raw) % 512 == 0, 'tar block alignment')
    seen = set()
    cursor = 0
    total = 0
    last = None
    while cursor < len(raw):
        header = raw[cursor:cursor + 512]
        if header == b'\0' * 512:
            require(len(raw) - cursor >= 1024 and raw[cursor:] == b'\0' * (len(raw) - cursor),
                'two terminal zero blocks and no hidden trailing tar payload')
            break
        require(header[156:157] == b'0' and header[257:263] == b'ustar\0' and
            header[263:265] == b'00', 'literal regular USTAR header only')
        info = tarfile.TarInfo.frombuf(header, encoding='ascii', errors='strict')
        payload_name(info.name)
        require(info.type == tarfile.REGTYPE and not info.linkname and info.pax_headers == {} and
            info.name in expected and info.name not in seen and info.uid == info.gid == 0 and
            info.mode == 0o644 and info.mtime == 0, 'unexpected/duplicate/nonregular archive header')
        row = expected[info.name]
        require(type(info.size) is int and info.size == row['bytes'] and 0 <= info.size <= MAX_FILE,
            'archive member size differs')
        start = cursor + 512
        end = start + info.size
        padded_end = start + ((info.size + 511) // 512) * 512
        require(padded_end <= len(raw) and raw[end:padded_end] == b'\0' * (padded_end - end),
            'complete member and zero alignment padding')
        require(hashlib.sha256(raw[start:end]).hexdigest() == row['sha256'], 'archive member byte drift')
        total += info.size
        require(total <= MAX_TOTAL and len(seen) < MAX_MEMBERS, 'bounded archive contents')
        seen.add(info.name)
        last = info.name
        cursor = padded_end
    else:
        require(False, 'missing terminal tar blocks')
    require(seen == set(expected) and last == manifest_relative, 'complete archive with terminal manifest')
    return len(seen)


def checked_dependencies(root):
    results = (
        (A + '/GPU_ARCHIVE_RESULT.json', A + '/GPU_EVIDENCE.tar.gz', A + '/GPU_DELIVERY_MANIFEST.json',
         'PASS_COMPLETED_GPU_EVIDENCE_ARCHIVE'),
        (A + '/NORMAL_NORMAL_DELIVERY_off01-final_RESULT.json',
         A + '/NORMAL_NORMAL_DELIVERY_off01-final.tar.gz',
         A + '/NORMAL_NORMAL_DELIVERY_off01-final_MANIFEST.json', 'PASS_NORMAL_NORMAL_DELIVERY_BYTE_ARCHIVE'))
    dependencies = []
    for relative, expected_archive, expected_manifest, expected_status in results:
        value, _, result_ref = read(root, relative)
        require(type(value) is dict and value.get('status') == expected_status, 'existing prior archive result')
        archive_ref = reference(value['archive'], MAX_TOTAL)
        manifest_ref = reference(value['manifest'])
        require(archive_ref['path'] == expected_archive and manifest_ref['path'] == expected_manifest,
            'known separately persisted prior archive')
        require(file_ref(root, archive_ref['path'], MAX_TOTAL) == archive_ref and
            file_ref(root, manifest_ref['path']) == manifest_ref, 'prior archive dependency byte drift')
        dependencies.append(dict(result_ref=result_ref, archive_ref=archive_ref, manifest_ref=manifest_ref,
            archive_embedded=False))
    return dependencies


def pack(root, audit_relative, tag, ledger_snapshot_relative):
    supplied_root = Path(root).absolute()
    require(supplied_root == ROOT and not supplied_root.is_symlink(), 'literal fixed server project only')
    root = supplied_root.resolve(strict=True)
    require(root == ROOT and not any(path.is_symlink() for path in root.parents), 'fixed non-symlink root ancestry')
    require(audit_relative == A and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', tag) is not None,
        'fixed audit directory and new short append-only tag')
    require(ledger_snapshot_relative.startswith((D + '/', A + '/')) and
        ledger_snapshot_relative.endswith('.json') and ledger_snapshot_relative != LEDGER,
        'caller-created final immutable ledger snapshot required')
    require(shutil.disk_usage(root).free >= 8 * 1024**3, 'preserve eight-GiB storage floor')
    ledger, ledger_bytes, ledger_ref = checked_ledger(root)
    snapshot, snapshot_bytes, snapshot_ref = read(root, ledger_snapshot_relative)
    require(snapshot_bytes == ledger_bytes and snapshot == ledger, 'final snapshot must equal current idle ledger bytes')
    _, lock_ref, locked = verify_source_closure(root)
    selected = {}
    aliases = {}
    excluded = set()

    def add(relative):
        payload_name(relative)
        row = file_ref(root, relative)
        require(relative not in selected or selected[relative] == row, 'conflicting source path')
        selected[relative] = row
        require(len(selected) < MAX_MEMBERS and sum(r['bytes'] for r in selected.values()) <= MAX_TOTAL,
            'bounded complete evidence selection')

    for base in (D, A):
        for relative in walk(root, base, aliases, excluded):
            add(relative)
    for relative in (BASE, 'experiments/prefix_io_v1/scripts/run_gpu_stage.py',
        A + '/PROJECT_GPU_AUTHORIZATION_AMENDMENT.json', ledger_snapshot_relative):
        add(relative)
    for relative, row in selected.items():
        if relative.startswith(D + '/') and relative.endswith('.py'):
            require(locked.get(relative) == row, 'frozen diagnostic source changed or untracked: ' + relative)
    require(LOCK in selected and D + '/PROTOCOL.json' in selected and
        D + '/PREREGISTRATION.json' in selected and D + '/PREREGISTRATION_LEDGER_SNAPSHOT.json' in selected,
        'actual common/protocol/preregistration evidence')
    protocol, _, _ = read(root, D + '/PROTOCOL.json')
    require(protocol.get('scope') == SCOPE and protocol.get('repetitions') == 3 and
        protocol.get('cost_upper_ns') == 16238752 and protocol.get('step_budget_ns') == 13171328,
        'original fixed three-slot thresholds')
    completed = []
    missing = []
    for index, label in enumerate(LABELS):
        token = 'off' + str(index + 1).zfill(2)
        run = 'experiments/prefix_io_v1/runs/' + label
        guard_relative = run + '/result.json'
        if not safe(root, run).exists():
            missing.append(dict(diagnostic_index=index, label=label, status='NO_ACTUAL_RUN_DIRECTORY'))
            continue
        require(safe(root, guard_relative).is_file(), 'existing run requires actual original guard completion')
        event, _, guard_ref = read(root, guard_relative)
        require(type(event) is dict and event.get('label') == label and event.get('session_drained') is True and
            event.get('session_members_after_cleanup') == [] and type(event.get('reservation_id')) is str and
            event['reservation_id'] and
            [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']] == [event],
            'actual unique drained original guard completion')
        for relative in walk(root, run, aliases, excluded):
            add(relative)
        require(run + '/process.log' in selected, 'original process log')
        for name in ('CONFIG_', 'AUTHORITY_', 'SCOPE_', 'HUMAN_GPU_GRANT_', 'EFFECTIVE_GPU_PERMISSION_',
            'LIVE_CONTEXT_', 'LAUNCH_INTENT_', 'LEDGER_BEFORE_'):
            require(D + '/' + name + token + '.json' in selected, 'actual per-slot authority/launch evidence')
        for phase in ('BEFORE', 'LAUNCH'):
            require(D + '/SOURCE_' + token + '_' + phase + '.json' in selected, 'actual per-slot source evidence')
        actual_report = selected.get(D + '/DIAGNOSTIC_RESULT_' + token + '.json')
        if actual_report is not None:
            report, _, _ = read(root, actual_report['path'])
            require(report.get('diagnostic_index') == index and report.get('scope') == SCOPE and
                report.get('normal_qualification_passed') is False and report.get('permits_next_mode') is None,
                'diagnostic report must retain its limited qualification')
            require(run + '/details/p4-single-file-runtime-result.json' in selected and
                D + '/SOURCE_' + token + '_AFTER.json' in selected, 'report requires actual raw and after proof')
        completed.append(dict(diagnostic_index=index, label=label, actual_guard_ref=guard_ref,
            actual_raw_ref=selected.get(run + '/details/p4-single-file-runtime-result.json'),
            actual_diagnostic_report_ref=actual_report, actual_exit=event.get('exit'),
            actual_child_exit=event.get('child_exit'), actual_elapsed_seconds=event.get('elapsed_seconds')))
    summary_path = D + '/REPEATABILITY_SUMMARY.json'
    stops = [row for path, row in selected.items() if path.startswith(D + '/REPEATABILITY_HARD_STOP_')]
    require(completed and (summary_path in selected or stops), 'actual terminal summary or hardstop evidence required')
    if summary_path in selected:
        summary, _, _ = read(root, summary_path)
        require(summary.get('scope') == SCOPE and summary.get('normal_qualification_passed') is False and
            summary.get('permits_next_mode') is None and summary.get('all_three_slots_retained') is True,
            'actual summary retains all fixed slots and no qualification promotion')
    dependencies = checked_dependencies(root)
    stem = A + '/REPEAT_REPEAT_DELIVERY_' + tag
    manifest_relative = stem + '_MANIFEST.json'
    archive_relative = stem + '.tar.gz'
    result_relative = stem + '_RESULT.json'
    require(all(not safe(root, path).exists() for path in (manifest_relative, archive_relative, result_relative)),
        'new append-only archive/result names')
    rows = [selected[key] for key in sorted(selected)]
    payload_bytes = sum(row['bytes'] for row in rows)
    manifest = dict(schema=SCHEMA, files=rows, count=len(rows), payload_bytes=payload_bytes,
        source_lock_ref=lock_ref, source_files_fully_verified=len(locked),
        original_ledger_snapshot_ref=snapshot_ref, original_ledger_ref_at_pack=ledger_ref,
        completed_diagnostic_slots=completed, missing_diagnostic_slots=missing,
        actual_summary_ref=selected.get(summary_path), actual_hardstop_refs=stops,
        immutable_dependency_archives=dependencies,
        model_aliases_verified_without_following_or_payload_read=list(aliases.values()),
        excluded_payload_paths_observed=sorted(excluded),
        payload_metadata_retained_in_original_raw_and_source_lock=True,
        excluded_payloads=['model_weights', 'SDK_binaries', 'native_shared_libraries',
            'runtime-cache', 'private-storage', '.private-sdk', 'ninja', 'compiledso',
            'fixture-tmp', 'duplicate prior calibration and normal archives'],
        excluded_payloads_preserved_on_server=True, manifest_self_excluded=True,
        archive_scope='actual diagnostic/source/grant/guard/raw/proof/CPU log bytes; no new qualification',
        GPU_operations_this_action=0, data_deletions_this_action=0,
        experiment_qualification_issued_this_action=False)
    manifest_bytes = document(manifest)
    require(len(manifest_bytes) <= MAX_FILE and payload_bytes + len(manifest_bytes) <= MAX_TOTAL and
        len(rows) + 1 <= MAX_MEMBERS, 'bounded archive including terminal manifest')
    manifest_ref = put_bytes(root, manifest_relative, manifest_bytes)
    with safe(root, archive_relative).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                for row in rows:
                    require(file_ref(root, row['path']) == row, 'evidence drift before archive insertion')
                    data = safe(root, row['path']).read_bytes()
                    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                        'actual archive member bytes')
                    archive.addfile(regular_info(row['path'], len(data)), io.BytesIO(data))
                archive.addfile(regular_info(manifest_relative, len(manifest_bytes)), io.BytesIO(manifest_bytes))
    count = verify_tar(root, archive_relative, manifest_relative, manifest_bytes, rows)
    for row in rows:
        require(file_ref(root, row['path']) == row, 'evidence drift after archive verification')
    require(verify_source_closure(root)[2] == locked, 'complete source closure after archive verification')
    for dependency in dependencies:
        require(file_ref(root, dependency['archive_ref']['path'], MAX_TOTAL) == dependency['archive_ref'] and
            file_ref(root, dependency['manifest_ref']['path']) == dependency['manifest_ref'] and
            file_ref(root, dependency['result_ref']['path']) == dependency['result_ref'],
            'prior archive dependency drift after pack')
    require(checked_ledger(root)[1] == ledger_bytes, 'original GPU ledger changed during archive')
    require(shutil.disk_usage(root).free >= 8 * 1024**3, 'storage floor after archive')
    archive_ref = file_ref(root, archive_relative, MAX_TOTAL)
    result = dict(status='PASS_FIXED_REPEAT_DELIVERY_BYTE_ARCHIVE', archive=archive_ref, manifest=manifest_ref,
        ledger_snapshot=snapshot_ref, original_gpu_ledger_before=ledger_ref, original_gpu_ledger_after=ledger_ref,
        gpu_ledger_byte_unchanged=True, source_files=len(rows), archive_members=count,
        complete_runtime_source_files_verified_before_and_after=len(locked), payload_bytes=payload_bytes,
        completed_diagnostic_indices=[row['diagnostic_index'] for row in completed],
        missing_diagnostic_indices=[row['diagnostic_index'] for row in missing],
        immutable_dependency_archives=dependencies, GPU_operations_this_action=0,
        data_deletions_this_action=0, experiment_qualification_issued_this_action=False)
    put_bytes(root, result_relative, document(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--audit-relative', default=A)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--ledger-snapshot-relative', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(pack(args.project, args.audit_relative, args.tag, args.ledger_snapshot_relative), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
