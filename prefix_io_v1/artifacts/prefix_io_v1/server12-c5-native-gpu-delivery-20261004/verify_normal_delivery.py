"""Verify a downloaded normal archive and append a flat local byte-evidence copy.

No SSH, downloader, engine, GPU or archive extractall API is used. The caller
downloads the archive and its result with the existing authenticated transport.
Original server names stay in the archive and in the mapping report; local flat
names avoid Windows MAX_PATH. A byte archive never grants experiment authority.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import tarfile

A = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
SCHEMA = 'server12_normal_normal_delivery_manifest_v1'
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 100 * 1024**2
MAX_MEMBERS = 1024
MAX_TAR = MAX_TOTAL + 2 * 1024**2
EXCLUDED_PARTS = {'__pycache__', 'runtime-cache', 'private-storage', 'fixture-tmp', 'recorded_files'}


def require(value, reason):
    if not value:
        raise ValueError('NORMAL_LOCAL_DELIVERY_REJECTED: ' + reason)


def relative_name(value):
    require(type(value) is str and 0 < len(value) <= 240 and
        all(32 <= ord(c) < 127 for c in value) and ':' not in value and '\\' not in value and
        not value.startswith('/') and all(p not in ('', '.', '..') for p in value.split('/')),
        'bounded plain project-relative archive name')
    require(not set(value.split('/')).intersection(EXCLUDED_PARTS), 'cache/fixture payload refused')
    return value


def no_links(path, *, regular=False):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        require(not candidate.is_symlink(), 'local input/output symlink refused')
    if regular:
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'regular local input required')
    return path


def pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'duplicate JSON field')
        value[key] = item
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def reference(row, *, max_bytes=MAX_FILE):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact file reference')
    relative_name(row['path'])
    require(type(row['bytes']) is int and 0 <= row['bytes'] <= max_bytes and
        type(row['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None,
        'typed file bytes/SHA')
    return row


def actual_ref(path, server_path, *, max_bytes):
    path = no_links(path, regular=True)
    before = path.stat()
    require(0 <= before.st_size <= max_bytes, 'bounded local input')
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            size += len(block)
            require(size <= max_bytes, 'input grew above limit')
            digest.update(block)
    after = path.stat()
    require(size == before.st_size == after.st_size and
        (before.st_dev, before.st_ino, before.st_mtime_ns) ==
        (after.st_dev, after.st_ino, after.st_mtime_ns), 'local input changed while hashing')
    return dict(path=server_path, bytes=size, sha256=digest.hexdigest())


def plain_tar(archive_path):
    """Bound gzip expansion and read every literal USTAR header, including EOF.

    Tarfile's normal iterator hides extension headers. This reader rejects
    those headers rather than letting a long-name or PAX header overwrite a
    seemingly safe logical member name. The packer emits regular USTAR only.
    """
    parts = []
    size = 0
    with gzip.open(archive_path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            size += len(block)
            require(size <= MAX_TAR, 'decompressed tar exceeded bound')
            parts.append(block)
    raw = b''.join(parts)
    require(len(raw) % 512 == 0, 'tar block alignment')
    cursor = 0
    members = []
    seen = set()
    payload = 0
    while cursor < len(raw):
        header = raw[cursor:cursor+512]
        if header == b'\0' * 512:
            require(len(raw) - cursor >= 1024 and raw[cursor:] == b'\0' * (len(raw) - cursor),
                'two terminal zero blocks and no hidden trailing tar payload')
            break
        require(header[156:157] == b'0' and header[257:263] == b'ustar\0' and
            header[263:265] == b'00', 'literal regular USTAR header only')
        info = tarfile.TarInfo.frombuf(header, encoding='ascii', errors='strict')
        relative_name(info.name)
        require(info.type == tarfile.REGTYPE and not info.linkname and not info.pax_headers and
            info.name not in seen and type(info.size) is int and 0 <= info.size <= MAX_FILE,
            'duplicate/link/extension/oversized member refused')
        require(info.uid == 0 and info.gid == 0 and info.mode == 0o644 and info.mtime == 0,
            'packer regular evidence header contract')
        seen.add(info.name)
        payload += info.size
        require(len(seen) <= MAX_MEMBERS and payload <= MAX_TOTAL, 'bounded total member bytes/count')
        start = cursor + 512
        end = start + info.size
        padded_end = start + ((info.size + 511) // 512) * 512
        require(padded_end <= len(raw) and raw[end:padded_end] == b'\0' * (padded_end - end),
            'complete member and zero alignment padding')
        members.append((info.name, raw[start:end]))
        cursor = padded_end
    else:
        require(False, 'missing terminal tar blocks')
    return members


def local_name(server_path):
    prefix = hashlib.sha256(server_path.encode('utf-8')).hexdigest()[:12]
    basename = server_path.rsplit('/', 1)[-1]
    basename = re.sub(r'[^A-Za-z0-9._-]', '_', basename)[:64]
    require(basename, 'nonempty mapped basename')
    return prefix + '_' + basename


def verify(archive_path, result_path, output_dir, prior_archive=None):
    archive_path = no_links(archive_path, regular=True)
    result_path = no_links(result_path, regular=True)
    require(result_path.stat().st_size <= MAX_FILE, 'bounded result JSON')
    result_bytes = result_path.read_bytes()
    result = parse(result_bytes)
    require(type(result) is dict and result.get('status') == 'PASS_NORMAL_NORMAL_DELIVERY_BYTE_ARCHIVE' and
        result.get('gpu_ledger_byte_unchanged') is True and
        result.get('GPU_operations_this_action') == 0 and result.get('data_deletions_this_action') == 0 and
        result.get('experiment_qualification_issued_this_action') is False,
        'actual byte-archive result with no experiment qualification')
    archive_ref = reference(result['archive'], max_bytes=MAX_TOTAL)
    manifest_ref = reference(result['manifest'])
    require(archive_ref['path'].startswith(A + '/NORMAL_NORMAL_DELIVERY_') and
        archive_ref['path'].endswith('.tar.gz'), 'fixed new normal archive path')
    stem = archive_ref['path'].removesuffix('.tar.gz')
    require(manifest_ref['path'] == stem + '_MANIFEST.json', 'same archive/manifest identity')
    require(result.get('original_gpu_ledger_before') == result.get('original_gpu_ledger_after'),
        'original ledger exact unchanged reference')
    require(actual_ref(archive_path, archive_ref['path'], max_bytes=MAX_TOTAL) == archive_ref,
        'downloaded archive byte SHA/size')
    members = plain_tar(archive_path)
    require(members and members[-1][0] == manifest_ref['path'], 'terminal manifest required')
    manifest_bytes = members[-1][1]
    require(len(manifest_bytes) == manifest_ref['bytes'] and
        hashlib.sha256(manifest_bytes).hexdigest() == manifest_ref['sha256'], 'terminal manifest byte pin')
    manifest = parse(manifest_bytes)
    require(manifest.get('schema') == SCHEMA and manifest.get('manifest_self_excluded') is True and
        manifest.get('experiment_qualification_issued_this_action') is False and
        type(manifest.get('files')) is list and 1 <= len(manifest['files']) < MAX_MEMBERS,
        'actual bounded evidence manifest schema')
    refs = {}
    for row in manifest['files']:
        reference(row)
        require(row['path'] not in refs and row['path'] != manifest_ref['path'], 'unique manifest files and no self row')
        refs[row['path']] = row
    payload = sum(row['bytes'] for row in refs.values())
    require(type(manifest.get('count')) is int and manifest['count'] == len(refs) and
        type(manifest.get('payload_bytes')) is int and manifest['payload_bytes'] == payload and
        result.get('source_files') == len(refs) and result.get('payload_bytes') == payload and
        result.get('archive_members') == len(refs) + 1 and len(members) == len(refs) + 1,
        'actual count and byte totals agree')
    require({name for name, _ in members[:-1]} == set(refs), 'complete exact manifest member set')
    names = set()
    mappings = []
    for name, raw in members[:-1]:
        require(len(raw) == refs[name]['bytes'] and hashlib.sha256(raw).hexdigest() == refs[name]['sha256'],
            'member byte SHA/size: ' + name)
        mapped = local_name(name)
        require(mapped.lower() not in names, 'flat name collision')
        names.add(mapped.lower())
        mappings.append(dict(refs[name], local_name=mapped))
    snapshot = reference(result['ledger_snapshot'])
    require(refs.get(snapshot['path']) == snapshot and manifest.get('original_ledger_snapshot_ref') == snapshot,
        'archived actual idle ledger snapshot')
    ledger_bytes = next(raw for name, raw in members if name == snapshot['path'])
    ledger = parse(ledger_bytes)
    require(ledger.get('active_reservation') is None and
        dict(path=manifest['original_ledger_ref_at_pack']['path'], bytes=len(ledger_bytes),
            sha256=hashlib.sha256(ledger_bytes).hexdigest()) == result['original_gpu_ledger_before'],
        'actual snapshot equals the original idle ledger bytes')
    prior_status = 'SEPARATELY_PINNED_ARCHIVE_NOT_REHASHED_BY_THIS_CALL'
    if prior_archive is not None:
        prior_ref = reference(manifest['prior_calibration_archive_ref'], max_bytes=MAX_TOTAL)
        require(actual_ref(prior_archive, prior_ref['path'], max_bytes=MAX_TOTAL) == prior_ref,
            'separate persisted calibration archive byte pin')
        prior_status = 'ACTUAL_PRIOR_CALIBRATION_ARCHIVE_BYTES_REVERIFIED'
    require(actual_ref(archive_path, archive_ref['path'], max_bytes=MAX_TOTAL) == archive_ref and
        result_path.read_bytes() == result_bytes, 'downloaded input drift after verification')
    output_dir = no_links(output_dir)
    require(not output_dir.exists() and output_dir.parent.is_dir(), 'new append-only local output directory')
    output_dir.mkdir(exist_ok=False)
    recorded = output_dir / 'recorded_files'
    recorded.mkdir(exist_ok=False)
    for name, raw in members:
        target = recorded / local_name(name)
        with target.open('xb') as stream:
            stream.write(raw)
        require(target.read_bytes() == raw, 'actual mapped local file byte verification')
    for mapping in mappings:
        mapping['local_path'] = str(recorded / mapping.pop('local_name'))
    report = dict(status='PASS_SERVER12_NORMAL_DELIVERY_LOCAL_BYTE_VERIFICATION', archive=archive_ref,
        manifest=manifest_ref, ledger_snapshot=snapshot, source_files_verified=len(refs),
        archive_members=len(members), payload_bytes=payload, GPU_operations=0, data_deletions=0,
        experiment_qualification_issued=False, prior_calibration_archive_verification=prior_status,
        local_filename_mapping='original archive paths retained; recorded_files/pathhash12_basename64',
        files=mappings, manifest_local_path=str(recorded / local_name(manifest_ref['path'])))
    with (output_dir / 'LOCAL_NORMAL_DELIVERY_BYTE_VERIFICATION.json').open('xb') as stream:
        stream.write((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
    return {key: value for key, value in report.items() if key != 'files'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prior-calibration-archive', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.archive, args.result, args.output_dir, args.prior_calibration_archive), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
