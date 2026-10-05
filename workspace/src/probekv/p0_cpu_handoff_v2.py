"""Bounded, immutable CPU-validated source handoff; no GPU or network authority.

The ZIP carries raw files, not a fabricated commit. ``handoff.json`` is an
index, excluded from its own member digest. Its ZIP digest is returned to the
caller for an out-of-band deployment check. The verifier is stdlib-only and
does not extract, import packaged code, instantiate a model, or run commands.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile


PACKET = 'ProbeKV_Codex_First_Handoff'
PACKET_MEMBERS = (
    '00_阅读入口.md', '01_系统方案与实验任务书_v2.md',
    '02_发给Codex的执行指令_v2.md', '03_逻辑问题与修订清单.md',
    '04_实验契约草案.json', '05_测试与验收矩阵.md',
    '06_依据与查新登记.md', '07_本包校验记录.json', '08_首次交接说明.md',
    'reference/README.md', 'reference/policy_test_log.txt',
    'reference/provenance_reference.py', 'reference/test_provenance_reference.py',
    'reference/validate_package.py',
)
EVIDENCE_KEYS = ('cpu_report', 'worktree_files', 'test_results', 'commands', 'package_checksums')
_ROOT_FILES = frozenset(('.gitattributes', '.gitignore', 'README.md', 'pyproject.toml', 'setup.cfg', 'setup.py'))
_EXTENSIONS = {
    'src': {'.py'}, 'tests': {'.py', '.json', '.txt'},
    'scripts': {'.py', '.sh', '.ps1'}, 'configs': {'.json', '.yaml', '.yml'},
    'docs': {'.md', '.json'}, 'patches': {'.patch', '.json', '.md'},
    'requirements': {'.txt'}, '.github': {'.yml', '.yaml'},
}
_MAX_JSON_BYTES = 32 * 1024 * 1024
_SHA = re.compile(r'^[0-9a-f]{64}$')


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('unsafe portable member name')
    parts = name.split('/')
    if any(not p or p in ('.', '..') or p.endswith(('.', ' ')) for p in parts):
        raise ValueError('unsafe portable member path')
    if PurePosixPath(name).is_absolute():
        raise ValueError('absolute member path')
    for part in parts:
        stem = part.split('.')[0].upper()
        if stem in {'CON', 'PRN', 'AUX', 'NUL'} or re.fullmatch('(COM|LPT)[1-9]', stem):
            raise ValueError('reserved portable path component')
    return name


def _limits(max_archive_bytes, max_uncompressed_bytes, max_members):
    if any(type(x) is not int or x <= 0 for x in (max_archive_bytes, max_uncompressed_bytes, max_members)):
        raise ValueError('explicit positive byte/member budgets required')


def _regular(path, root=None):
    path = Path(path).absolute()
    root = Path(root).resolve() if root is not None else Path(path.anchor)
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('symlink/reparse member forbidden: ' + str(cursor))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('nonregular source member forbidden')
    return info.st_size


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _code_allowed(name):
    _safe_name(name)
    parts = name.split('/')
    lowered = [p.lower() for p in parts]
    if any(p in ('.git', '.ssh', '__pycache__', '.env', 'credentials', 'secrets')
           or p.endswith(('.pem', '.key', '.p12', '.bundle', '.safetensors', '.pt', '.bin'))
           for p in lowered):
        return False
    if name in _ROOT_FILES:
        return True
    if name in ('examples/manifest_input.example.jsonl', 'examples/README.md',
                'examples/hotpot_fixture.json', 'docs/NOVELTY_AUDIT_SOURCES.tsv'):
        return True
    return len(parts) > 1 and Path(name).suffix.lower() in _EXTENSIONS.get(parts[0], set())


def _git_inventory(workspace):
    def git(*args):
        return subprocess.check_output(['git'] + list(args), cwd=str(workspace))
    head = git('rev-parse', 'HEAD').decode().strip()
    tracked = set(git('ls-files', '-z').decode('utf-8').split('\0')) - {''}
    additions = set(git('ls-files', '--others', '--exclude-standard', '-z', '--',
                        'src', 'scripts', 'tests', 'docs').decode('utf-8').split('\0')) - {''}
    return head, tracked, tracked | additions


def _read_ref(ref, label):
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or
            not isinstance(ref.get('path'), str) or not _SHA.fullmatch(str(ref.get('sha256')))):
        raise ValueError('explicit evidence reference required: ' + label)
    path = Path(ref['path']).absolute()
    if _regular(path) > _MAX_JSON_BYTES or _file_hash(path) != ref['sha256']:
        raise ValueError('missing, oversized or changed evidence: ' + label)
    return json.loads(path.read_bytes()), path


def _excluded_head_member(workspace, head, name, expected, tracked):
    if name not in tracked:
        raise ValueError('untracked excluded file cannot be recovered from HEAD: ' + name)
    raw = subprocess.check_output(['git', 'show', head + ':' + name], cwd=str(workspace))
    blob_hash = _hash(raw)
    if blob_hash == expected:
        transform = 'identity'
    elif b'\r\n' not in raw and _hash(raw.replace(b'\n', b'\r\n')) == expected:
        transform = 'LF_TO_CRLF'
    else:
        raise ValueError('excluded file differs from HEAD beyond explicit checkout EOL: ' + name)
    return dict(sha256=expected, head_blob_sha256=blob_hash,
                reconstruction=transform, reason='outside portable source allowlist; not archived')


def build_cpu_handoff(workspace, *, evidence_refs, output_archive,
                      max_archive_bytes, max_uncompressed_bytes, max_members):
    """Create a new ZIP only from current, fully validated CPU evidence.

    All five evidence refs are required. The current Git inventory must exactly
    match the validation inventory (including untracked implementation files).
    Safe omitted tracked files must reconstruct from the unchanged HEAD; they
    remain explicit omissions rather than being passed off as transferred.
    """
    _limits(max_archive_bytes, max_uncompressed_bytes, max_members)
    root = Path(workspace).resolve()
    output = Path(output_archive).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('handoff output already exists')
    if not output.parent.is_dir():
        raise ValueError('explicit existing output directory required')
    if not isinstance(evidence_refs, dict) or set(evidence_refs) != set(EVIDENCE_KEYS):
        raise ValueError('exact five CPU evidence references required')
    loaded = {key: _read_ref(evidence_refs[key], key) for key in EVIDENCE_KEYS}
    files = loaded['worktree_files'][0]
    report = loaded['cpu_report'][0]
    if not isinstance(files, dict) or not isinstance(files.get('files'), dict):
        raise ValueError('validated file map missing')
    head, tracked, current = _git_inventory(root)
    present = set()
    for name in current:
        _safe_name(name)
        path = root / name
        if path.exists() or path.is_symlink():
            _regular(path, root)
            present.add(name)
    if head != files.get('base_commit') or present != set(files['files']):
        raise ValueError('stale validation: Git HEAD or current file inventory differs')
    if tracked - present:
        raise ValueError('deleted tracked files need an explicit deployment removal contract')
    for name, expected in files['files'].items():
        _safe_name(name)
        if not _SHA.fullmatch(str(expected)) or _file_hash(root / name) != expected:
            raise ValueError('stale validation: changed file ' + name)

    # Reuse the actual raw-test/log verifier rather than trusting passed=true.
    from .p0_stage_readiness_v2 import _verify_cpu_artifacts
    for stage in ('CONTROLLED_P0_E', 'CONTROLLED_P0_M'):
        _verify_cpu_artifacts(root, evidence_refs, stage)
    if report.get('gpu_actions_executed') != 0 or report.get('cuda_visible_devices') != '':
        raise ValueError('not a CPU-only validation run')
    reference = loaded['commands'][0].get('packet_reference')
    if not isinstance(reference, dict) or reference.get('returncode') != 0 or reference.get('passed') is not True:
        raise ValueError('packet reference check missing or failed')

    sources, entries, excluded = {}, {}, {}
    def add(name, path, expected, kind):
        _safe_name(name)
        if name.casefold() in {n.casefold() for n in entries}:
            raise ValueError('duplicate/case-colliding archive member')
        size = _regular(path, root if str(path).startswith(str(root) + os.sep) else None)
        if size > max_uncompressed_bytes:
            raise ValueError('member exceeds explicit uncompressed budget')
        if _file_hash(path) != expected:
            raise ValueError('member changed after validation: ' + name)
        entries[name] = dict(sha256=expected, bytes=size, kind=kind)
        sources[name] = Path(path)
    for name, expected in sorted(files['files'].items()):
        if _code_allowed(name):
            add('workspace/' + name, root / name, expected, 'validated_repository_file')
        else:
            excluded[name] = _excluded_head_member(root, head, name, expected, tracked)

    packet = root / PACKET
    checksum_path = packet / 'checksums.sha256'
    _regular(checksum_path, root)
    packet_checks = {}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        if name not in PACKET_MEMBERS or name in packet_checks or not _SHA.fullmatch(expected):
            raise ValueError('taskbook checksum member not explicitly allowed')
        packet_checks[name] = expected
    if set(packet_checks) != set(PACKET_MEMBERS):
        raise ValueError('taskbook checksum set incomplete')
    checks = loaded['package_checksums'][0]
    if (not isinstance(checks, list) or len(checks) != len(packet_checks) or
            {r.get('file') for r in checks} != set(packet_checks) or any(
                row.get('expected') != packet_checks[row['file']] or
                row.get('actual') != packet_checks[row['file']] or row.get('passed') is not True
                for row in checks)):
        raise ValueError('taskbook validation report does not match checksum file')
    for name, expected in sorted(packet_checks.items()):
        add('workspace/' + PACKET + '/' + name, packet / name, expected, 'supplied_taskbook')
    add('workspace/' + PACKET + '/checksums.sha256', checksum_path,
        _file_hash(checksum_path), 'supplied_taskbook_checksum_index')

    evidence_index = {}
    for key, (value, path) in loaded.items():
        name = 'evidence/' + key + '.json'
        add(name, path, evidence_refs[key]['sha256'], 'cpu_evidence')
        evidence_index[key] = name
    logs = {}
    for row in loaded['test_results'][0]:
        logs[row['evidence_path']] = row['evidence_hash']
    commands, command_path = loaded['commands']
    for row in commands['repository_checks'] + [commands['packet_reference']]:
        _safe_name(row['log'])
        path = command_path.parent / row['log']
        logs[str(path)] = row['log_sha256']
    log_index = {}
    for index, (path, expected) in enumerate(sorted(logs.items())):
        name = 'evidence/log-%03d.txt' % index
        add(name, Path(path), expected, 'cpu_raw_log')
        log_index[path] = name
    metadata = dict(kind='probekv_p0_cpu_handoff_v2', format_version=1,
        base_commit=head, branch=files.get('branch'), worktree_digest=files['digest'],
        dirty_worktree_snapshot=True, commit_is_not_snapshot=True,
        members=entries, excluded_worktree_members=excluded,
        evidence_index=evidence_index, original_log_to_member=log_index,
        tests_run=report['tests_run'], tests_skipped=report['skipped'],
        scope='Portable CPU-validated source snapshot, not model/GPU qualification',
        gpu_execution_allowed=False, automatic_rental_allowed=False,
        native_runtime_qualified=False, P1_E_execution_allowed=False,
        P1_M_execution_allowed=False, paper_evidence=False, locked_test_accessed=False)
    metadata_data = _json_bytes(metadata)
    total = sum(row['bytes'] for row in entries.values()) + len(metadata_data)
    if len(entries) + 1 > max_members or total > max_uncompressed_bytes:
        raise ValueError('handoff member/uncompressed budget exceeded')
    fd, temporary = tempfile.mkstemp(prefix='.cpu-handoff-', suffix='.partial', dir=str(output.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3; info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                digest, count = hashlib.sha256(), 0
                _regular(sources[name], root if name.startswith('workspace/') else None)
                with sources[name].open('rb') as source, archive.open(info, 'w') as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk); count += len(chunk); target.write(chunk)
                if count != entries[name]['bytes'] or digest.hexdigest() != entries[name]['sha256']:
                    raise ValueError('file changed during packaging: ' + name)
                if Path(temporary).stat().st_size > max_archive_bytes:
                    raise ValueError('compressed archive budget exceeded')
            info = zipfile.ZipInfo('handoff.json', date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3; info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, metadata_data)
        archive_digest = _file_hash(temporary)
        verify_cpu_handoff_archive(temporary, expected_sha256=archive_digest,
            max_archive_bytes=max_archive_bytes, max_uncompressed_bytes=max_uncompressed_bytes,
            max_members=max_members)
        if _git_inventory(root) != (head, tracked, current):
            raise ValueError('Git inventory changed during packaging')
        for name, expected in files['files'].items():
            _regular(root / name, root)
            if _file_hash(root / name) != expected:
                raise ValueError('worktree changed during packaging: ' + name)
        # Hard-link publication is atomic and never replaces an existing output.
        os.link(temporary, str(output))
        return dict(archive_path=str(output), archive_sha256=archive_digest,
                    archive_bytes=output.stat().st_size, metadata=metadata)
    finally:
        Path(temporary).unlink()


def verify_cpu_handoff_archive(path, *, expected_sha256, max_archive_bytes,
                               max_uncompressed_bytes, max_members):
    """Read/hash every bounded regular member, with no extraction or code import."""
    _limits(max_archive_bytes, max_uncompressed_bytes, max_members)
    path = Path(path)
    if not _SHA.fullmatch(str(expected_sha256)) or _regular(path) > max_archive_bytes:
        raise ValueError('invalid archive identity or byte budget')
    if _file_hash(path) != expected_sha256:
        raise ValueError('archive digest mismatch')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > max_members or sum(info.file_size for info in infos) > max_uncompressed_bytes:
            raise ValueError('archive expanded/member budget exceeded')
        names = set()
        for info in infos:
            name = _safe_name(info.filename)
            if name.casefold() in names:
                raise ValueError('duplicate/case-colliding archive name')
            names.add(name.casefold())
            mode = info.external_attr >> 16
            if not stat.S_ISREG(mode) or info.flag_bits & 1:
                raise ValueError('archive members must be unencrypted regular files')
        if 'handoff.json' not in names or archive.getinfo('handoff.json').file_size > _MAX_JSON_BYTES:
            raise ValueError('bounded handoff metadata missing')
        metadata = json.loads(archive.read('handoff.json'))
        if (not isinstance(metadata, dict) or metadata.get('kind') != 'probekv_p0_cpu_handoff_v2' or metadata.get('format_version') != 1
                or not re.fullmatch('[0-9a-f]{40}', str(metadata.get('base_commit')))
                or not _SHA.fullmatch(str(metadata.get('worktree_digest')))
                or metadata.get('dirty_worktree_snapshot') is not True
                or metadata.get('commit_is_not_snapshot') is not True
                or any(metadata.get(key) is not False for key in ('gpu_execution_allowed',
                    'automatic_rental_allowed', 'native_runtime_qualified', 'P1_E_execution_allowed',
                    'P1_M_execution_allowed', 'paper_evidence', 'locked_test_accessed'))):
            raise ValueError('handoff must remain CPU-only and unqualified')
        entries = metadata.get('members')
        if not isinstance(entries, dict) or set(entries) != {i.filename for i in infos} - {'handoff.json'}:
            raise ValueError('handoff membership does not exactly match archive')
        for name, row in entries.items():
            if name.startswith('workspace/'):
                relative = name[len('workspace/'):]
                allowed_packet = {PACKET + '/' + p for p in PACKET_MEMBERS + ('checksums.sha256',)}
                if not _code_allowed(relative) and relative not in allowed_packet:
                    raise ValueError('unexpected workspace member')
            elif not re.fullmatch(r'evidence/(?:(?:' + '|'.join(EVIDENCE_KEYS) + r')\.json|log-[0-9]{3}\.txt)', name):
                raise ValueError('unexpected evidence member')
            if (not isinstance(row, dict) or set(row) != {'sha256', 'bytes', 'kind'}
                    or type(row['bytes']) is not int or row['bytes'] != archive.getinfo(name).file_size
                    or not _SHA.fullmatch(str(row['sha256']))):
                raise ValueError('invalid member digest/size declaration')
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('archive member digest mismatch')
    return metadata
