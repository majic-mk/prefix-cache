"""Append explicit public CPU input leaves to the immutable actual V13 closure.

Uses only the standard library. It neither imports a model/backend nor issues
GPU authority. It streams every inherited byte once; additions are an explicit
manifest, never a directory scan. Existing files are never written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

AUDIT = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
SELF = AUDIT + '/source_freeze/append_public_source_lock_v14.py'
ANCESTOR = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005/PRERENT_SOURCE_LOCK_V13.json'
ANCESTOR_SHA = '1d1aef9591a8c319668260d56b1cbe55b8775f0f18b39ea89a36ca83952070b1'
ANCESTOR_COUNT = 4990
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'
MANIFEST_SCHEMA = 'bounded_public_source_append_manifest_v1'
CHUNK = 4 * 1024**2


def require(ok, why):
    if not ok:
        raise ValueError('V14_CPU_SOURCE_REJECTED: ' + why)


def safe(root, relative, *, existing=True):
    require(type(relative) is str and relative and not relative.startswith('/') and
            ':' not in relative and '\\' not in relative and '\0' not in relative and
            all(part not in ('', '.', '..') for part in relative.split('/')),
            'contained project-relative POSIX path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink source/output: ' + relative)
    require(path.resolve(strict=existing).is_relative_to(root), 'project containment')
    return path


def stat_identity(info):
    return (info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def capture_file_ref(root, relative):
    path = safe(root, relative)
    before = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(before.st_mode), 'regular file: ' + relative)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        # Windows path-stat ctime is creation time while fd-stat can report
        # the modification time. Cross-check identity/mtime, then compare
        # each API's complete stability tuple against its own later sample.
        require(stat_identity(before)[:5] == stat_identity(opened)[:5],
                'file replaced while opening: ' + relative)
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            digest.update(chunk)
            count += len(chunk)
        last_fd = os.fstat(stream.fileno())
    after = safe(root, relative).stat(follow_symlinks=False)
    require(stat_identity(before) == stat_identity(after) and
            stat_identity(opened) == stat_identity(last_fd) and
            count == before.st_size, 'source drift while hashing: ' + relative)
    return (dict(path=relative, bytes=count, sha256=digest.hexdigest()), stat_identity(after))


def file_ref(root, relative):
    return capture_file_ref(root, relative)[0]


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))


def validate_row(row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact source row')
    require(type(row['bytes']) is int and row['bytes'] >= 0, 'source byte count')
    sha = row['sha256']
    require(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
            'source SHA-256')
    require(type(row['path']) is str, 'source relative path string')


def audit_path(root, relative, *, output=False):
    require(type(relative) is str and relative.startswith(AUDIT + '/'), 'new audit path only')
    path = safe(root, relative, existing=not output)
    if output:
        require(path.suffix == '.json' and path.parent.is_dir() and not path.exists(),
                'fresh JSON output with existing parent')
    return path


def serialized(document):
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def freeze(root, *, addition_manifest, source_lock_output, proof_output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation')
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), 'existing project root')
    require(Path(__file__).resolve(strict=True) == safe(root, SELF), 'executing the actual bound append freezer')
    started = time.monotonic()
    manifest_path = audit_path(root, addition_manifest)
    lock_path = audit_path(root, source_lock_output, output=True)
    proof_path = audit_path(root, proof_output, output=True)
    require(source_lock_output != proof_output, 'distinct new lock and proof paths')
    manifest_ref = file_ref(root, addition_manifest)
    manifest_raw = manifest_path.read_bytes()
    require(len(manifest_raw) == manifest_ref['bytes'] <= 1024**2 and
            hashlib.sha256(manifest_raw).hexdigest() == manifest_ref['sha256'], 'stable bounded addition manifest')
    manifest = parse_json(manifest_raw)
    require(type(manifest) is dict and manifest.get('schema') == MANIFEST_SCHEMA,
            'explicit addition manifest schema')
    additions = manifest.get('files')
    require(type(additions) is list and 1 <= len(additions) <= 1024 and
            all(type(name) is str for name in additions), 'explicit nonempty relative source string list')
    require(len(set(additions)) == len(additions), 'unique addition paths; no glob expansion')
    require(not ({source_lock_output, proof_output} & set(additions)) and
            source_lock_output not in (addition_manifest, SELF, ANCESTOR) and
            proof_output not in (addition_manifest, SELF, ANCESTOR), 'outputs cannot be input leaves')
    for name in additions:
        require(not name.endswith(('.partial', '.part', '.tmp')), 'unfinished partial source excluded')
        safe(root, name)
    ledger_path = safe(root, LEDGER)
    ledger_raw = ledger_path.read_bytes()
    ledger = parse_json(ledger_raw)
    require(hashlib.sha256(ledger_raw).hexdigest() == LEDGER_SHA and type(ledger) is dict and
            'active_reservation' in ledger and ledger['active_reservation'] is None, 'unchanged idle original GPU ledger')
    ancestor_path = safe(root, ANCESTOR)
    ancestor_ref = file_ref(root, ANCESTOR)
    require(ancestor_ref['sha256'] == ANCESTOR_SHA, 'exact actual V13 ancestor SHA')
    ancestor_raw = ancestor_path.read_bytes()
    require(len(ancestor_raw) == ancestor_ref['bytes'] <= 16 * 1024**2 and
            hashlib.sha256(ancestor_raw).hexdigest() == ANCESTOR_SHA, 'stable V13 ancestor bytes')
    ancestor = parse_json(ancestor_raw)
    require(type(ancestor) is dict and ancestor.get('schema') == 'strong_trace_source_lock_v1' and
            type(ancestor.get('files')) is list and len(ancestor['files']) == ANCESTOR_COUNT,
            'all 4990 actual V13 source/model/SDK leaves')
    rows = {}
    snapshots = {}

    def append(name, expected=None):
        actual, snapshot = capture_file_ref(root, name)
        require(expected is None or actual == expected, 'inherited actual byte drift: ' + name)
        if name in rows:
            require(rows[name] == actual, 'strict overlapping source byte mismatch: ' + name)
        rows[name] = actual
        snapshots[name] = snapshot

    for expected in ancestor['files']:
        validate_row(expected)
        require(expected['path'] not in rows, 'duplicate inherited source path')
        append(expected['path'], expected)
    append(ANCESTOR, ancestor_ref)
    append(addition_manifest, manifest_ref)
    append(SELF)
    for name in additions:
        append(name)
    for expected in ancestor['files']:
        require(rows.get(expected['path']) == expected, 'strict complete V13 inheritance')
    require(len(rows) <= 8192, 'existing bounded source-lock capacity')
    # Detect later writes during the one full-byte pass without hashing the
    # entire inherited model/SDK a second time. Atime changes are irrelevant.
    for name, snapshot in snapshots.items():
        require(stat_identity(safe(root, name).stat(follow_symlinks=False)) == snapshot,
                'source metadata drift after hashing: ' + name)
    require(ancestor_path.read_bytes() == ancestor_raw and manifest_path.read_bytes() == manifest_raw,
            'ancestor/manifest unchanged through append')
    require(ledger_path.read_bytes() == ledger_raw, 'GPU ledger bytes unchanged before outputs')
    audit_path(root, source_lock_output, output=True)
    audit_path(root, proof_output, output=True)
    lock = dict(schema='strong_trace_source_lock_v1', files=[rows[name] for name in sorted(rows)],
                ancestry_ref=ancestor_ref, addition_manifest_ref=manifest_ref, append_freezer_ref=rows[SELF],
                GPU_launch_allowed=False, production_qualified=False, strategy_effect_verified=False,
                model_and_SDK_loaded=False, observation_scope='this_CPU_public_source_append_freeze_only')
    lock_bytes = serialized(lock)
    lock_ref = dict(path=source_lock_output, bytes=len(lock_bytes),
                    sha256=hashlib.sha256(lock_bytes).hexdigest())
    proof = dict(schema='strong_trace_source_proof_v1', status='PASS_FULL_CPU_SOURCE_BYTES',
                 source_lock_ref=lock_ref, ancestry_ref=ancestor_ref, addition_manifest_ref=manifest_ref,
                 append_freezer_ref=rows[SELF], source_count=len(rows), inherited_source_count=ANCESTOR_COUNT,
                 strict_all_ancestor_rows_preserved=True, explicit_addition_count=len(additions),
                 total_bytes_verified=sum(row['bytes'] for row in rows.values()),
                 elapsed_seconds=time.monotonic() - started, original_idle_ledger_sha256=LEDGER_SHA,
                 actual_gpu_runs=0, source_ancestry_is_not_GPU_authority=True,
                 GPU_qualification=False, effect_verified=False, natural_family_receipt_created=False)
    with lock_path.open('xb') as stream:
        stream.write(lock_bytes)
    require(file_ref(root, source_lock_output) == lock_ref, 'actual written lock bytes')
    require(ledger_path.read_bytes() == ledger_raw, 'GPU ledger bytes unchanged after lock append')
    with proof_path.open('xb') as stream:
        stream.write(serialized(proof))
    require(ledger_path.read_bytes() == ledger_raw, 'GPU ledger bytes unchanged after proof append')
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--addition-manifest', required=True)
    parser.add_argument('--source-lock-output', required=True)
    parser.add_argument('--proof-output', required=True)
    args = parser.parse_args(argv)
    result = freeze(args.project_root, addition_manifest=args.addition_manifest,
                    source_lock_output=args.source_lock_output, proof_output=args.proof_output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
