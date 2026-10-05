"""Exact approved CPU fixture cleanup; stdlib only, no recursive deletion/GPU imports.

preflight creates reviewable evidence without deleting. execute repeats all gates,
unlinks only frozen objects with directory-fd and inode checks, then verifies all
protected SHA references and the entire noncandidate project/AUX inventory.
"""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import os
import stat
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
OUT = ROOT / 'artifacts/prefix_io_v1/server08-storage-audit-20261001'
MANIFEST = OUT / 'CPU_TEMP_CLEANUP_PROPOSAL.json'
MANIFEST_SHA = 'b0c347fc4fc50384c088b8539b63ac09d77446a9563f33fba83f7844514c410e'
PLAN_SHA = '41611b8f173c78d66579086734abaf1d4166606e4d62ef291d1ab6d6a22e4e12'
PREFLIGHT_INVENTORY_SHA = '72a9aa97a255fe61bba09b924c0bf551fb08ad5efa5643092f58c5c91f376b89'
PLAN = None
ALLOWED_EXTERNAL_PROTECTED = {
    '/root/prefix-io-v1-validation/runs/server07-p3-c2-fullref-01/details/result.json',
}
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1'
PERMISSIONS = 'experiments/prefix_io_v1/configs/permissions.yaml'
PERMISSIONS_SHA = '795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50'
AUTH_TEXT = '确定，但是要保证后续后续需要的实验数据和代码不被删除'
FIELDS = ('device', 'inode', 'mode', 'nlink', 'bytes', 'allocated_bytes', 'mtime_ns')
ANCHORS = {}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_new(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def meta(s):
    return dict(device=s.st_dev, inode=s.st_ino, mode=s.st_mode,
                nlink=s.st_nlink, bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                mtime_ns=s.st_mtime_ns)


def identity(s):
    return (s.st_dev, s.st_ino, s.st_mode)


def exact(s, row, expected_nlink=None):
    m = meta(s)
    expected = {k: row[k] for k in FIELDS}
    if expected_nlink is not None:
        expected['nlink'] = expected_nlink
    check(m == expected, 'metadata differs: ' + row['path'])


def stable_sha(p, expected=None):
    before = p.lstat()
    check(stat.S_ISREG(before.st_mode), 'hash requires canonical regular file: ' + str(p))
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        exact(os.fstat(fd), dict(path=str(p), **meta(before)))
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
        exact(os.fstat(fd), dict(path=str(p), **meta(before)))
        exact(p.lstat(), dict(path=str(p), **meta(before)))
    finally:
        os.close(fd)
    digest = h.hexdigest()
    if expected is not None:
        check(digest == expected, 'SHA differs: ' + str(p))
    return digest


def absolute(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def inside(p, roots):
    return any(p == r or p.is_relative_to(r) for r in roots)


def canonical_directory(p):
    check(p.is_absolute(), 'directory must be absolute')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        current = Path('/')
        for part in p.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
            current = current / part
            ident = identity(os.fstat(fd))
            old = ANCHORS.setdefault(str(current), ident)
            check(ident == old, 'directory anchor differs: ' + str(current))
        return fd
    except BaseException:
        os.close(fd)
        raise


def fresh_manifest():
    global PLAN
    check(Path.cwd() == ROOT and ROOT.resolve() == ROOT, 'wrong canonical project cwd')
    stable_sha(MANIFEST, MANIFEST_SHA)
    m = json.loads(MANIFEST.read_text())
    check(m['ROOT'] == str(ROOT), 'manifest ROOT differs')
    check(len(m['candidate_roots']) == 78, 'root count differs')
    check(m['regular_paths'] == 8423 and m['symlinks'] == 822 and m['synthetic_FIFOs'] == 6,
          'object counts differ')
    check(m['conservative_regular_allocated_bytes'] == 3221516288, 'allocation differs')
    roots = []
    for row in m['candidate_roots']:
        rel = Path(row['path'])
        check(not rel.is_absolute() and '..' not in rel.parts, 'unsafe root')
        roots.append(ROOT / rel)
    check(len(set(roots)) == 78, 'duplicate roots')
    for a in roots:
        check(not any(a != b and a.is_relative_to(b) for b in roots), 'overlapping roots')
    rows = {r['path']: r for r in m['files'] + m['directories']}
    check(len(rows) == len(m['files']) + len(m['directories']), 'duplicate paths')
    for key in rows:
        p = Path(key)
        check(not p.is_absolute() and '..' not in p.parts and inside(ROOT / p, roots),
              'object outside approved roots: ' + key)
    plan_path = OUT / 'RESTRICTED_CPU_CLEANUP_EXECUTION_PLAN.json'
    stable_sha(plan_path, PLAN_SHA)
    PLAN = json.loads(plan_path.read_text())
    check(PLAN['approved_manifest_sha256'] == MANIFEST_SHA and PLAN['no_new_paths'], 'plan scope differs')
    skipped = {row['path'] for row in PLAN['skipped_files']}
    check(len(skipped) == 6, 'skip count differs')
    for key in skipped:
        row = rows.get(key)
        check(row and row.get('type') == 'regular' and row['bytes'] == 0 and row['nlink'] == 1
              and row['allocated_bytes'] == 4096
              and row['sha256'] == hashlib.sha256(b'').hexdigest()
              and key.endswith('/test_actual_alignment_error_pr0/aligned'), 'unexpected skipped object')
    retained_dirs = {r['path'] for r in m['directories']
                     if any(Path(p).is_relative_to(r['path']) for p in skipped)}
    check(retained_dirs == set(PLAN['skipped_directories']) and len(retained_dirs) == 12,
          'skip ancestor set differs')
    check(PLAN['expected_deleted'] == dict(regular=8417, symlink=822, synthetic_test_FIFO=6,
                                          directory=11300), 'subset deletion count differs')
    check(PLAN['expected_absent_roots'] == 72 and PLAN['expected_retained_roots'] == 6
          and PLAN['conservative_regular_reclaim_bytes'] == 3221491712, 'subset allocation/root count differs')
    return m, roots


def candidate_gate(m, roots):
    objects, directories = {}, {}
    for root in roots:
        fd = canonical_directory(root)
        os.close(fd)
        for base, dirs, files in os.walk(root, followlinks=False,
                                          onerror=lambda e: (_ for _ in ()).throw(e)):
            b = Path(base)
            s = b.lstat()
            directories[b.relative_to(ROOT).as_posix()] = meta(s)
            anchor = identity(s)
            check(ANCHORS.setdefault(str(b), anchor) == anchor, 'candidate directory anchor differs')
            for name in list(dirs):
                if (b / name).is_symlink():
                    dirs.remove(name)
                    files.append(name)
            for name in files:
                p = b / name
                objects[p.relative_to(ROOT).as_posix()] = meta(p.lstat())
    check(set(objects) == {r['path'] for r in m['files']}, 'candidate object set differs')
    check(set(directories) == {r['path'] for r in m['directories']}, 'candidate directory set differs')
    groups = collections.defaultdict(list)
    for row in m['directories']:
        check(directories[row['path']] == {k: row[k] for k in FIELDS},
              'candidate directory metadata differs: ' + row['path'])
    for row in m['files']:
        p = ROOT / row['path']
        expected = {k: row[k] for k in FIELDS}
        if row['path'] in {r['path'] for r in PLAN['skipped_files']}:
            # These six are never unlinked. Only this observed block-count drift
            # is accepted, with every identity/content field still checked.
            expected['allocated_bytes'] = 0
        check(objects[row['path']] == expected,
              'candidate object metadata differs: ' + row['path'])
        if row['type'] == 'regular':
            check(stat.S_ISREG(row['mode']), 'regular type differs')
            groups[(row['device'], row['inode'])].append(row)
        elif row['type'] == 'symlink':
            check(stat.S_ISLNK(row['mode']) and os.readlink(p) == row['target'],
                  'symlink target differs: ' + row['path'])
            check(inside(p.resolve(strict=True), roots), 'symlink target outside candidate')
        else:
            check(row['type'] == 'synthetic_test_FIFO' and stat.S_ISFIFO(row['mode']),
                  'unexpected special candidate')
    for aliases in groups.values():
        check(all(r['nlink'] == len(aliases) for r in aliases), 'external hardlink')
        stable_sha(ROOT / aliases[0]['path'], aliases[0]['sha256'])
    check(len(groups) == 8405, 'inode count differs')
    check(sum(a[0]['allocated_bytes'] for a in groups.values()) == 3221516288,
          'allocated blocks differ')
    return groups


def proc_gate(roots):
    active, errors = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            try:
                value = os.readlink(proc / 'cwd')
                if inside(Path(value.removesuffix(' (deleted)')), roots):
                    active.append(dict(pid=proc.name, kind='cwd', path=value))
            except FileNotFoundError:
                pass
            for fd in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if inside(Path(value.removesuffix(' (deleted)')), roots):
                    active.append(dict(pid=proc.name, kind='fd', path=value))
            try:
                for line in (proc / 'maps').read_text().splitlines():
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6 and inside(Path(parts[5].removesuffix(' (deleted)')), roots):
                        active.append(dict(pid=proc.name, kind='mmap', path=parts[5]))
            except FileNotFoundError:
                pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError as exc:
            errors.append(dict(pid=proc.name, error=str(exc)))
    check(not active and not errors, 'active candidate references/proc scan errors: ' + repr((active, errors)))
    return dict(active_references=active, scan_errors=errors)


def protected_gate(m, roots, groups):
    expected, counts = {}, {}
    def add(path, sha, size=None):
        lexical = absolute(path)
        resolved = lexical.resolve(strict=True)
        check(not inside(lexical, roots) and not inside(resolved, roots),
              'protected path intersects candidate: ' + str(lexical))
        check(resolved.is_relative_to(ROOT) or str(resolved) in ALLOWED_EXTERNAL_PROTECTED,
              'unexpected external source SHA reference: ' + str(resolved))
        prior = expected.get(str(resolved))
        check(prior is None or prior['sha256'] == sha, 'conflicting expected SHA: ' + str(resolved))
        expected[str(resolved)] = dict(path=str(resolved), sha256=sha, bytes=size)
    for lock in m['protected_source_locks']:
        add(lock['path'], lock['sha256'], lock['bytes'])
        stable_sha(absolute(lock['path']), lock['sha256'])
        data = json.loads(absolute(lock['path']).read_text())
        if 'files' in data:
            rows = data['files']
        else:
            # P3 lock is an entire path->SHA map; ALL keys, relative and absolute.
            rows = [dict(path=key, sha256=digest) for key, digest in data.items()]
        counts[lock['path']] = len(rows)
        for row in rows:
            add(row['path'], row['sha256'], row.get('bytes'))
    check(sorted(counts.values()) == [70, 2048, 2198, 2724], 'protected lock count differs')
    prior_path = ROOT / 'artifacts/prefix_io_v1/server08-p3-16/source-preservation-final-p3.json'
    prior = json.loads(prior_path.read_text())
    for key in ('source_manifest', 'source_registration'):
        add(prior[key]['path'], prior[key]['sha256'])
        stable_sha(absolute(prior[key]['path']), prior[key]['sha256'])
    source = Path(prior['source_root'])
    kv = json.loads(absolute(prior['source_manifest']['path']).read_text())
    check(len(kv) == 3048 and sum(r['bytes'] for r in kv) == 2796552192, 'registered KV count/bytes differ')
    expected_bins = {str(source / r['path']) for r in kv}
    check({str(p) for p in source.rglob('*.bin')} == expected_bins, 'extra or missing registered KV')
    for row in kv:
        add(source / row['path'], row['sha256'], row['bytes'])
    for row in m['retained_benchmark_evidence']:
        add(row['path'], row['sha256'], row['bytes'])
    add(LEDGER, LEDGER_SHA)
    add(PERMISSIONS, PERMISSIONS_SHA)
    checked = []
    bytes_hashed = 0
    for key, row in sorted(expected.items()):
        p = Path(key)
        s = p.lstat()
        check((s.st_dev, s.st_ino) not in groups, 'protected inode intersects candidate: ' + key)
        if row['bytes'] is not None:
            check(s.st_size == row['bytes'], 'protected size differs: ' + key)
        stable_sha(p, row['sha256'])
        checked.append(dict(path=key, sha256=row['sha256'], **meta(s)))
        bytes_hashed += s.st_size
    ledger = json.loads((ROOT / LEDGER).read_text())
    # Reservations vary by historical schema; the frozen ledger itself must match.
    check(ledger.get('active') is None and ledger.get('active_reservation') is None,
          'active GPU reservation')
    return dict(status='PASS_ALL_PROTECTED_BYTES', lock_reference_counts=counts,
                registered_KV_files=3048, registered_KV_bytes=2796552192,
                unique_protected_files=len(checked), bytes_hashed=bytes_hashed,
                GPU_runs=0, files=checked)


def inventory(roots, output_name):
    # All noncandidate objects, including original source, models, env, private
    # GPU caches and immutable patch baselines. AUX is metadata-only as well.
    ancestors = {ROOT}
    for p in roots + [OUT]:
        ancestors.update(p.parents)
    result = {}
    external = Path('/root/prefix-io-v1-validation')
    scopes = [(ROOT, '')]
    if external.exists():
        check(external.is_dir() and not external.is_symlink(), 'unexpected AUX scope')
        scopes.append((external, '@AUX/'))
    for scope, prefix in scopes:
        for base, dirs, files in os.walk(scope, followlinks=False,
                                          onerror=lambda e: (_ for _ in ()).throw(e)):
            b = Path(base)
            key = prefix + (b.relative_to(scope).as_posix() if b != scope else '.')
            row = meta(b.lstat())
            if b in ancestors:
                for f in ('mtime_ns', 'bytes', 'allocated_bytes', 'nlink'):
                    row.pop(f)
            result[key] = row
            for name in list(dirs):
                p = b / name
                if scope == ROOT and (p in roots or p == OUT):
                    dirs.remove(name)
                elif p.is_symlink():
                    dirs.remove(name)
                    files.append(name)
            for name in files:
                p = b / name
                key = prefix + p.relative_to(scope).as_posix()
                row = meta(p.lstat())
                if stat.S_ISLNK(row['mode']):
                    row['target'] = os.readlink(p)
                result[key] = row
    with (OUT / output_name).open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            for key, row in sorted(result.items()):
                zipped.write((json.dumps(dict(path=key, **row), sort_keys=True) + '\n').encode())
        raw.flush()
        os.fsync(raw.fileno())
    return result


def disk():
    s = os.statvfs(ROOT)
    return dict(created_utc=utc(), total_bytes=s.f_blocks * s.f_frsize,
                free_bytes=s.f_bavail * s.f_frsize, filesystem_free_bytes=s.f_bfree * s.f_frsize,
                used_bytes=(s.f_blocks - s.f_bfree) * s.f_frsize)


def verified_preflight_inventory():
    # Reuse the completed no-deletion baseline, extending the comparison window
    # to the original preflight. Manifest, all protected SHA and candidate gates
    # are still repeated immediately in this execute process.
    baseline = OUT / 'PREFLIGHT_V3_NONCANDIDATE_BEFORE.jsonl.gz'
    stable_sha(baseline, PREFLIGHT_INVENTORY_SHA)
    prior = json.loads((OUT / 'PREFLIGHT_V3_PREFLIGHT.json').read_text())
    command = json.loads((OUT / 'PREFLIGHT_COMMAND_03_V3.json').read_text())
    check(prior['status'] == 'PASS_CLEANUP_PREFLIGHT_NO_DELETION' and command['exit'] == 0
          and prior['frozen_manifest_sha256'] == MANIFEST_SHA
          and prior['restricted_plan_sha256'] == PLAN_SHA
          and not prior['deletion_performed'], 'completed baseline gate differs')
    rows = {}
    with gzip.open(baseline, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            key = row.pop('path')
            check(key not in rows, 'duplicate baseline path')
            rows[key] = row
    check(len(rows) == prior['noncandidate_metadata_objects'] == 1020725,
          'baseline object count differs')
    write_new('EXECUTE_V4_REUSED_PREFLIGHT_BASELINE.json', dict(
        baseline_path=str(baseline), baseline_sha256=PREFLIGHT_INVENTORY_SHA,
        baseline_bytes=baseline.stat().st_size, metadata_objects=len(rows),
        comparison_starts_at_completed_no_deletion_preflight=True,
        all_protected_SHA_and_candidate_gates_repeated_in_execute=True))
    return rows


def parent_fd(row):
    p = ROOT / row['path']
    return canonical_directory(p.parent), p.name


def unlink_exact(row, remaining):
    # No Linux unlink API offers atomic inode compare-and-delete. All candidate
    # generators must remain stopped; proc gates plus descriptor anchors and
    # immediate metadata/SHA rechecks reduce the check-then-act race.
    fd, name = parent_fd(row)
    try:
        expected_nlink = remaining[(row['device'], row['inode'])] if row['type'] == 'regular' else None
        exact(os.stat(name, dir_fd=fd, follow_symlinks=False), row, expected_nlink)
        if row['type'] == 'regular':
            obj = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                exact(os.fstat(obj), row, expected_nlink)
                h = hashlib.sha256()
                while True:
                    block = os.read(obj, 4 * 1024 * 1024)
                    if not block:
                        break
                    h.update(block)
                check(h.hexdigest() == row['sha256'], 'unlink SHA differs: ' + row['path'])
                exact(os.fstat(obj), row, expected_nlink)
                exact(os.stat(name, dir_fd=fd, follow_symlinks=False), row, expected_nlink)
                os.unlink(name, dir_fd=fd)
            finally:
                os.close(obj)
            remaining[(row['device'], row['inode'])] -= 1
        elif row['type'] == 'symlink':
            check(os.readlink(name, dir_fd=fd) == row['target'], 'unlink symlink differs')
            os.unlink(name, dir_fd=fd)
        else:
            check(stat.S_ISFIFO(row['mode']), 'unlink unexpected type')
            os.unlink(name, dir_fd=fd)
    finally:
        os.close(fd)


def remove_empty_directory(row):
    fd, name = parent_fd(row)
    try:
        s = os.stat(name, dir_fd=fd, follow_symlinks=False)
        check(identity(s) == (row['device'], row['inode'], row['mode']) and stat.S_ISDIR(s.st_mode),
              'directory inode/mode differs: ' + row['path'])
        # Kernel rmdir fails if any unapproved/new child remains; never recurse.
        os.rmdir(name, dir_fd=fd)
    finally:
        os.close(fd)


def run(mode):
    check(mode in ('preflight', 'execute'), 'unsupported mode')
    started = time.monotonic()
    m, roots = fresh_manifest()
    check(stable_sha(OUT / 'APPROVED_CPU_CLEANUP_AUTHORIZATION.json') == AUTH_SHA,
          'cleanup authorization receipt differs')
    auth = json.loads((OUT / 'APPROVED_CPU_CLEANUP_AUTHORIZATION.json').read_text())
    check(auth['human_text'] == AUTH_TEXT and auth['approved_manifest_sha256'] == MANIFEST_SHA,
          'cleanup authorization scope differs')
    groups = candidate_gate(m, roots)
    protected_before = protected_gate(m, roots, groups)
    before_inventory = (verified_preflight_inventory() if mode == 'execute'
                        else inventory(roots, 'PREFLIGHT_V4_NONCANDIDATE_BEFORE.jsonl.gz'))
    process_scan = proc_gate(roots)
    pre = dict(status='PASS_CLEANUP_PREFLIGHT_NO_DELETION', created_utc=utc(),
               frozen_manifest_sha256=MANIFEST_SHA, authorized_roots=78,
               regular_paths=8423, symlinks=822, synthetic_FIFOs=6, directories=len(m['directories']),
               protected=protected_before, noncandidate_metadata_objects=len(before_inventory),
               restricted_plan_sha256=PLAN_SHA, skip_files=PLAN['skipped_files'],
               skip_directories=PLAN['skipped_directories'],
               process_scan=process_scan, GPU_runs=0, deletion_performed=False,
               elapsed_seconds=time.monotonic() - started)
    write_new(mode.upper() + '_V4_PREFLIGHT.json', pre)
    if mode == 'preflight':
        print(json.dumps({k: v for k, v in pre.items() if k != 'protected'}, ensure_ascii=False))
        return
    independent = json.loads((OUT / 'INDEPENDENT_BOUNDARY_REVIEW.json').read_text())
    check(independent['status'] == 'PASS' and independent['frozen_manifest_sha256'] == MANIFEST_SHA,
          'independent boundary review absent or failed')
    code_review = json.loads((OUT / 'INDEPENDENT_CLEANUP_SCRIPT_REVIEW.json').read_text())
    check(code_review['status'] == 'PASS' and code_review['script_sha256'] == stable_sha(Path(__file__)),
          'independent cleanup script review absent or failed')
    # Recheck object-set/SHA/proc directly before the first mutation.
    candidate_gate(m, roots)
    proc_gate(roots)
    stable_sha(ROOT / LEDGER, LEDGER_SHA)
    before_disk = disk()
    write_new('DISK_IMMEDIATELY_BEFORE_CLEANUP.json', before_disk)
    remaining = {key: len(value) for key, value in groups.items()}
    skipped_files = {r['path'] for r in PLAN['skipped_files']}
    skipped_directories = set(PLAN['skipped_directories'])
    count = collections.Counter()
    journal = OUT / 'EXACT_DELETION_JOURNAL.jsonl'
    mutation_error = None
    with journal.open('x', encoding='utf-8', buffering=1) as log:
        try:
            for row in m['files']:
                if row['path'] in skipped_files:
                    continue
                unlink_exact(row, remaining)
                count[row['type']] += 1
                log.write(json.dumps(dict(operation='unlink', path=row['path'], type=row['type'],
                                          device=row['device'], inode=row['inode'])) + '\n')
            for row in sorted(m['directories'], key=lambda r: (-len(Path(r['path']).parts), r['path'])):
                if row['path'] in skipped_directories:
                    continue
                remove_empty_directory(row)
                count['directory'] += 1
                log.write(json.dumps(dict(operation='rmdir_empty', path=row['path'],
                                          device=row['device'], inode=row['inode'])) + '\n')
        except BaseException as exc:
            mutation_error = dict(type=type(exc).__name__, message=str(exc))
        log.flush()
        os.fsync(log.fileno())
    after_inventory = inventory(roots, 'EXECUTE_V4_NONCANDIDATE_AFTER.jsonl.gz')
    removed_outside = sorted(set(before_inventory) - set(after_inventory))
    added_outside = sorted(set(after_inventory) - set(before_inventory))
    changed_outside = sorted(k for k in set(before_inventory) & set(after_inventory)
                             if before_inventory[k] != after_inventory[k])
    write_new('NONCANDIDATE_METADATA_COMPARISON.json', dict(
        before_objects=len(before_inventory), after_objects=len(after_inventory),
        removed=removed_outside, added=added_outside, changed=changed_outside,
        excluded_current_audit_directory=str(OUT),
        candidate_and_audit_ancestor_directory_child_metadata_excluded=True))
    protected_after = protected_gate(m, roots, groups)
    write_new('PROTECTED_BYTES_AFTER_CLEANUP.json', protected_after)
    after_disk = disk()
    absent = [str(p.relative_to(ROOT)) for p in roots if not os.path.lexists(p)]
    remaining_files, remaining_dirs = set(), set()
    for root in roots:
        if not os.path.lexists(root):
            continue
        for base, dirs, files in os.walk(root, followlinks=False,
                                          onerror=lambda e: (_ for _ in ()).throw(e)):
            b = Path(base)
            remaining_dirs.add(b.relative_to(ROOT).as_posix())
            for name in list(dirs):
                if (b / name).is_symlink():
                    dirs.remove(name)
                    files.append(name)
            remaining_files.update((b / name).relative_to(ROOT).as_posix() for name in files)
    skipped_after = []
    for row in m['files']:
        if row['path'] in skipped_files:
            p = ROOT / row['path']
            now = meta(p.lstat())
            check({k: now[k] for k in FIELDS if k != 'allocated_bytes'} ==
                  {k: row[k] for k in FIELDS if k != 'allocated_bytes'}, 'retained object identity changed')
            stable_sha(p, row['sha256'])
            skipped_after.append(dict(path=row['path'], **now, sha256=row['sha256']))
    write_new('RETAINED_SIX_ZERO_BYTE_FIXTURES_AFTER.json', skipped_after)
    retained_dirs_after = []
    for row in m['directories']:
        if row['path'] in skipped_directories:
            s = (ROOT / row['path']).lstat()
            check(identity(s) == (row['device'], row['inode'], row['mode']) and stat.S_ISDIR(s.st_mode),
                  'retained directory identity changed: ' + row['path'])
            retained_dirs_after.append(dict(path=row['path'], **meta(s)))
    write_new('RETAINED_TWELVE_DIRECTORY_IDENTITIES_AFTER.json', retained_dirs_after)
    passed = (mutation_error is None and len(absent) == PLAN['expected_absent_roots'] and
              count == PLAN['expected_deleted'] and
              remaining_files == skipped_files and remaining_dirs == skipped_directories and
              not removed_outside and not added_outside and not changed_outside and
              protected_before['files'] == protected_after['files'])
    receipt = dict(status='PASS_APPROVED_CPU_TEMP_CLEANUP' if passed else 'INCOMPLETE_OR_VERIFICATION_FAILED',
                   created_utc=utc(), human_authorization=AUTH_TEXT,
                   frozen_manifest_sha256=MANIFEST_SHA, deleted=dict(count), absent_roots=len(absent),
                   restricted_execution_plan_sha256=PLAN_SHA,
                   skipped_files=sorted(skipped_files), skipped_directories=sorted(skipped_directories),
                   retained_roots=78-len(absent),
                   remaining_candidate_objects_exactly_skipped=remaining_files == skipped_files
                   and remaining_dirs == skipped_directories,
                   deleted_only_frozen_paths=True, unexpected_mutation_error=mutation_error,
                   noncandidate_metadata_objects=len(before_inventory),
                   noncandidate_before_inventory_sha256=PREFLIGHT_INVENTORY_SHA,
                   noncandidate_before_inventory_is_completed_no_deletion_preflight=True,
                   noncandidate_removed=removed_outside, noncandidate_added=added_outside,
                   noncandidate_changed=changed_outside, protected_bytes_before_after_equal=
                   protected_before['files'] == protected_after['files'],
                   protected_unique_files=protected_after['unique_protected_files'],
                   protected_reference_counts=protected_after['lock_reference_counts'],
                   registered_KV_files=3048, registered_KV_bytes=2796552192,
                   conservative_regular_allocated_bytes=PLAN['conservative_regular_reclaim_bytes'],
                   candidate_directory_allocated_bytes=m['directory_allocated_bytes']
                   - PLAN['retained_directory_allocated_bytes'],
                   disk_before=before_disk, disk_after=after_disk,
                   measured_free_increase_bytes=after_disk['free_bytes'] - before_disk['free_bytes'],
                   AUX_reclaim_bytes=0, AUX_20GiB_cap_unchanged=True,
                   GPU_initialized=False, GPU_runs=0, GPU_budget_seconds_added=0,
                   ledger_sha256=stable_sha(ROOT / LEDGER, LEDGER_SHA),
                   permissions_sha256=stable_sha(ROOT / PERMISSIONS, PERMISSIONS_SHA),
                   elapsed_seconds=time.monotonic() - started)
    write_new('APPROVED_CPU_CLEANUP_RESULT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    check(passed, 'cleanup incomplete or postflight verification failed; see receipt')


if __name__ == '__main__':
    # Receipt SHA substituted when creating this exact execution artifact.
    AUTH_SHA = '622717dc2c3fc64489f7c63076dd7878411cbf7650ee671550ba3eb4e5c3bd3b'
    run(sys.argv[1])
