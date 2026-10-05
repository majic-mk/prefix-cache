"""Seal completed normal CPU evidence without model, GPU, deletion or execution.

Candidate/runtime files remain untouched. Test counts derive from the frozen
test suite; completion and source proofs come from actual server outputs. Only
bounded regular files in the two new delivery directories are archived.
"""
import argparse
import ast
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004'
DELIVERY = 'artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
COMMON_LOCK = D + '/COMMON_SOURCE_LOCK.json'
CPU_RESULT = DELIVERY + '/SERVER_CPU_01/CPU_RESULT.json'
TEST_STDOUT = DELIVERY + '/SERVER_CPU_01/TEST_STDOUT.log'
ARCHIVE = DELIVERY + '/NORMAL_NATIVE_CPU_DELIVERY.tar.gz'
MANIFEST = DELIVERY + '/NORMAL_NATIVE_CPU_DELIVERY_MANIFEST.json'
RECEIPT = DELIVERY + '/NORMAL_NATIVE_CPU_DELIVERY_SEAL_RECEIPT.json'
PER_FILE_LIMIT = 32 * 1024**2
TOTAL_LIMIT = 100 * 1024**2
FREE_FLOOR = 8 * 1024**3
FORBIDDEN_IMPORTS = ('torch', 'vllm', 'py_kvcache')


def require(value, reason):
    if not value:
        raise ValueError('CPU_DELIVERY_SEAL_REJECTED: ' + reason)


def relative_shape(relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(part not in ('', '.', '..') for part in relative.split('/')),
        'single project-relative POSIX path')
    return relative


def safe(relative):
    relative_shape(relative)
    root = ROOT.resolve(strict=True)
    require(root == ROOT and not ROOT.is_symlink(), 'fixed original server project root')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink source/output refused: ' + relative)
    require(path.resolve().is_relative_to(root), 'path outside original project')
    return path


def regular(relative):
    path = safe(relative)
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and 0 <= st.st_size <= PER_FILE_LIMIT,
        'bounded regular file required: ' + relative)
    return path, st


def ref(relative):
    path, st = regular(relative)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(path.stat().st_size == st.st_size, 'file size changed while hashing: ' + relative)
    return dict(path=relative, bytes=st.st_size, sha256=digest)


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))


def read(relative):
    path, _ = regular(relative)
    return parse(path.read_bytes())


def exact(value, expected):
    return type(value) is type(expected) and value == expected


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        'finite typed ' + name)
    return value


def checked_document(relative, suffix):
    relative_shape(relative)
    require(relative.startswith(DELIVERY + '/') and Path(relative).suffix == suffix,
        'completed document inside this delivery')
    return ref(relative)


def source_rows(lock):
    rows = lock.get('files')
    require(type(rows) is list and 1 <= len(rows) <= 10000, 'complete bounded common source rows')
    result = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
            and type(row['bytes']) is int and row['bytes'] >= 0
            and type(row['sha256']) is str and len(row['sha256']) == 64
            and all(char in '0123456789abcdef' for char in row['sha256']), 'exact source metadata')
        relative_shape(row['path'])
        require(row['path'] not in result and row['path'] != LEDGER, 'unique immutable source path, no live ledger')
        result[row['path']] = row
    return rows, result


def frozen_test_count(locked):
    runner = D + '/run_cpu_normal_native.py'
    require(runner in locked and ref(runner) == locked[runner], 'actual frozen CPU runner')
    tree = ast.parse(safe(runner).read_bytes())
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'TEST_NAMES' for target in node.targets)]
    require(len(declarations) == 1, 'one explicit frozen test suite')
    names = ast.literal_eval(declarations[0])
    require(type(names) is tuple and names and len(set(names)) == len(names)
        and all(type(name) is str and name.startswith('test_') and '/' not in name
        and '\\' not in name and name.endswith('.py') for name in names), 'bounded frozen test names')
    count = 0
    for name in names:
        relative = D + '/' + name
        require(relative in locked and ref(relative) == locked[relative], 'actual frozen test source: ' + name)
        test_tree = ast.parse(safe(relative).read_bytes())
        for node in test_tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                and base.value.id == 'unittest' and base.attr == 'TestCase' for base in node.bases):
                continue
            methods = [child.name for child in node.body if isinstance(child, ast.FunctionDef)
                and child.name.startswith('test_')]
            require(len(set(methods)) == len(methods), 'unique discovered test methods')
            count += len(methods)
    require(count > 0, 'nonempty completed frozen tests')
    return count


def completion(ancestry_relative, ledger_raw):
    lock_ref = ref(COMMON_LOCK)
    lock = read(COMMON_LOCK)
    require(lock.get('schema') == 'c5_normal_common_source_lock_v1'
        and lock.get('scope') == 'server11_c5_normal_native_runtime_v1'
        and exact(lock.get('GPU_operations'), 0) and lock.get('gpu_authority_issued') is False
        and lock.get('normal_runtime_qualified') is False, 'actual CPU-only common source lock')
    rows, locked = source_rows(lock)
    row_digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    count = frozen_test_count(locked)
    cpu = read(CPU_RESULT)
    expected = dict(schema='c5_normal_native_server_cpu_result_v1', status='PASS_NORMAL_NATIVE_CPU_CHECKS',
        location='server_cpu', tests=count, passed=count, failed=0, errors=0, skipped=0,
        source_count=len(rows), source_before=row_digest, source_after=row_digest,
        source_lock_sha256=lock_ref['sha256'], GPU_jobs=0, GPU_seconds=0,
        actual_model_processes_started=0, gpu_ledger_unchanged=True,
        gpu_ledger_sha256=hashlib.sha256(ledger_raw).hexdigest(),
        torch_vllm_native_backend_imports=[], forbidden_model_GPU_import_attempts=[],
        normal_native_execution_verified=False, actual_on_installed=False,
        normal_runtime_condition_qualified=False, full_runtime_cost_qualified=False,
        strategy_effect_verified=False, P4_completed=False, old_tests_or_CPU_benchmark_rerun=False)
    require(all(exact(cpu.get(key), value) for key, value in expected.items()),
        'actual complete test/source/ledger/import CPU result required')
    ledger = parse(ledger_raw)
    require(ledger.get('active_reservation') is None, 'original GPU ledger must be idle')
    used = number(ledger.get('gpu_wall_seconds'), 'actual cumulative GPU usage')
    require(used <= 28800 and exact(cpu.get('gpu_seconds_used'), used)
        and exact(cpu.get('remaining_GPU_seconds'), 28800 - used), 'original GPU budget accounting')
    require(safe(TEST_STDOUT).stat().st_size > 0, 'completed actual test log')
    ancestry = read(ancestry_relative)
    require(ancestry.get('schema') == 'normal_native_frozen_original_source_ancestry_v1'
        and ancestry.get('status') == 'PASS_ORIGINAL_ENGINE_AND_NUMERICAL_ANCESTRY'
        and exact(ancestry.get('GPU_jobs'), 0) and exact(ancestry.get('model_loads'), 0)
        and ancestry.get('normal_runtime_qualified') is False and ancestry.get('strategy_effect_verified') is False
        and ancestry.get('serializer_byte_identical') is True
        and ancestry.get('notification_adapter_byte_identical') is True,
        'completed original-engine and numerical ancestry audit')
    proof = ancestry.get('unchanged_callable_AST')
    common = ancestry.get('frozen_G_common_Python_files_verified')
    require(type(proof) is list and proof and exact(ancestry.get('unchanged_callable_count'), len(proof))
        and type(common) is list and common
        and exact(ancestry.get('frozen_G_common_Python_file_count'), len(common)), 'complete actual ancestry evidence')
    for row in (lock.get('normal_binding_ref'), lock.get('calibration_binding_ref'), lock.get('calibration_source_lock_ref')):
        require(type(row) is dict and ref(row['path']) == row, 'immutable actual calibration/normal provenance')
    for relative in (D + '/NORMAL_RUNTIME_BINDING.json', D + '/GPU03_LEDGER_SNAPSHOT.json', D + '/CONFIG_off.json'):
        regular(relative)
    # No current-mode authorization was granted by this CPU preparation.
    for mode in ('off', 'shadow', 'on'):
        for stem in ('AUTHORITY_', 'HUMAN_GPU_GRANT_', 'EFFECTIVE_GPU_PERMISSION_', 'SCOPE_'):
            require(not safe(D + '/' + stem + mode + '.json').exists(), 'new GPU authority not part of CPU sealing')
    return dict(cpu_result_ref=ref(CPU_RESULT), test_stdout_ref=ref(TEST_STDOUT),
        ancestry_ref=ref(ancestry_relative), common_source_lock_ref=lock_ref,
        actual_tests=count, actual_source_count=len(rows), current_gpu_seconds=used)


def collect_files(required):
    values = set()
    for base, suffixes in ((D, {'.py', '.md', '.json'}),
        (DELIVERY, {'.py', '.md', '.json', '.log'})):
        directory = safe(base)
        require(directory.is_dir(), 'existing fixed delivery directory')
        for path in directory.iterdir():
            require(not path.is_symlink(), 'symlink in delivery refused')
            if path.is_file() and path.suffix in suffixes:
                relative = path.relative_to(ROOT).as_posix()
                require(relative not in (ARCHIVE, MANIFEST, RECEIPT), 'existing seal output must not be re-archived')
                values.add(relative)
    # Only these two completed CPU outputs are needed from the fixture directory.
    values.update((CPU_RESULT, TEST_STDOUT))
    require(set(required).issubset(values), 'final report/proposal/audit not included')
    rows = [ref(relative) for relative in sorted(values)]
    require(sum(row['bytes'] for row in rows) <= TOTAL_LIMIT, 'source input size exceeds 100 MiB')
    require(all('/fixture-tmp/' not in row['path'] and '/__pycache__/' not in row['path'] for row in rows),
        'fixture and bytecode files excluded')
    return rows


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.size += len(value)
        return value


def tar_info(relative, size, mtime):
    info = tarfile.TarInfo(name=relative)
    info.type = tarfile.REGTYPE
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mtime = int(mtime)
    return info


def append_json(relative, document):
    raw = (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    require(len(raw) <= PER_FILE_LIMIT, 'bounded sealing metadata')
    with safe(relative).open('xb') as stream:
        stream.write(raw)
    return raw


def verify_archive(rows, manifest_raw):
    expected = {row['path']: row for row in rows}
    expected[MANIFEST] = dict(path=MANIFEST, bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest())
    require(len(expected) == len(rows) + 1, 'manifest self-reference excluded')
    order = [row['path'] for row in rows] + [MANIFEST]
    seen = []; total = 0
    with tarfile.open(safe(ARCHIVE), mode='r:gz') as archive:
        for member in archive:
            relative_shape(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and not member.pax_headers,
                'only single regular tar members')
            require(member.name in expected and member.name not in seen and member.linkname == '',
                'duplicate or foreign tar header path')
            row = expected[member.name]
            require(member.size == row['bytes'] and 0 <= member.size <= PER_FILE_LIMIT, 'tar header size mismatch')
            stream = archive.extractfile(member)
            require(stream is not None, 'regular archived file stream')
            with stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            require(digest == row['sha256'], 'archived bytes differ from actual source')
            total += member.size; seen.append(member.name)
            require(total <= TOTAL_LIMIT + len(manifest_raw), 'bounded total archive content')
    require(seen == order and seen[-1] == MANIFEST, 'exact source members and terminal manifest')
    return len(seen), total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ancestry-relative', required=True)
    parser.add_argument('--report-relative', required=True)
    parser.add_argument('--proposal-relative', required=True)
    args = parser.parse_args(argv)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only hidden GPU environment')
    require(not any(name.split('.')[0] in FORBIDDEN_IMPORTS for name in sys.modules), 'no model/GPU/backend imports')
    for relative in (ARCHIVE, MANIFEST, RECEIPT):
        require(not safe(relative).exists(), 'append-only seal output already exists')
    report_ref = checked_document(args.report_relative, '.md')
    proposal_ref = checked_document(args.proposal_relative, '.json')
    ancestry_ref = checked_document(args.ancestry_relative, '.json')
    ledger_raw = safe(LEDGER).read_bytes()
    gate = completion(args.ancestry_relative, ledger_raw)
    rows = collect_files((args.ancestry_relative, args.report_relative, args.proposal_relative,
        str(Path(__file__).resolve().relative_to(ROOT).as_posix())))
    _, locked = source_rows(read(COMMON_LOCK))
    for row in rows:
        if row['path'] in locked:
            require(row == locked[row['path']], 'sealed source differs from completed frozen CPU closure')
        if row['path'].startswith(D + '/') and row['path'].endswith('.py'):
            require(row['path'] in locked, 'unfrozen candidate Python source cannot be sealed as validated')
    require(next(row for row in rows if row['path'] == COMMON_LOCK) == gate['common_source_lock_ref'],
        'common source lock must retain its CPU-result-bound bytes')
    manifest = dict(schema='c5_normal_cpu_delivery_manifest_v1',
        origin='actual_completed_server_cpu_evidence', scope='bounded_source_and_CPU_delivery_only',
        files=rows, source_count=len(rows), source_bytes=sum(row['bytes'] for row in rows),
        archive_relative=ARCHIVE, manifest_relative=MANIFEST, manifest_self_reference_excluded=True,
        completion_evidence=gate, final_report_ref=report_ref, next_GPU_proposal_ref=proposal_ref,
        ancestry_ref=ancestry_ref, gpu_ledger_sha256_before=hashlib.sha256(ledger_raw).hexdigest(),
        GPU_commands_invoked_this_action=0, normal_GPU_experiment_performed=False,
        normal_runtime_qualified=False, full_runtime_cost_qualified=False, strategy_effect_verified=False,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    require(len(manifest_raw) <= PER_FILE_LIMIT and manifest['source_bytes'] + len(manifest_raw) <= TOTAL_LIMIT,
        'whole manifest and source input <=100 MiB')
    # Keep the floor even under worst-case incompressible output and tar headers.
    reserve = manifest['source_bytes'] + len(manifest_raw) + (len(rows) + 1) * 2048 + 1024**2
    require(shutil.disk_usage(ROOT).free >= FREE_FLOOR + reserve, '8 GiB PRIMARY floor with archive reserve')
    append_json(MANIFEST, manifest)
    require(safe(MANIFEST).read_bytes() == manifest_raw, 'exact terminal manifest bytes')
    with safe(ARCHIVE).open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz', format=tarfile.USTAR_FORMAT) as archive:
            for row in rows:
                path, st = regular(row['path'])
                require(ref(row['path']) == row, 'source drift before archiving')
                with path.open('rb') as source:
                    reader = HashingReader(source)
                    archive.addfile(tar_info(row['path'], row['bytes'], st.st_mtime), reader)
                    require(reader.size == row['bytes'] and reader.digest.hexdigest() == row['sha256'],
                        'source changed during archive stream')
            archive.addfile(tar_info(MANIFEST, len(manifest_raw), safe(MANIFEST).stat().st_mtime), io.BytesIO(manifest_raw))
    # Archive may be slightly larger than bounded inputs; hash by streaming.
    archive_path = safe(ARCHIVE)
    with archive_path.open('rb') as stream:
        archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive_size = archive_path.stat().st_size
    member_count, expanded_bytes = verify_archive(rows, manifest_raw)
    for row in rows:
        require(ref(row['path']) == row, 'source drift after complete tar verification')
    require(safe(MANIFEST).read_bytes() == manifest_raw, 'manifest drift after tar verification')
    require(safe(LEDGER).read_bytes() == ledger_raw, 'GPU budget ledger changed during CPU sealing')
    require(shutil.disk_usage(ROOT).free >= FREE_FLOOR, '8 GiB storage floor after sealing')
    with archive_path.open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == archive_sha, 'archive changed after validation')
    require(archive_path.stat().st_size == archive_size, 'archive size changed after validation')
    receipt = dict(schema='c5_normal_cpu_delivery_seal_receipt_v1', status='PASS_BOUNDED_CPU_DELIVERY_ARCHIVE',
        archive_ref=dict(path=ARCHIVE, bytes=archive_size, sha256=archive_sha), manifest_ref=ref(MANIFEST),
        source_count=len(rows), member_count=member_count, expanded_bytes=expanded_bytes,
        all_tar_members_verified=True, source_files_unchanged=True, no_files_deleted=True,
        GPU_ledger_before_SHA256=hashlib.sha256(ledger_raw).hexdigest(),
        GPU_ledger_after_SHA256=hashlib.sha256(safe(LEDGER).read_bytes()).hexdigest(),
        gpu_ledger_unchanged=True, GPU_commands_invoked_this_action=0,
        normal_GPU_experiment_performed=False, normal_runtime_qualified=False,
        full_runtime_cost_qualified=False, strategy_effect_verified=False, completion_evidence=gate)
    append_json(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
