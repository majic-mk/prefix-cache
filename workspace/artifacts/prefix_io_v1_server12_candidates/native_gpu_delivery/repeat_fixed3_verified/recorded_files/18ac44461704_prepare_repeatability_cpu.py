"""Freeze an explicit diagnostic extension and prepare three CPU-only configs.

Every inherited asset is byte checked. Calibration Q/N/H/S, binding, mathematics
and the prior failed normal qualification remain in their original directories.
This helper creates neither permissions nor GPU authority and imports no engine.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
NORMAL = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
BASE_LOCK = NORMAL + '/COMMON_SOURCE_LOCK.json'
BASE_LOCK_SHA = 'aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
REQUIRED_SOURCES = (
    'control_p4_single_file.py', 'run_p4_single_file_experiment.py', 'process_entry_context.py',
    'verify_repeatability.py', 'notification_runtime_adapter.py', 'observation_cpu_cost.py',
    'PROTOCOL.json', 'prepare_repeatability_cpu.py', 'run_fixed_repeatability.py',
    'test_repeatability_launcher.py', 'test_process_entry_context.py',
    'test_repeatability_entry.py', 'audit_process_entry_source.py',
    'PREREGISTRATION.json', 'PREREGISTRATION_LEDGER_SNAPSHOT.json',
)
OPTIONAL_SOURCES = ('test_repeatability_protocol.py', 'test_repeatability_verifier.py',
    'README_PROTOCOL.md', 'CPU_PROTOCOL_SOURCE_AUDIT.json')
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
IMPORT_ATTEMPTS = []


def require(value, reason):
    if not value:
        raise ValueError('REPEATABILITY_CPU_PREPARATION_REJECTED: ' + reason)


class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('CPU source preparation cannot import model/native backend: ' + fullname)
        return None


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative and
        not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'project-relative source/evidence path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'source/evidence symlink refused')
    require(path.resolve().is_relative_to(root), 'source/evidence outside project')
    return path


def shape(row, root):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
        type(row['bytes']) is int and row['bytes'] >= 0 and type(row['sha256']) is str and
        re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None, 'exact typed immutable reference')
    safe(root, row['path'])
    return row


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual regular source/evidence file')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
        (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'file changed while hashing')
    return dict(path=relative, bytes=after.st_size, sha256=digest)


def checked(root, row):
    shape(row, root)
    require(ref(root, row['path']) == row, 'actual byte/SHA reference drift: ' + row['path'])
    return row


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def read(root, relative):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, 'bounded JSON document')
    return parse(path.read_bytes())


def load(root, row, name):
    checked(root, row)
    spec = importlib.util.spec_from_file_location(name, safe(root, row['path']))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        checked(root, row)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


def require_idle_ledger(value):
    require(type(value) is dict and value.get('active_reservation') is None and
        type(value.get('events')) is list and len(value['events']) <= 10000, 'actual original idle ledger')
    seconds = value.get('gpu_wall_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 28800,
        'actual original eight-hour usage')
    require(seconds + 960 <= 28800, 'fixed three-job reservation exceeds original remaining budget')
    return seconds


def freeze(root, expected_ledger_sha256):
    root = Path(root).resolve(strict=True)
    require(root == ROOT and re.fullmatch(r'[0-9a-f]{64}', expected_ledger_sha256 or '') is not None,
        'fixed server root and actual complete ledger digest required')
    require(not safe(root, LOCK).exists(), 'append-only new common source lock')
    ledger_bytes = safe(root, LEDGER).read_bytes()
    require(hashlib.sha256(ledger_bytes).hexdigest() == expected_ledger_sha256, 'fresh ledger bytes differ')
    seconds = require_idle_ledger(parse(ledger_bytes))
    base_ref = ref(root, BASE_LOCK)
    require(base_ref['sha256'] == BASE_LOCK_SHA, 'actual immutable normal common lock')
    base = read(root, BASE_LOCK)
    require(base.get('schema') == 'c5_normal_common_source_lock_v1' and base.get('gpu_authority_issued') is False and
        base.get('normal_runtime_qualified') is False and type(base.get('files')) is list and
        1 <= len(base['files']) <= 10000, 'original normal source closure metadata')
    rows = {}
    def add(row):
        checked(root, row)
        require(row['path'] != LEDGER and row['path'] != LOCK, 'live budget/common lock cannot enter itself')
        require(row['path'] not in rows or rows[row['path']] == row, 'conflicting frozen source path')
        rows[row['path']] = row
    # Inherited JSON assets are immutable leaves, each fully SHA verified.
    for row in base['files']:
        require(row['path'] not in rows, 'duplicate inherited source path')
        add(row)
    add(base_ref)
    overlay = []
    for name in REQUIRED_SOURCES + OPTIONAL_SOURCES:
        path = safe(root, D + '/' + name)
        if name in OPTIONAL_SOURCES and not path.exists():
            continue
        row = ref(root, D + '/' + name)
        add(row)
        overlay.append(row)
    verifier = load(root, rows[D + '/verify_repeatability.py'], '_repeatability_prepare_readonly_verifier')
    protocol_ref = verifier.prepare_protocol(root)
    require(protocol_ref == rows[D + '/PROTOCOL.json'], 'actual preregistered protocol bytes')
    protocol = read(root, protocol_ref['path'])
    require(protocol['repetitions'] == 3 and type(protocol['repetitions']) is int and
        protocol['cost_upper_ns'] == 16238752 and protocol['step_budget_ns'] == 13171328,
        'original thresholds and fixed repetitions')
    prereg_snapshot_ref = rows[D + '/PREREGISTRATION_LEDGER_SNAPSHOT.json']
    require(prereg_snapshot_ref['sha256'] == expected_ledger_sha256 and
        safe(root, prereg_snapshot_ref['path']).read_bytes() == ledger_bytes,
        'actual preregistration snapshot is the same original idle ledger bytes')
    prior = protocol['immutable_prior_off_counterexample']
    for key in ('qualification_ref', 'raw_result_ref', 'guard_ref'):
        add(prior[key])
    add(protocol['original_verifier_ref'])
    qualification = read(root, prior['qualification_ref']['path'])
    require(qualification.get('native_execution_verified') is True and qualification.get('qualification_passed') is False and
        qualification.get('selected_gpu_elapsed_ns') == 16893473 and qualification.get('permits_next_mode') is None,
        'actual original failed normal gate is preserved')
    evidence = qualification.get('evidence_refs')
    require(type(evidence) is dict and set(evidence) == {'config', 'raw_result', 'actual_guard', 'source_before', 'source_after'},
        'actual complete prior counterexample evidence references')
    for row in evidence.values():
        add(row)
    require(len(rows) <= 10000, 'bounded full new source closure')
    require(safe(root, LEDGER).read_bytes() == ledger_bytes and ref(root, BASE_LOCK) == base_ref,
        'ledger/base source lock changed during CPU preparation')
    lock = dict(schema='c5_repeatability_common_source_lock_v1', scope='server12_c5_normal_repeatability_v1',
        files=[rows[path] for path in sorted(rows)], base_normal_lock_ref=base_ref,
        diagnostic_overlay_refs=overlay, protocol_ref=protocol_ref,
        preregistration_ref=rows[D + '/PREREGISTRATION.json'],
        preregistration_ledger_snapshot_ref=prereg_snapshot_ref,
        current_host_asset_verification=base['current_host_asset_verification'],
        immutable_prior_off_counterexample_refs={key: prior[key] for key in
            ('qualification_ref', 'raw_result_ref', 'guard_ref')},
        all_inherited_leaf_bytes_verified=True, inherited_source_count=len(base['files']),
        reused_original_binding_ref=rows[NORMAL + '/NORMAL_RUNTIME_BINDING.json'],
        reused_public_canonical_ref=rows[NORMAL + '/p4_single_file_receipt.py'],
        original_ledger_sha256_at_CPU_preparation=expected_ledger_sha256,
        GPU_operations=0, gpu_authority_issued=False, normal_runtime_qualified=False,
        native_execution_verified_by_this_freeze=False, calibration_refit=False, thresholds_changed=False)
    with safe(root, LOCK).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(lock, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    controller = load(root, rows[D + '/control_p4_single_file.py'], '_repeatability_cpu_prepare_controller')
    require(controller.D == D and controller.LOCK == LOCK and controller.BINDING == NORMAL + '/NORMAL_RUNTIME_BINDING.json',
        'actual diagnostic controller reuses the immutable real normal binding')
    prepared = controller.prepare_all(root)
    for row in rows.values():
        checked(root, row)
    require(safe(root, LEDGER).read_bytes() == ledger_bytes, 'GPU ledger changed during CPU config preparation')
    require(IMPORT_ATTEMPTS == [] and not any(name == p or name.startswith(p + '.')
        for name in sys.modules for p in FORBIDDEN), 'CPU preparation imported a model/native backend')
    return dict(status='PASS_FIXED_REPEATABILITY_CPU_SOURCE_AND_CONFIG_PREPARATION', source_count=len(rows),
        source_lock_ref=ref(root, LOCK), protocol_ref=protocol_ref, prepared_configs=prepared,
        actual_gpu_seconds_before=seconds, actual_gpu_seconds_after=seconds, gpu_ledger_byte_unchanged=True,
        original_gpu_ledger_sha256=expected_ledger_sha256, GPU_operations=0, gpu_authority_issued=False,
        gpu_launch_allowed=False, normal_qualification_passed=False, permits_next_mode=None,
        calibration_refit=False, thresholds_changed=False, forbidden_import_attempts=list(IMPORT_ATTEMPTS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--expected-ledger-sha256', required=True)
    args = parser.parse_args(argv)
    hook = CPUOnlyImports()
    sys.meta_path.insert(0, hook)
    try:
        print(json.dumps(freeze(args.project_root, args.expected_ledger_sha256), sort_keys=True))
    finally:
        sys.meta_path.remove(hook)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
