"""Replay completed server12 evidence and issue its finite public receipt on CPU.

This posthoc helper is outside the frozen execution revision. It creates no GPU
authority and must run only after the original guard, after audit and --verify.
The output JSON is a receipt summary, not a transferable typed receipt or P4
qualification. Existing evidence and output files are never overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PROJECT = '/root/autodl-tmp/prefix-io-v1-handoff/project'
ENTRY = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
JOB = 'server12-c5-native-common-cost-gpu01'
COMMON = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate'
RUN = 'experiments/prefix_io_v1/runs/' + JOB
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
CANONICAL = ENTRY + '/p4_single_file_receipt.py'
CANONICAL_PIN = dict(path=CANONICAL, bytes=16497,
    sha256='8deb840a759249f339015634bc58adb95e986e7c787ff1b7f71b698fa4ec7a0d')
NATIVE_PIN = dict(path=ENTRY + '/native_conditional_cost.py', bytes=41752,
    sha256='eebf31e14ca2eb7a662c4548adb5a18022b3e5a2bced541fcc1bbf5b0623b64a')
SERIALIZER_PIN = dict(path=ENTRY + '/prepare_and_verify_native_cost.py', bytes=24942,
    sha256='cc777713612ebebf361f4c476440317e97610a0f8a0553cb3d7d33e26ea3ea64')
ORIGINAL_RUNNER = 'third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
IMPORT_ATTEMPTS = []


class MetadataOnlyImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('posthoc CPU evidence replay refuses GPU/model import: ' + fullname)
        return None


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'exact project-relative evidence/output path required')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink evidence/output refused: ' + relative)
    require(path.resolve().is_relative_to(root), 'evidence/output outside project')
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2,
        'bounded actual evidence/source file required: ' + relative)
    raw = path.read_bytes()
    require(len(raw) == path.stat().st_size, 'file changed while reading: ' + relative)
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(root, relative):
    actual = ref(root, relative)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate evidence key')
            result[key] = value
        return result
    raw = safe(root, relative).read_bytes()
    require(len(raw) == actual['bytes'] and hashlib.sha256(raw).hexdigest() == actual['sha256'],
        'evidence changed before parsing: ' + relative)
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite evidence JSON')))


def checked(root, row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
        and type(row['bytes']) is int and row['bytes'] > 0 and type(row['sha256']) is str
        and len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']),
        'exact typed actual file reference')
    require(ref(root, row['path']) == row, 'actual evidence/source SHA drift: ' + row['path'])
    return dict(row)


def load_source(root, expected, name):
    checked(root, expected)
    path = safe(root, expected['path'])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
        checked(root, expected)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


def put_new(root, relative, value):
    path = safe(root, relative)
    require(path.parent.is_dir(), 'existing output directory required')
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    return ref(root, relative)


def qualifies(result):
    require(type(result) is dict, 'actual native verification result required')
    for key in ('native_execution_verified', 'conditional_cost_cell_qualified', 'heldout_covered'):
        require(result.get(key) is True, 'NO_RECEIPT: real native qualification failed: ' + key)
    for key in ('production_qualified', 'strategy_effect_verified', 'holdout_used_to_refit', 'resource_release_credit'):
        require(result.get(key) is False, 'unsupported qualification or refit: ' + key)
    require(type(result.get('calibration_pairs')) is int and result['calibration_pairs'] == 2
        and type(result.get('validation_pairs')) is int and result['validation_pairs'] == 1
        and type(result.get('full_output_tokens')) is int and result['full_output_tokens'] == 768,
        'two calibration pairs, one heldout and 768 real outputs required')
    errors = result.get('heldout_errors')
    require(type(errors) is list and errors and all(type(row) is dict
        and type(row.get('underprediction_ns')) is int and row['underprediction_ns'] == 0 for row in errors),
        'actual heldout underprediction must be zero; no refit or upper override')


def issue(root, binding_relative, summary_relative):
    require(root.as_posix() == PROJECT, 'actual fixed server project required')
    require(binding_relative == ENTRY + '/NATIVE_SINGLE_FILE_BINDING.json'
        and summary_relative == ENTRY + '/REAL_CANONICAL_RECEIPT_SUMMARY.json',
        'only new server12 posthoc output names are allowed')
    require(not safe(root, binding_relative).exists() and not safe(root, summary_relative).exists(),
        'append-only outputs already exist; read the existing result instead')
    require(not any(name == base or name.startswith(base + '.') for name in sys.modules for base in FORBIDDEN),
        'fresh CPU metadata process required')
    sys.meta_path.insert(0, MetadataOnlyImports())

    plan_path = ENTRY + '/NATIVE_COST_PLAN.json'
    measured_path = ENTRY + '/NATIVE_PAIRED_MEASUREMENTS.json'
    verified_path = ENTRY + '/NATIVE_COST_VERIFICATION.json'
    guard_path = RUN + '/result.json'
    source_paths = {phase: ENTRY + '/SOURCE_' + phase.upper() + '_VERIFICATION.json'
                    for phase in ('before', 'after')}
    stable_paths = [plan_path, measured_path, verified_path, guard_path,
        ENTRY + '/PLAN_REFERENCE.json', ENTRY + '/SOURCE_LAUNCH_VERIFICATION.json',
        ENTRY + '/GPU_LAUNCH_INTENT.json', ENTRY + '/SITE_SOURCE_LOCK.json',
        RUN + '/details/native-cost-runtime-result.json', *source_paths.values()]
    actual_refs = {path: ref(root, path) for path in stable_paths}
    ledger_before_ref = ref(root, LEDGER)
    result = read(root, verified_path)
    qualifies(result)  # No output is created when heldout or execution fails.
    plan = read(root, plan_path)
    record = read(root, measured_path)
    require(plan.get('job_id') == JOB and plan.get('evidence_origin') == 'native_runtime_preregistered'
        and plan.get('cpu_preparation_only') is False, 'new actual server12 plan required')
    plan_ref = actual_refs[plan_path]
    measurements_ref = actual_refs[measured_path]
    guard_ref = actual_refs[guard_path]
    require(read(root, ENTRY + '/PLAN_REFERENCE.json') == plan_ref, 'independent preregistered plan pin drift')
    source_refs = {phase: actual_refs[path] for phase, path in source_paths.items()}
    require(result.get('evidence_refs') == dict(plan=plan_ref, measurements=measurements_ref,
        actual_guard=guard_ref, sources=source_refs), 'verification does not bind these actual inputs')
    require(record.get('plan_ref') == plan_ref and record.get('source_verification_refs') == source_refs
        and record.get('origin') == 'native_gpu_recording', 'actual raw measurement provenance required')
    require(plan.get('source_lock_ref') == actual_refs[ENTRY + '/SITE_SOURCE_LOCK.json'],
        'actual completed site source lock drift')
    lock = read(root, ENTRY + '/SITE_SOURCE_LOCK.json')
    require(type(lock.get('files')) is list, 'actual complete site source closure required')
    refs = {}
    for row in lock['files']:
        require(type(row) is dict and type(row.get('path')) is str and row['path'] not in refs,
            'duplicate/unknown site source path')
        refs[row['path']] = row
    require(len(refs) == plan.get('source_lock_files'), 'actual source count mismatch')
    for expected in (CANONICAL_PIN, NATIVE_PIN, SERIALIZER_PIN):
        require(refs.get(expected['path']) == expected, 'new frozen canonical/verifier/serializer pin required')
        checked(root, expected)

    serializer = load_source(root, SERIALIZER_PIN, '_server12_posthoc_actual_serializer')
    reconstructed = serializer.serialize_runtime_record(root,
        runtime_relative=RUN + '/details/native-cost-runtime-result.json',
        plan_ref=plan_ref, source_verification_refs=source_refs)
    require(reconstructed == record, 'record differs from all actual raw parent/child receipts')
    replayed = serializer.C.verify_native_cell(root, plan_ref=plan_ref,
        measurements_ref=measurements_ref, guard_ref=guard_ref,
        expected_plan_ref=plan_ref, expected_guard_ref=guard_ref,
        original_source_path=safe(root, serializer.ORIGINAL))
    qualifies(replayed)
    require(json.dumps(replayed, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')) ==
        json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')),
        'stored verification wire bytes differ from complete actual native replay')
    canonical = load_source(root, CANONICAL_PIN, '_server12_posthoc_actual_public_receipt')
    common_paths = [plan['model_plan_ref']['path'], plan['model_config_ref']['path'],
        plan['cuda_event_source_ref']['path'], ORIGINAL_RUNNER, plan['native_source_ref']['path']]
    overlay_paths = [plan['collector_source_ref']['path'], COMMON + '/single_file_runtime_binding.py',
        CANONICAL, NATIVE_PIN['path'], SERIALIZER_PIN['path'], ENTRY + '/gpu_entry_binding.py',
        ENTRY + '/control_native_cost_job.py', ENTRY + '/run_native_cost_experiment.py',
        ENTRY + '/site_sdk_binding.py']
    require(len(set(common_paths + overlay_paths)) == len(common_paths + overlay_paths),
        'common and overlay references must be distinct')
    common = [checked(root, refs[path]) for path in common_paths]
    overlay = [checked(root, refs[path]) for path in overlay_paths]
    binding = dict(schema_version=1, scope='p4_single_file_native_binding_v1',
        calibration=dict(plan_ref=plan_ref, measurements_ref=measurements_ref, guard_ref=guard_ref),
        runtime_common_refs=common, runtime_overlay_refs=overlay,
        kernel_mode=canonical._kernel_mode(record, root))
    for path, expected in actual_refs.items():
        require(ref(root, path) == expected, 'actual input changed during posthoc replay: ' + path)
    require(ref(root, LEDGER) == ledger_before_ref, 'ledger changed during posthoc replay')
    binding_ref = put_new(root, binding_relative, binding)
    receipt = canonical.load_verified_single_file(root, binding_relative)
    require(type(receipt) is canonical.ExactSingleFileReceipt and receipt.condition_only is True
        and receipt.production_qualified is False and receipt.binding_ref.mapping() == binding_ref,
        'only the actual public canonical typed receipt is accepted')
    require(receipt.cost_upper_ns == replayed['calibration_predicted_upper_ns'],
        'canonical upper differs from original calibration math')
    require(not IMPORT_ATTEMPTS and not any(name == base or name.startswith(base + '.')
        for name in sys.modules for base in FORBIDDEN), 'CPU-only public replay import boundary')
    for path, expected in actual_refs.items():
        require(ref(root, path) == expected, 'actual input changed during public issuance: ' + path)
    require(ref(root, LEDGER) == ledger_before_ref, 'ledger changed during public issuance')
    for row in (CANONICAL_PIN, NATIVE_PIN, SERIALIZER_PIN, *common, *overlay):
        checked(root, row)
    summary = dict(schema='server12_actual_canonical_receipt_summary_v1',
        status='PASS_REAL_CALIBRATION_DOMAIN_TYPED_RECEIPT',
        origin='actual_completed_native_gpu_recording_reverified',
        job=JOB, gpu_uuid=plan['gpu_uuid'], signature=list(receipt.signature),
        cost_upper_ns=receipt.cost_upper_ns, step_budget_ns=receipt.step_budget_ns,
        fits_calibration_a_only_budget=receipt.cost_upper_ns <= receipt.step_budget_ns,
        binding_ref=binding_ref, canonical_source_ref=CANONICAL_PIN,
        verification_ref=actual_refs[verified_path], calibration_refs=binding['calibration'],
        source_verification_refs=source_refs, site_source_lock_ref=plan['source_lock_ref'],
        receipt_python_type=type(receipt).__name__,
        common_source_refs=[row.mapping() for row in receipt.runtime_common_refs],
        overlay_source_refs=[row.mapping() for row in receipt.runtime_overlay_refs],
        native_execution_verified=True, finite_condition_cost_qualified=True,
        full_output_tokens=replayed['full_output_tokens'], heldout_covered=replayed['heldout_covered'],
        holdout_used_to_refit=False, normal_runtime_binding_verified=False,
        actual_on_installed=False, full_runtime_cost_qualified=False,
        production_qualified=False, strategy_effect_verified=False,
        resource_release_credit=False, GPU_operations_this_action=0,
        forbidden_import_attempts=IMPORT_ATTEMPTS, ledger_before_ref=ledger_before_ref,
        ledger_after_ref=ref(root, LEDGER), summary_is_not_the_typed_receipt=True,
        qualification_scope='finite_engineering_condition_not_SLO_or_probability_guarantee')
    summary_ref = put_new(root, summary_relative, summary)
    return dict(status=summary['status'], binding_ref=binding_ref, summary_ref=summary_ref,
        cost_upper_ns=receipt.cost_upper_ns, step_budget_ns=receipt.step_budget_ns,
        fits_calibration_a_only_budget=summary['fits_calibration_a_only_budget'],
        native_execution_verified=True, finite_condition_cost_qualified=True,
        normal_runtime_binding_verified=False, full_runtime_cost_qualified=False,
        actual_on_installed=False, strategy_effect_verified=False,
        GPU_operations_this_action=0, summary_is_not_the_typed_receipt=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--binding', default=ENTRY + '/NATIVE_SINGLE_FILE_BINDING.json')
    parser.add_argument('--summary', default=ENTRY + '/REAL_CANONICAL_RECEIPT_SUMMARY.json')
    args = parser.parse_args(argv)
    try:
        result = issue(args.project.resolve(strict=True), args.binding, args.summary)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='POSTHOC_NATIVE_RECEIPT_REJECTED',
            reason=str(exc), error_type=type(exc).__name__, valid_native_receipt=False,
            native_qualification_claim=False, GPU_operations_this_action=0,
            normal_runtime_binding_verified=False, actual_on_installed=False,
            strategy_effect_verified=False, forbidden_import_attempts=IMPORT_ATTEMPTS),
            ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
