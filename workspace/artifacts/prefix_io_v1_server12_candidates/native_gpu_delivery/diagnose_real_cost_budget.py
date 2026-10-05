"""CPU-only, append-only diagnosis of the completed server12 finite cost cell.

The public receipt is replayed from real evidence. A separate explicitly marked
CPU contract projection calls the unchanged policy on that recorded condition;
its typed observation is not a live capability or an on-mode GPU experiment.
No model, GPU work, authority, alternative candidate or refitted bound is made.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = '/root/autodl-tmp/prefix-io-v1-handoff/project'
ENTRY = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
JOB = 'server12-c5-native-common-cost-gpu01'
COMMON = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate'
PACKAGE = COMMON + '/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
Q = ENTRY + '/p4_single_file_receipt.py'
Q_PIN = dict(path=Q, bytes=16497,
    sha256='8deb840a759249f339015634bc58adb95e986e7c787ff1b7f71b698fa4ec7a0d')
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
IMPORT_ATTEMPTS = []
LOADED_REFS = {}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def integer(value, reason, minimum=0):
    require(type(value) is int and value >= minimum, reason)
    return value


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'project-relative path required')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink evidence refused')
    require(path.resolve().is_relative_to(root), 'path outside project')
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, 'bounded actual source/evidence')
    raw = path.read_bytes()
    require(len(raw) == path.stat().st_size, 'file changed while hashing')
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(root, expected):
    require(type(expected) is dict and set(expected) == {'path', 'bytes', 'sha256'}
        and type(expected['bytes']) is int and expected['bytes'] > 0
        and type(expected['sha256']) is str and len(expected['sha256']) == 64,
        'exact typed source/evidence reference required')
    require(ref(root, expected['path']) == expected, 'actual byte drift: ' + expected['path'])
    return dict(expected)


def read(root, relative):
    expected = ref(root, relative)
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    raw = safe(root, relative).read_bytes()
    require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
        'actual evidence changed while parsing')
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite evidence')))


class FrozenContractLoader:
    def __init__(self, root, row):
        self.root, self.row = root, row

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        checked(self.root, self.row)
        path = safe(self.root, self.row['path'])
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
        checked(self.root, self.row)
        LOADED_REFS[self.row['path']] = self.row


class FrozenMetadataFinder:
    def __init__(self, root, refs):
        self.root, self.refs = root, refs

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == base or fullname.startswith(base + '.') for base in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('CPU diagnostic refuses GPU/model import: ' + fullname)
        if fullname != 'prefix_io_control' and not fullname.startswith('prefix_io_control.'):
            return None
        if fullname == 'prefix_io_control':
            relative, package = PACKAGE + '/__init__.py', True
        elif fullname == 'prefix_io_control.p4_single_file_receipt':
            relative, package = Q, False
        else:
            relative, package = PACKAGE + '/' + fullname.removeprefix('prefix_io_control.').replace('.', '/') + '.py', False
        require(relative in self.refs, 'contract source missing from actual site closure: ' + relative)
        row = checked(self.root, self.refs[relative])
        return importlib.util.spec_from_file_location(fullname, safe(self.root, relative),
            loader=FrozenContractLoader(self.root, row),
            submodule_search_locations=[str(safe(self.root, PACKAGE))] if package else None)


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def ceil_mean(values):
    require(values and all(type(v) is int for v in values), 'actual integer samples required')
    return (sum(values) + len(values) - 1) // len(values)


def measured_components(window, offset):
    capture = window['capture']
    require(len(capture['frames']) == len(capture['event_witnesses']) == 128, '128 actual frames/witnesses')
    frame, event = capture['frames'][offset], capture['event_witnesses'][offset]
    host = integer(frame['end_ns'], 'actual frame end', 1) - integer(frame['start_ns'], 'actual frame start', 1)
    gpu = integer(event['gpu_elapsed_ns'], 'actual CUDA full-step duration', 1)
    actions = capture['actions']
    require(type(actions) is list and len(actions) == 1, 'one recorded diagnostic action')
    action = actions[0]
    trigger = integer(action['trigger_after_ns'], 'trigger end', 1) - integer(action['trigger_before_ns'], 'trigger start', 1)
    io = []
    if window['arm'] == 'action':
        require(action.get('action_enabled') is True, 'actual diagnostic action required')
        accepted = [e for e in window['native_journal']['events'] if e['kind'] == 'accepted'
            and event['start_completed_query_ns'] < e['at_ns'] < event['end_record_before_ns']]
        require(len(accepted) == 1 and accepted[0]['stage'] == 'ssd_read', 'one actual SSD operation')
        for start in accepted:
            ends = [e for e in window['native_journal']['events'] if e['kind'] == 'completed'
                and e['operation_sequence'] == start['operation_sequence']]
            require(len(ends) == 1, 'unique actual completion')
            end = ends[0]
            io.append(dict(stage=start['stage'], operations=1, physical_bytes=start['physical_bytes'],
                accepted_at_ns=start['at_ns'], completed_at_ns=end['at_ns'],
                observed_owner_completion_latency_ns=end['at_ns'] - start['at_ns'],
                cqe_observed_after_selected_host_frame=end['at_ns'] > frame['end_ns']))
    else:
        require(action.get('action_enabled') is False, 'actual baseline without action')
    return dict(selected_offset=offset, gpu_full_step_ns=gpu, host_closed_frame_ns=host,
        trigger_host_interval_ns=trigger, native_io=io,
        host_intervals_are_not_isolated_gpu_or_ssd_cost=True,
        CUDA_and_host_absolute_clocks_are_not_mapped=True)


def policy_projection(window, receipt):
    from prefix_io_control.p4_policy import P4Policy
    from prefix_io_control.p4_types import P4Config, SystemSnapshot, WorkDescriptor
    from prefix_io_control.dispatch_shadow import ShadowState
    from prefix_io_control.dispatch_budget import ZERO
    frame = window['capture']['frames'][16]
    event = window['capture']['event_witnesses'][16]
    action = window['capture']['actions'][0]
    now = action['trigger_before_ns']
    prior = [row for row in window['native_journal']['frames'] if row['captured_ns'] <= now]
    require(prior, 'actual pre-trigger owner snapshot required')
    owner = prior[-1]
    require(all(type(row.get('ops')) is int and row['ops'] == 0
        and type(row.get('nbytes', row.get('bytes'))) is int
        and row.get('nbytes', row.get('bytes')) == 0 for row in owner['inflight']),
        'actual recorded existing I/O must be zero')
    epoch = integer(frame['native_step_ordinal'], 'actual scalar ordinal')
    age = max(1, now - owner['captured_ns'], now - event['start_completed_query_ns'])
    config = P4Config(mode='interference', sample_max_age_ns=age, max_wait_ns=1,
        internal_step_budget_ns=receipt.step_budget_ns, candidate_batches=(1,))
    # These are CPU scalar contract projections over the recorded action, not
    # owner-issued live capabilities. The actual calibration bridge was None.
    snapshot = SystemSnapshot(run_id=JOB, snapshot_epoch=epoch, monotonic_ns=now,
        capabilities=frozenset(('native_ready_work', 'conditional_single_file_live_step')),
        native_state=ShadowState(JOB, owner['captured_ns'], inflight=ZERO),
        load_signature=receipt.signature)
    keys = window['independent_payload']['preload_key_sha256']
    require(len(keys) == 1, 'only the actual single-file key may be projected')
    work = WorkDescriptor(run_id=JOB, snapshot_epoch=epoch, parent_id=0,
        child_id='preload:' + keys[0], stage='ssd_read', nbytes=917504,
        native_order=0, created_ns=now, submitted=False, minimum_unit_bytes=917504)
    policy = P4Policy(JOB, config, single_file=receipt)
    state = (epoch, event['start_record_before_ns'], event['start_completed_query_ns'], 1, 1, 0, 144)
    preview = policy.issue_preview(work, snapshot, now_ns=now, expected_epoch=epoch,
        execution='production', _single_file_state=state)
    require(preview.action == ('issue' if receipt.cost_upper_ns <= receipt.step_budget_ns else 'defer')
        and preview.reason == 'verified_single_file_experimental_condition'
        and preview.predicted_total_ns == receipt.cost_upper_ns and preview.production_qualified is False,
        'unchanged actual finite policy disagrees with verified cell')
    outside = policy.issue_preview(work, snapshot, now_ns=now, expected_epoch=epoch,
        execution='cpu_mock', _single_file_state=state)
    require(outside.action == 'native_fallback' and outside.reason == 'outside_verified_single_file_condition',
        'CPU contract replay must not grant native execution')
    return dict(scope='POSTHOC_RECORDED_CONDITION_CPU_POLICY_REPLAY',
        preview=dataclasses.asdict(preview), native_CPU_execution_rejection=dataclasses.asdict(outside),
        recorded_actual_preload_key_sha256=keys[0], contract_config=dataclasses.asdict(config),
        scalar_values_derived_from_actual_capture=True,
        capability_flags_and_ready_work_object_are_CPU_contract_projections=True,
        live_owner_snapshot_or_capability_qualified=False,
        actual_policy_installed_on_GPU=False, actual_native_dispatch_affected=False,
        native_execution_authorized_by_replay=False, candidate_count=1,
        unknown_alternative_candidates='reject_no_calibrated_cell')


def diagnose(root, output):
    require(root.as_posix() == PROJECT, 'fixed actual server project required')
    require(output == ENTRY + '/REAL_COST_BUDGET_DIAGNOSIS.json', 'new append-only diagnostic output only')
    require(not safe(root, output).exists(), 'existing diagnostic must not be overwritten')
    require(not any(name == base or name.startswith(base + '.') for name in sys.modules
        for base in (*FORBIDDEN, 'prefix_io_control')), 'fresh CPU evidence process required')
    paths = {name: ENTRY + '/' + filename for name, filename in (
        ('verification', 'NATIVE_COST_VERIFICATION.json'), ('summary', 'REAL_CANONICAL_RECEIPT_SUMMARY.json'),
        ('binding', 'NATIVE_SINGLE_FILE_BINDING.json'), ('plan', 'NATIVE_COST_PLAN.json'),
        ('measurements', 'NATIVE_PAIRED_MEASUREMENTS.json'), ('site_lock', 'SITE_SOURCE_LOCK.json'))}
    input_refs = {name: ref(root, path) for name, path in paths.items()}
    ledger_ref = ref(root, LEDGER)
    result, summary, binding, plan, record, lock = (read(root, paths[name]) for name in paths)
    require(plan.get('job_id') == JOB and record.get('origin') == 'native_gpu_recording', 'actual server12 data only')
    for flag in ('native_execution_verified', 'conditional_cost_cell_qualified', 'heldout_covered'):
        require(result.get(flag) is True, 'actual native conditional cell failed: ' + flag)
    require(result.get('holdout_used_to_refit') is False and result.get('production_qualified') is False
        and result.get('full_output_tokens') == 768 and summary.get('summary_is_not_the_typed_receipt') is True,
        'finite no-refit evidence boundary')
    require(summary.get('binding_ref') == input_refs['binding'] and summary.get('verification_ref') == input_refs['verification']
        and summary.get('canonical_source_ref') == Q_PIN and plan.get('source_lock_ref') == input_refs['site_lock'],
        'actual binding/result/source provenance')
    refs = {}
    for row in lock['files']:
        require(type(row) is dict and row['path'] not in refs, 'duplicate source path')
        refs[row['path']] = row
    require(refs.get(Q) == Q_PIN, 'new actual canonical source pin')
    sys.meta_path.insert(0, FrozenMetadataFinder(root, refs))
    from prefix_io_control.p4_single_file_receipt import ExactSingleFileReceipt, load_verified_single_file
    receipt = load_verified_single_file(root, paths['binding'])
    require(type(receipt) is ExactSingleFileReceipt and receipt.production_qualified is False,
        'actual public receipt required')
    require(receipt.cost_upper_ns == result['calibration_predicted_upper_ns'] == summary['cost_upper_ns']
        and receipt.step_budget_ns == summary['step_budget_ns'], 'stored and public numbers differ')
    require(record['plan_ref'] == input_refs['plan'] and len(record['windows']) == len(result['windows']) == 6,
        'all six actual windows required')
    raw = {(w['pair_id'], w['arm']): w for w in record['windows']}
    selected = {(w['pair_id'], w['arm']): w for w in result['windows']}
    require(len(raw) == len(selected) == 6, 'duplicate actual window')
    pair_rows = []
    calibration_a, calibration_b = [], []
    for entry in plan['entries']:
        pid = entry['pair_id']
        a, b = selected[(pid, 'baseline')]['selected_gpu_elapsed_ns'], selected[(pid, 'action')]['selected_gpu_elapsed_ns']
        integer(a, 'actual A selected GPU duration', 1); integer(b, 'actual B selected GPU duration', 1)
        components = {arm: measured_components(raw[(pid, arm)], plan['measured_offset'])
                      for arm in ('baseline', 'action')}
        require(components['baseline']['gpu_full_step_ns'] == a and components['action']['gpu_full_step_ns'] == b,
            'actual raw/verified selected duration differs')
        pair_rows.append(dict(pair_id=pid, split=entry['split'], arm_order=entry['arm_order'],
            seed=entry['seed'], selected_offset=plan['measured_offset'], A_gpu_ns=a, B_gpu_ns=b,
            paired_B_minus_A_ns=b-a, B_minus_A_only_budget_ns=b-receipt.step_budget_ns,
            upper_minus_B_ns=receipt.cost_upper_ns-b, measured_components=components))
        if entry['split'] == 'calibration':
            calibration_a.append(a); calibration_b.append(b)
    require(len(calibration_a) == len(calibration_b) == 2, 'two calibration pairs only')
    cost = result['calibration_only_cost']
    baseline = integer(cost['baseline_ns'], 'actual original baseline', 1)
    increment = integer(cost['incremental_or_joint_ns'], 'actual original increment')
    residual = integer(cost['uncertainty_ns'], 'actual original residual')
    require(baseline == ceil_mean(calibration_a)
        and increment == max(0, ceil_mean([b-a for a,b in zip(calibration_a,calibration_b)]))
        and residual == max(0, max(b-baseline-increment for b in calibration_b))
        and baseline + increment + residual == receipt.cost_upper_ns,
        'original frozen numerical expressions disagree with actual samples')
    require(receipt.step_budget_ns == baseline + max(0, max(a-baseline for a in calibration_a)) == max(calibration_a),
        'A-only budget must use calibration A only')
    policy = policy_projection(raw[(plan['entries'][0]['pair_id'], 'action')], receipt)
    diagnosis = dict(schema='server12_actual_cost_budget_diagnosis_v1', status='PASS_CPU_REAL_EVIDENCE_DIAGNOSIS',
        scope='finite_recorded_single_file_condition_only', origin='actual_completed_native_gpu_evidence_readonly_replay',
        job=JOB, gpu_uuid=plan['gpu_uuid'], input_refs=input_refs,
        calibration_refs=binding['calibration'], public_canonical_source_ref=Q_PIN,
        signature=list(receipt.signature), pairs=pair_rows,
        frozen_math=dict(formula=result['formula'], original_estimator_proof=result['original_estimator_proof'],
            baseline_ns=baseline, incremental_or_joint_ns=increment, uncertainty_ns=residual,
            upper_ns=receipt.cost_upper_ns, A_only_budget_ns=receipt.step_budget_ns,
            A_only_baseline_slack_ns=receipt.step_budget_ns-baseline,
            upper_minus_budget_ns=receipt.cost_upper_ns-receipt.step_budget_ns,
            gap_decomposition_ns=dict(increment=increment, residual=residual,
                subtract_A_only_baseline_slack=receipt.step_budget_ns-baseline),
            fits_calibration_A_only_budget=receipt.cost_upper_ns<=receipt.step_budget_ns,
            holdout_used_to_refit=False, action_or_heldout_used_for_budget=False,
            numerical_expressions_changed=False),
        native_conditional_cost_qualified=True, full_output_tokens=result['full_output_tokens'],
        heldout_errors=result['heldout_errors'], finite_candidate_policy=policy,
        source_loaded_refs=[LOADED_REFS[key] for key in sorted(LOADED_REFS)],
        interpretation=dict(measured_full_step_includes_combined_action_path_and_variation=True,
            isolated_SSD_or_CPU_interference_cost_qualified=False,
            full_runtime_observation_cost_measured=False, causal_subcomponent_attribution_proved=False,
            alternative_size_load_or_release_credit_qualified=False),
        actual_on_installed=False, normal_runtime_binding_verified=False,
        full_runtime_cost_qualified=False, production_qualified=False,
        strategy_effect_verified=False, performance_benefit_proved=False,
        GPU_operations_this_action=0, forbidden_import_attempts=IMPORT_ATTEMPTS,
        ledger_before_ref=ledger_ref, ledger_after_ref=ref(root, LEDGER),
        outputs_do_not_authorize_GPU_or_native_dispatch=True)
    require(not IMPORT_ATTEMPTS and not any(name == base or name.startswith(base+'.') for name in sys.modules
        for base in FORBIDDEN), 'CPU-only import boundary')
    for name, expected in input_refs.items():
        require(ref(root, paths[name]) == expected, 'actual input changed during diagnosis')
    for row in LOADED_REFS.values():
        checked(root, row)
    require(ref(root, LEDGER) == ledger_ref, 'ledger changed during diagnosis')
    path = safe(root, output)
    require(path.parent.is_dir(), 'existing GPU audit directory required')
    with path.open('xb') as stream:
        stream.write((json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    return dict(status=diagnosis['status'], output_ref=ref(root, output),
        upper_ns=receipt.cost_upper_ns, A_only_budget_ns=receipt.step_budget_ns,
        upper_minus_budget_ns=receipt.cost_upper_ns-receipt.step_budget_ns,
        source_policy_preview=policy['preview'], replay_is_CPU_contract_projection=True,
        actual_on_installed=False, performance_benefit_proved=False, GPU_operations_this_action=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--output', default=ENTRY + '/REAL_COST_BUDGET_DIAGNOSIS.json')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(diagnose(args.project.resolve(strict=True), args.output), ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='REAL_COST_BUDGET_DIAGNOSIS_REJECTED', reason=str(exc),
            error_type=type(exc).__name__, GPU_operations_this_action=0,
            actual_on_installed=False, performance_benefit_proved=False,
            forbidden_import_attempts=IMPORT_ATTEMPTS), ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
