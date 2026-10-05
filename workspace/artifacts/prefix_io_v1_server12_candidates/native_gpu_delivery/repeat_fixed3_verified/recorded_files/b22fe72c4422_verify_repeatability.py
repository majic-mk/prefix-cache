"""CPU-only replay of the three preregistered fixed-workload off diagnostics.

The original validator is source-pinned and compiled in a fresh namespace after
explicit metadata-only AST changes. No original module, callable, receipt, raw
record, estimator or policy is patched. Recorded threshold coverage is a
diagnostic fact, never a normal-mode predecessor or a performance qualification.
"""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.abc
import importlib.util
import json
from pathlib import Path
import statistics
import sys


D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
OLD = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
SCOPE = 'server12_c5_normal_repeatability_v1'
PROTOCOL_REF = dict(path=D + '/PROTOCOL.json', bytes=5756,
    sha256='ddfcacd283cc9f5fc6d2ec5bd75333c34c8dd956fe02487092b5702c7d420dcd')
V_REF = dict(path=OLD + '/verify_p4_single_file.py', bytes=35880,
    sha256='619b3474a413e8dbeb271e95b394312cb527bb8d1ff7e194c2111441d24bf316')
Q_REF = dict(path=OLD + '/p4_single_file_receipt.py', bytes=17559,
    sha256='3ef359a623d02d9ed220c7e44a0d5fbfa3b4bcbe0b598b8523b800d5aff8a73e')
ANALYSIS_FUNCTIONS = ('require', 'integer', 'migration_gate', 'safe', 'ref',
    'load_module', 'contract', 'accounting_validator', 'validate_native_source_binding',
    'validate_deferral', '_window_io', 'analyze_runtime')
FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'cupy')
IMPORT_ATTEMPTS = []


def require(value, reason):
    if not value:
        raise ValueError('REPEATABILITY_REJECTED: ' + reason)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name)
    return value


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(x not in ('', '.', '..') for x in relative.split('/')),
        'explicit project-relative path required')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink evidence refused')
    require(path.resolve().is_relative_to(root), 'path outside project')
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, 'bounded real source/evidence')
    data = path.read_bytes()
    require(len(data) == path.stat().st_size, 'evidence changed while hashing')
    return dict(path=relative, bytes=len(data), sha256=sha256(data).hexdigest())


def checked(root, expected):
    require(type(expected) is dict and set(expected) == {'path', 'bytes', 'sha256'}
        and type(expected['bytes']) is int and expected['bytes'] > 0
        and type(expected['sha256']) is str and len(expected['sha256']) == 64,
        'exact typed byte reference required')
    require(ref(root, expected['path']) == expected, 'actual source/evidence byte drift: ' + expected['path'])
    return dict(expected)


def read(root, relative):
    reference = ref(root, relative)
    data = safe(root, relative).read_bytes()
    require(len(data) == reference['bytes'] and sha256(data).hexdigest() == reference['sha256'],
        'evidence changed while parsing')
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON field')
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('CPU diagnostic cannot import GPU/model backend: ' + fullname)
        return None


def clean_process():
    require(not IMPORT_ATTEMPTS and not any(name == item or name.startswith(item + '.')
        for name in sys.modules for item in FORBIDDEN), 'clean CPU-only process required')


def validate_protocol_document(p):
    """A CPU schema check only; this cannot validate native execution."""
    require(type(p) is dict, 'protocol object required')
    expected = dict(schema='c5_fixed_workload_repeatability_protocol_v1', scope=SCOPE,
        repetitions=3, mode='off', seed=2829, prompt_first_token=28100,
        cost_upper_ns=16238752, step_budget_ns=13171328, measured_offset=16,
        complete_output_tokens=128, complete_capture_frames=128, ssd_read_operations=1,
        ssd_read_physical_bytes=917504, authority_issued_by_this_document=False,
        native_execution_verified_by_this_document=False, GPU_runs_performed_by_this_document=0)
    require(all(type(p.get(key)) is type(value) and p.get(key) == value for key, value in expected.items()),
        'fixed preregistered protocol differs')
    require(p['budget']['execution_seconds_per_attempt'] == 300
        and type(p['budget']['execution_seconds_per_attempt']) is int
        and p['budget']['cleanup_seconds_per_attempt'] == 20
        and type(p['budget']['cleanup_seconds_per_attempt']) is int
        and p['budget']['planned_max_reserved_seconds'] == 960
        and type(p['budget']['planned_max_reserved_seconds']) is int,
        'original finite three-attempt reservation required')
    require(p['decision_rule']['normal_qualification_passed'] is False
        and p['decision_rule']['permits_next_mode'] is None
        and p['decision_rule']['P4_strategy_effect_verified'] is False
        and p['decision_rule']['performance_benefit_proved'] is False,
        'diagnostic protocol cannot promote any normal mode or benefit')
    require(type(p.get('jobs')) is list and len(p['jobs']) == 3
        and all(type(job.get('diagnostic_index')) is int and job['diagnostic_index'] == index
                and job.get('label') == 'server12-c5-native-repeat-off' + str(index+1).zfill(2)
                for index,job in enumerate(p['jobs'])), 'exact ordered three preregistered slots')
    return p


def prepare_protocol(root):
    """Read and validate the already uploaded preregistration; never issue it."""
    root = Path(root).resolve(strict=True)
    checked(root, PROTOCOL_REF)
    p = validate_protocol_document(read(root, PROTOCOL_REF['path']))
    checked(root, V_REF)
    checked(root, Q_REF)
    prior = p['immutable_prior_off_counterexample']
    for key in ('qualification_ref', 'raw_result_ref', 'guard_ref'):
        checked(root, prior[key])
    q = read(root, prior['qualification_ref']['path'])
    require(q.get('native_execution_verified') is True and q.get('qualification_passed') is False
        and q.get('frozen_cost_migration_pass') is False and q.get('permits_next_mode') is None
        and integer(q['selected_gpu_elapsed_ns'], 'prior actual selected duration', 1) == 16893473
        and integer(q['frozen_cost_upper_ns'], 'prior upper', 1) == p['cost_upper_ns']
        and integer(q['frozen_a_only_budget_ns'], 'prior A-only budget', 1) == p['step_budget_ns']
        and q['evidence_refs']['raw_result'] == prior['raw_result_ref']
        and q['evidence_refs']['actual_guard'] == prior['guard_ref'],
        'actual immutable prior counterexample must remain failed')
    raw = read(root, prior['raw_result_ref']['path'])
    require(raw.get('origin') == 'native_gpu_recording' and raw.get('original_engine_shutdown_returned') is True
        and prior_selected_gpu_ns(raw) == q['selected_gpu_elapsed_ns'],
        'prior actual raw counterexample differs')
    return dict(PROTOCOL_REF)


def prior_selected_gpu_ns(raw):
    """Read the original pinned counterexample's actual Event witness.

    Raw frame GPU durations are collector placeholders. Full C.validate_capture
    remains unchanged in actual repetition replay; this initialization only
    cross-checks the already byte-pinned prior raw against its prior report.
    """
    require(type(raw.get('windows')) is list and len(raw['windows']) == 1, 'one prior actual window')
    capture = raw['windows'][0]['capture']
    frames, witnesses = capture['frames'], capture['event_witnesses']
    require(type(frames) is list and len(frames) == 128 and type(witnesses) is list and len(witnesses) == 128,
        'complete prior frame and actual Event witness lists')
    frame, witness = frames[16], witnesses[16]
    require(type(frame) is dict and type(witness) is dict
        and type(frame.get('native_step_ordinal')) is int and frame['native_step_ordinal'] > 0
        and type(witness.get('native_step_ordinal')) is int
        and witness['native_step_ordinal'] == frame['native_step_ordinal']
        and witness.get('event_elapsed_source') == 'torch.cuda.Event.elapsed_time'
        and witness.get('cross_clock_absolute_mapping') is False,
        'actual prior selected frame and Event witness identity')
    return integer(witness.get('gpu_elapsed_ns'),'actual prior Event GPU elapsed integer',1)


def _functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _dump(node):
    return ast.dump(node, include_attributes=False)


def adapted_validator_tree(original):
    """Only explicit source/domain metadata changes; all scientific ASTs equal."""
    tree = deepcopy(original)
    changes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('DELIVERY', 'SCOPE', 'CANONICAL'):
                before = _dump(node.value)
                value = {'DELIVERY': D, 'SCOPE': SCOPE, 'CANONICAL': Q_REF['path']}[name]
                node.value = ast.Constant(value=value)
                changes.append(dict(kind='literal_domain_metadata', name=name, before=before, after=_dump(node.value)))
    funcs = _functions(tree)
    for name, old_tag, new_tag in (
        ('controller_api', '_server12_normal_mode_controller', '_server12_repeatability_controller'),
        ('notification_adapter', '_server12_c5_cpu_notification_adapter', '_server12_repeatability_notification_adapter')):
        count = 0
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Constant) and node.value == old_tag:
                node.value = new_tag
                count += 1
        require(count == 1, 'exact isolated metadata module cache adaptation')
        changes.append(dict(kind='isolated_module_cache_name', function=name, before=old_tag, after=new_tag))
    count = 0
    for node in ast.walk(funcs['verify_runtime']):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'control' and node.func.attr == 'ref' and len(node.args) == 2):
            value = _dump(node.args[0])
            for phase in ('before', 'after'):
                original_expression = ast.parse("DELIVERY+'/SOURCE_'+mode+'_" + phase.upper() + ".json'", mode='eval').body
                if value == _dump(original_expression):
                    replacement = ast.parse("control.proof_path(actual_config['diagnostic_index'], '" + phase + "')",
                        mode='eval').body
                    node.args[0] = replacement
                    changes.append(dict(kind='indexed_actual_source_proof_path', phase=phase,
                        before=value, after=_dump(replacement)))
                    count += 1
    require(count == 2 and len(changes) == 7, 'exact metadata-only adaptation count')
    original_funcs = _functions(original)
    proof = []
    for name in ANALYSIS_FUNCTIONS:
        require(_dump(original_funcs[name]) == _dump(funcs[name]), 'scientific/accounting AST changed: ' + name)
        proof.append(dict(function=name, identical_AST=True,
            sha256=sha256(_dump(funcs[name]).encode()).hexdigest()))
    allowed = {'controller_api', 'notification_adapter', 'verify_runtime'}
    require(all(_dump(original_funcs[name]) == _dump(node) for name, node in funcs.items() if name not in allowed),
        'nonmetadata original callable changed')
    return ast.fix_missing_locations(tree), dict(original_verifier_ref=V_REF,
        metadata_AST_changes=changes, unchanged_scientific_functions=proof,
        original_module_mutated=False, original_callable_monkeypatched=False)


def validator(root):
    checked(root, V_REF)
    source = safe(root, V_REF['path']).read_bytes()
    tree, proof = adapted_validator_tree(ast.parse(source))
    namespace = dict(__name__='_server12_repeatability_fresh_validator',
        __file__=str(safe(root, D + '/verify_repeatability.py')))
    exec(compile(tree, str(safe(root, V_REF['path'])) + '::explicit_metadata_only', 'exec', dont_inherit=True), namespace)
    require(namespace['PROMPT'] == [28100] + list(range(1001,1129)) and namespace['SEED'] == 2829
        and namespace['BYTES'] == 917504 and namespace['CANONICAL'] == Q_REF['path'],
        'fixed actual fourth workload and real original public canonical required')
    checked(root, V_REF)
    clean_process()
    return namespace, proof


def verify_one(root, diagnostic_index):
    root = Path(root).resolve(strict=True)
    index = integer(diagnostic_index, 'exact planned diagnostic index')
    require(index < 3, 'only the three preregistered slots')
    protocol_ref = prepare_protocol(root)
    protocol = read(root, protocol_ref['path'])
    job = protocol['jobs'][index]
    require(job['diagnostic_index'] == index and type(job['diagnostic_index']) is int,
        'typed immutable planned slot')
    config_path = D + '/' + job['config']
    config = read(root, config_path)
    require(config.get('diagnostic_index') == index and type(config.get('diagnostic_index')) is int
        and config.get('label') == job['label'] and config.get('mode') == 'off'
        and config.get('protocol_ref') == protocol_ref, 'actual config/slot/protocol identity')
    run = 'experiments/prefix_io_v1/runs/' + job['label']
    raw_path = run + '/details/p4-single-file-runtime-result.json'
    guard_path = run + '/result.json'
    raw = read(root, raw_path)
    require(raw.get('diagnostic_index') == index and type(raw.get('diagnostic_index')) is int
        and raw.get('protocol_ref') == protocol_ref and raw.get('label') == job['label'],
        'actual raw diagnostic identity required')
    namespace, source_proof = validator(root)
    # This is the full source/actual authority/guard/ledger/Event/SDK/owner/IO/
    # output/drain replay, not a supplied boolean or a fabricated native object.
    original = namespace['verify_runtime'](root, config_ref=ref(root, config_path),
        result_ref=ref(root, raw_path), guard_ref=ref(root, guard_path),
        before_ref=ref(root, D + '/' + job['source_before']),
        after_ref=ref(root, D + '/' + job['source_after']))
    require(original['native_execution_verified'] is True
        and integer(original['frozen_cost_upper_ns'], 'real public frozen upper', 1) == protocol['cost_upper_ns']
        and integer(original['frozen_a_only_budget_ns'], 'real public A-only budget', 1) == protocol['step_budget_ns']
        and original['full_output_tokens'] == original['full_frames'] == 128
        and raw.get('gpu_uuid') == protocol['current_gpu_uuid']
        and raw['model']['manifest_sha256'] == protocol['model_manifest_sha256'],
        'actual full execution and original calibration limits must replay')
    prior = read(root, protocol['immutable_prior_off_counterexample']['qualification_ref']['path'])
    require(original['output_token_ids'] == prior['output_token_ids']
        and original['native_io']['preload_key_sha256'] == prior['native_io']['preload_key_sha256'],
        'same actual output and payload as the fixed fourth workload')
    before, launch, after = [D + '/' + job[key] for key in ('source_before', 'source_launch', 'source_after')]
    launch_ref = ref(root, launch)
    launch_proof = read(root, launch)
    require(launch_proof.get('phase') == 'launch' and launch_proof.get('failed') == []
        and launch_proof.get('config_ref') == ref(root, config_path)
        and launch_proof.get('source_lock_ref') == original['source_lock_ref'], 'actual full launch source proof')
    lock = read(root, original['source_lock_ref']['path'])
    require(type(launch_proof.get('files_verified')) is int
        and launch_proof['files_verified'] == len(lock['files']), 'actual complete launch source count')
    selected = integer(original['selected_gpu_elapsed_ns'], 'actual selected GPU duration', 1)
    coverage = selected <= protocol['cost_upper_ns']
    require(type(original['frozen_cost_migration_pass']) is bool
        and original['frozen_cost_migration_pass'] is coverage, 'original unchanged migration comparison')
    # Retain all original descriptive details but prevent their normal predecessor
    # flags from escaping into a diagnostic document.
    analysis = dict(original)
    for key in ('runtime_condition_qualified', 'qualification_passed', 'permits_next_mode'):
        analysis.pop(key)
    guard = read(root, guard_path)
    clean_process()
    return dict(schema='c5_fixed_workload_repeatability_record_v1', scope=SCOPE,
        origin='complete_actual_guarded_native_evidence_CPU_replay', diagnostic_index=index,
        label=job['label'], status=('VALID_RECORD_COST_COVERED' if coverage else 'VALID_RECORD_COST_EXCEEDED'),
        native_execution_verified=True, diagnostic_evidence_valid=True, lifecycle_valid=True,
        threshold_coverage_observed=coverage, covered_original_upper=coverage,
        frozen_cost_upper_ns=protocol['cost_upper_ns'], frozen_a_only_budget_ns=protocol['step_budget_ns'],
        selected_gpu_elapsed_ns=selected, upper_exceedance_ns=selected-protocol['cost_upper_ns'],
        A_only_budget_exceedance_ns=selected-protocol['step_budget_ns'],
        full_output_tokens=128, full_capture_frames=128, original_analysis_result=analysis,
        original_semantic_predicate_observed=original['semantic_conditions_passed'],
        actual_guard_elapsed_seconds=guard['elapsed_seconds'], evidence_refs=original['evidence_refs'],
        source_launch_ref=launch_ref, protocol_ref=protocol_ref, validator_source_adaptation=source_proof,
        normal_qualification_passed=False, normal_runtime_condition_qualified=False,
        runtime_condition_qualified=False, qualification_passed=False,
        permits_next_mode=None, P4_strategy_effect_verified=False, performance_benefit_proved=False,
        calibration_refit=False, thresholds_changed=False, GPU_actions_by_verifier=0,
        forbidden_imports=list(IMPORT_ATTEMPTS))


def classify_durations(values, upper):
    """Finite arithmetic only, never an evidence or qualification validator."""
    require(type(values) is list and len(values) == 3, 'exact three arithmetic values')
    for value in values:
        integer(value,'duration integer',1)
    integer(upper,'upper integer',1)
    covered = sum(value <= upper for value in values)
    return ('ALL_THREE_RECORDED_STEPS_COVERED_ONLY' if covered == 3 else
        'ALL_THREE_RECORDED_STEPS_EXCEEDED' if covered == 0 else 'OBSERVED_THRESHOLD_CROSSING_VARIABILITY')


def summarize_records(records, protocol):
    require(type(records) is list and len(records) == 3, 'all three slots must remain in the report')
    valid = [row for row in records if row.get('native_execution_verified') is True]
    values = [integer(row['selected_gpu_elapsed_ns'], 'actual diagnostic selected duration', 1) for row in valid]
    if len(valid) != 3:
        classification = 'INCOMPLETE_OR_INVALID_NO_REPEATABILITY_CONCLUSION'
    else:
        require(all(type(row.get('threshold_coverage_observed')) is bool
            and row['threshold_coverage_observed'] is (row['selected_gpu_elapsed_ns']<=protocol['cost_upper_ns'])
            for row in valid), 'exact replayed coverage booleans')
        classification = classify_durations(values,protocol['cost_upper_ns'])
        require(len({json.dumps(row['original_analysis_result']['binding_ref'],sort_keys=True) for row in valid}) == 1
            and len({json.dumps(row['original_analysis_result']['source_lock_ref'],sort_keys=True) for row in valid}) == 1,
            'do not pool different public bindings or source revisions')
    stats = None
    if len(values) == 3:
        stats = dict(n=3, min_ns=min(values), max_ns=max(values), range_ns=max(values)-min(values),
            median_ns=statistics.median(values), sum_ns=sum(values), mean_numerator_ns=sum(values),
            mean_denominator=3, exact_selected_values_ns=values,
            covered_count=sum(value <= protocol['cost_upper_ns'] for value in values),
            exceeded_count=sum(value > protocol['cost_upper_ns'] for value in values),
            population_or_tail_bound_proved=False, causal_attribution_proved=False)
    return dict(classification=classification, all_three_slots_retained=True, actual_valid_records=len(valid),
        planned_attempts=3, original_prior_off_counterexample_retained=protocol['immutable_prior_off_counterexample'],
        prior_and_new_revisions_pooled=False, descriptive_statistics=stats, records=records,
        normal_qualification_passed=False, runtime_condition_qualified=False, qualification_passed=False,
        permits_next_mode=None, P4_strategy_effect_verified=False, performance_benefit_proved=False,
        new_receipt_issued=False, calibration_refit=False, thresholds_changed=False,
        frozen_cost_upper_ns=protocol['cost_upper_ns'], frozen_a_only_budget_ns=protocol['step_budget_ns'],
        proof_limits=protocol['proof_limits'])


def available_slot_references(root,job):
    run='experiments/prefix_io_v1/runs/'+job['label']
    paths=dict(config=D+'/'+job['config'],actual_guard=run+'/result.json',
        actual_raw=run+'/details/p4-single-file-runtime-result.json',
        source_before=D+'/'+job['source_before'],source_launch=D+'/'+job['source_launch'],
        source_after=D+'/'+job['source_after'])
    found={};missing=[]
    for name,path in paths.items():
        if safe(root,path).exists():
            found[name]=ref(root,path)
        else:
            missing.append(dict(field=name,path=path))
    return dict(actual_available_refs=found,actual_missing_files=missing)


def verify_all(root):
    root = Path(root).resolve(strict=True)
    prepare_protocol(root)
    protocol = read(root, PROTOCOL_REF['path'])
    records = []
    for index in range(3):
        job = protocol['jobs'][index]
        guard_relative = 'experiments/prefix_io_v1/runs/' + job['label'] + '/result.json'
        available=available_slot_references(root,job)
        if not safe(root, guard_relative).exists():
            records.append(dict(diagnostic_index=index, label=job['label'], status='NO_ACTUAL_COMPLETION_GUARD',
                native_execution_verified=False, selected_gpu_elapsed_ns=None, actual_guard_ref=None,
                qualification_passed=False, permits_next_mode=None,**available))
            continue
        guard_reference = ref(root, guard_relative)
        try:
            records.append(verify_one(root, index))
        except Exception as exc:
            records.append(dict(diagnostic_index=index, label=job['label'], status='ACTUAL_EVIDENCE_REPLAY_REJECTED',
                native_execution_verified=False, selected_gpu_elapsed_ns=None, actual_guard_ref=guard_reference,
                reason=str(exc), qualification_passed=False, permits_next_mode=None,**available))
    clean_process()
    return dict(schema='c5_fixed_workload_repeatability_summary_v1', scope=SCOPE,
        origin='CPU_readonly_replay_of_actual_preregistered_diagnostics', protocol_ref=PROTOCOL_REF,
        **summarize_records(records, protocol), GPU_actions_by_verifier=0, forbidden_imports=list(IMPORT_ATTEMPTS))


def verify_repetition(root,diagnostic_index):
    """Supervisor API: actual evidence errors raise; a cost miss is retained."""
    clean_process()
    if not any(isinstance(finder,CPUOnlyImports) for finder in sys.meta_path):
        sys.meta_path.insert(0,CPUOnlyImports())
    return verify_one(root,diagnostic_index)


def summarize(root):
    """Supervisor API: replay all three slots; no normal-mode promotion."""
    clean_process()
    if not any(isinstance(finder,CPUOnlyImports) for finder in sys.meta_path):
        sys.meta_path.insert(0,CPUOnlyImports())
    return verify_all(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--index', type=int, choices=(0,1,2))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    require(args.output.startswith(D + '/') and args.output.endswith('.json'), 'new diagnostic output path required')
    clean_process()
    sys.meta_path.insert(0, CPUOnlyImports())
    result = verify_all(root) if args.index is None else verify_one(root, args.index)
    clean_process()
    output = safe(root, args.output)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output_ref=ref(root,args.output), scope=SCOPE,
        native_execution_verified=result.get('native_execution_verified',False),
        classification=result.get('classification'), permits_next_mode=None, GPU_actions=0),sort_keys=True))
    # A complete valid cost exceedance is a retained diagnostic outcome, not an
    # execution error. Missing/invalid actual evidence is a hard stop.
    return 2 if result.get('classification') == 'INCOMPLETE_OR_INVALID_NO_REPEATABILITY_CONCLUSION' else 0


if __name__ == '__main__':
    raise SystemExit(main())
