"""Append-only construction of the thin V3 adapter; never runs a model."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREP_BASE = HERE.parent.parent
PREP = PREP_BASE / 'gpu_prerental_preparation_20261004'
if not PREP.is_dir():
    PREP = PREP_BASE / 'server12-gpu-prerental-preparation-20261004'
source = (PREP / 'runner/native_runtime_v2.py').read_text(encoding='utf-8')

helpers = r'''
def _load_original_reserve_join(root, finite_activation, *, driver):
    """Load the unchanged pure read-only verifier and restore its import alias."""
    value = finite_activation['descriptor']
    join_ref = value['reserve_join_source_ref']
    host_relative = Path(join_ref['path']).with_name('host_control_observer.py').as_posix()
    refs = finite_activation['runtime_refs']
    driver.require(refs.get(join_ref['path']) == join_ref and host_relative in refs,
                   'actual original reserve join and host dependency frozen')
    previous, present = sys.modules.get('host_control_observer'), 'host_control_observer' in sys.modules
    try:
        host = driver.load(root, refs[host_relative], '_formal_read_only_original_host_' + str(time.monotonic_ns()))
        sys.modules['host_control_observer'] = host
        join = driver.load(root, join_ref, '_formal_read_only_original_reserve_' + str(time.monotonic_ns()))
    finally:
        if present:
            sys.modules['host_control_observer'] = previous
        else:
            sys.modules.pop('host_control_observer', None)
    return join


def _replay_original_reserve(root, finite_activation, table, issuer, *, driver):
    """Return original raw replay JSON; a JSON receipt is never a private table."""
    value = finite_activation['descriptor']
    join = _load_original_reserve_join(root, finite_activation, driver=driver)
    def absolute(row):
        driver.check_ref(root, row)
        return dict(row, path=(root / row['path']).as_posix())
    receipt = driver.read(driver.check_ref(root, value['development_reserve_ref']))
    source = driver.project_ref(root, receipt['native_result_ref'])
    driver.require(finite_activation['runtime_refs'].get(source['path']) == source,
                   'actual original reserve source inherited exactly')
    replay = join.verify_existing_native_reserve(root,
        reserve_source_ref=absolute(source), reserve_receipt_ref=absolute(value['development_reserve_ref']),
        native_verification_ref=absolute(value['reserve_verification_ref']),
        plan_ref=absolute(value['development_plan_ref']), expected_plan_ref=absolute(value['development_plan_ref']),
        guard_ref=absolute(value['development_guard_ref']), expected_guard_ref=absolute(value['development_guard_ref']),
        observation_ref=absolute(value['development_observation_ref']),
        native_result_ref=absolute(value['development_native_result_ref']),
        qualified_controller_table=table, issuer=issuer)
    driver.require(type(replay) is dict and replay.get('schema') == 'existing_actual_native_development_reserve_verified_v1' and
                   replay.get('status') == 'PASS_READ_ONLY_NATIVE_RESERVE_REPLAY' and
                   replay.get('actual_native_gpu_run') is True and replay.get('synthetic_fixture') is False and
                   replay.get('native_qualification_verifier_required_separately') is False and
                   replay.get('frozen_budget') == finite_activation['independent_budget'] and
                   replay.get('new_qualified_receipts_written') == 0,
                   'unchanged actual read-only native reserve replay; no CPU algebra or new authority')
    return replay


def preflight_formal_reserve_replay(root, gates, *, driver):
    """CPU-only actual private issuer + original reserve replay, then discard authority.

    Used equally for formal Uoff and Ion. It never returns a table and never
    imports a model/native backend. The guarded runtime reissues its own table.
    Existing policy imports are rejected instead of aliased or replaced.
    """
    config, refs = gates['config'], gates['refs']
    driver.require((config.get('phase'), config.get('mode'), config.get('arm')) in
                   (('effect', 'off', 'U'), ('effect', 'on', 'I')),
                   'formal effect Uoff/Ion reserve replay only')
    finite = gates.get('finite_activation')
    driver.require(type(finite) is dict and finite.get('phase') == 'effect' and
                   type(finite.get('descriptor')) is dict and type(finite.get('independent_budget')) is dict and
                   type(config.get('activation_ref')) is dict,
                   'real effect activation and independently measured development reserve required')
    required = ('development_reserve_ref', 'development_budget_ref', 'reserve_verification_ref',
                'reserve_join_source_ref', 'development_plan_ref', 'development_guard_ref',
                'development_observation_ref', 'development_native_result_ref')
    for key in required:
        row = finite['descriptor'].get(key)
        driver.require(type(row) is dict and refs.get(row.get('path')) == row,
                       'missing actual development prerequisite before policy import: ' + key)
        driver.check_ref(root, row)
    activation_path = Path(config['runtime_ref']['path']).with_name('activation_request.py').as_posix()
    driver.require(activation_path in refs, 'original finite activation verifier frozen')
    activation = driver.load(root, refs[activation_path], '_formal_CPU_original_activation_' + str(time.monotonic_ns()))
    actual = activation.verify(root, config['activation_ref'], refs=refs, pair=gates['pair'],
                               gpu_uuid=config['gpu_uuid'], driver=driver, phase='effect')
    for key in ('descriptor', 'independent_budget', 'activation_source_path', 'calibration_plan',
                'phase', 'runtime_refs', 'expected_common_domain_sha256'):
        driver.require(actual.get(key) == finite.get(key), 'fresh actual finite source gate differs: ' + key)
    driver.require(not any(name == 'prefix_io_control' or name.startswith('prefix_io_control.')
                           for name in sys.modules), 'fresh sole CPU private policy verifier package')
    before_modules, before_path = set(sys.modules), list(sys.path)
    try:
        sys.path.insert(0, str(root / actual['activation_source_path']))
        table, issuer = activation.issue(root, actual, driver=driver)
        replay = _replay_original_reserve(root, actual, table, issuer, driver=driver)
        # JSON only; no private issuer/table instance escapes this CPU boundary.
        json.dumps(replay, allow_nan=False)
        return replay
    finally:
        sys.path[:] = before_path
        for name in set(sys.modules) - before_modules:
            if name == 'prefix_io_control' or name.startswith('prefix_io_control.'):
                sys.modules.pop(name, None)


def preflight_runtime_route(root, gates, *, driver):
    """Repeat formal CPU gates before output creation, framework or model import."""
    config = gates['config']
    role = config.get('phase'), config.get('mode'), config.get('arm')
    allowed = {('qualification', 'off', 'U'), ('shadow', 'shadow', 'I'),
               ('development', 'shadow', 'I'), ('effect', 'on', 'I'),
               ('development', 'off', 'U'), ('effect', 'off', 'U')}
    driver.require(role in allowed, 'bounded unchanged native U/I runtime role')
    formal = gates.get('workload', {}).get('schema') == 'natural_trace_workload_v1'
    if formal:
        driver.require(role in {('development', 'off', 'U'), ('development', 'shadow', 'I'),
                               ('effect', 'off', 'U'), ('effect', 'on', 'I')} and
                       type(config.get('formal_trace_binding_ref')) is dict,
                       'actual formal input source closure before runtime')
        checked = driver.preflight_formal_runtime(root, gates)
        driver.require(type(checked) is dict and checked.get('schema') == 'formal_trace_phase_CPU_preflight_v1' and
                       checked.get('role') == list(role) and checked.get('gpu_eligible') is False and
                       checked.get('formal_goodput_allowed') is False and
                       gates.get('formal_phase_preflight') == checked,
                       'fresh unchanged formal phase CPU replay before model')
    else:
        driver.require(role not in {('development', 'off', 'U'), ('effect', 'off', 'U')} and
                       config.get('formal_trace_binding_ref') is None,
                       'new U baseline role requires actual formal input, never qualification relabeling')
    controller = config['phase'] in ('development', 'effect') and config['arm'] == 'I'
    if config['arm'] == 'U':
        policy = gates['pair']['U']['engine']['kv_transfer_config']['kv_connector_extra_config']['prefix_io_p4_policy']
        driver.require(policy.get('mode') == 'off' and policy.get('cost_table') is None,
                       'real unchanged U off config; no finite controller injection')
    return dict(formal=formal, role=list(role), install_finite_controller=controller,
                require_private_table=config['phase'] in ('development', 'effect'))


def prepare_finite_controller(root, gates, *, native_module, table, issuer, binding_module, driver):
    """U returns before loading or constructing any optional I controller."""
    config, finite = gates['config'], gates['finite_activation']
    if config['arm'] == 'U':
        driver.require(config['mode'] == 'off' and config['phase'] in ('development', 'effect'),
                       'formal U controller branch is strictly off')
        return None
    driver.require(config['arm'] == 'I' and
                   (config['phase'], config['mode']) in (('development', 'shadow'), ('effect', 'on')),
                   'same finite development-shadow or effect-on I controller')
    startup = driver.load(root, finite['descriptor']['startup_ref'],
                          '_strong_preconstruction_finite_' + str(time.monotonic_ns()))
    reactor_module = importlib.import_module('py_kvcache.reactor')
    return startup.FiniteStartup(native_module=native_module, reactor_module=reactor_module,
        table=table, issuer=issuer, run_id=config['run_id'],
        budget_ns=finite['independent_budget']['internal_step_budget_ns'], root=root,
        runtime_refs=gates['refs'], driver=driver, binding_module=binding_module,
        validate_drained=validate_drained_snapshot, observation_only=config['phase'] == 'development',
        reserve_covered_cell_signatures=finite.get('reserve_covered_cell_signatures'))


def verify_formal_off_native_prerequisite(root, prerequisite, *, config, refs, driver):
    """Development off prerequisite remains an actual U-only native proof."""
    driver.require(prerequisite.get('arm') == 'U' and prerequisite.get('mode') == 'off',
                   'actual formal Uoff native prerequisite only')
    return verify_formal_peer_native_prerequisite(root, prerequisite, config=config, refs=refs, driver=driver)


def verify_formal_peer_native_prerequisite(root, prerequisite, *, config, refs, driver):
    """Re-read complete closed native U/I bytes; preserve prospective AB/BA order."""
    role = prerequisite.get('phase'), prerequisite.get('mode'), prerequisite.get('arm')
    driver.require(role in {('development', 'off', 'U'), ('effect', 'off', 'U'), ('effect', 'on', 'I')} and
                   role == (config.get('phase'), config.get('mode'), config.get('arm')),
                   'same actual formal native peer role and original guarded config')
    names = ('actual_request_outputs_ref', 'actual_original_full_step_capture_ref')
    actual = []
    for name in names:
        row = prerequisite.get(name)
        driver.require(type(row) is dict and refs.get(row.get('path')) == row,
                       'closed actual formal native peer evidence required: ' + name)
        actual.append(driver.read(driver.check_ref(root, row)))
    frontend, capture = actual
    driver.require(frontend == prerequisite.get('frontend') and
                   capture == prerequisite.get('full_original_step_capture'),
                   'actual raw formal native peer bytes match guard-closed native result')
    policy = prerequisite.get('strategy_runtime')
    driver.require(type(policy) is dict, 'actual closed native peer strategy metadata')
    if role[2] == 'U':
        driver.require(policy.get('bridge_is_none') is True and policy.get('bridge_mode') is None and
                       policy.get('real_interference_table_installed') is False and
                       policy.get('optional_I_controller_constructed') is False and
                       policy.get('actual_I_strategy_activated') is False and policy.get('native_U_preserved') is True,
                       'actual original formal U stayed off without I controller')
    else:
        driver.require(policy.get('bridge_is_none') is False and policy.get('bridge_mode') == 'interference' and
                       policy.get('real_interference_table_installed') is True and
                       policy.get('optional_I_controller_constructed') is True and
                       policy.get('actual_I_strategy_activated') is True and policy.get('native_U_preserved') is False and
                       policy.get('real_private_cost_table_reissued_for_identity') is True and
                       prerequisite.get('finite_running_identity_verified') is True and
                       prerequisite.get('finite_startup_restored_after_original_shutdown') is True,
                       'actual finite I peer requires original private table identity and restored startup')
        installed = prerequisite.get('finite_current_owner_install')
        driver.require(type(installed) is dict and installed.get('owner_thread_install') is True and
                       installed.get('before_first_request') is True and installed.get('drained_before_attach') is True,
                       'actual I native owner attachment proof required')
    driver.require(prerequisite.get('original_engine_shutdown_returned') is True and
                   prerequisite.get('native_tail_drained') is True and
                   verify_original_tail(prerequisite.get('post_original_shutdown')),
                   'actual original native peer shutdown tail revalidated')
    source = prerequisite.get('complete_observation_config', {}).get('source_ref')
    driver.require(type(source) is dict and refs.get(source.get('path')) == source and
                   source.get('sha256') == 'a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0' and
                   source.get('bytes') == 13016,
                   'same actual complete original U/I observation collector source')
    driver.check_ref(root, source)
    descriptor = driver.read(driver.check_ref(root, config['activation_ref']))
    finite = dict(descriptor=descriptor, runtime_refs=refs)
    if role[2] == 'I':
        pair_row = config.get('pair_config_ref')
        driver.require(type(pair_row) is dict and refs.get(pair_row.get('path')) == pair_row,
                       'actual I peer original strong pair source closed')
        pair = driver.read(driver.check_ref(root, pair_row))['configurations']
        activation_path = Path(config['runtime_ref']['path']).with_name('activation_request.py').as_posix()
        driver.require(activation_path in refs, 'actual I peer original activation source frozen')
        activation = driver.load(root, refs[activation_path], '_formal_peer_original_activation_' + str(time.monotonic_ns()))
        finite = activation.verify(root, config['activation_ref'], refs=refs, pair=pair,
                                   gpu_uuid=config['gpu_uuid'], driver=driver, phase='effect')
        preflight_formal_reserve_replay(root, dict(config=config, refs=refs, pair=pair, finite_activation=finite), driver=driver)
        driver.require(prerequisite.get('engine_config') == pair['I']['engine'],
                       'actual I peer used the unchanged frozen I engine configuration')
    original = dict(config=dict(run_id=prerequisite['run_id']), finite_activation=finite)
    checked = validate_formal_capture(root, original, capture, frontend, source_ref=source, driver=driver)
    return dict(schema='formal_native_peer_CPU_replay_v1', peer_role=list(role),
                actual_raw_sources_revalidated=True, original_capture_replay=checked,
                original_native_tail_revalidated=True, cpu_metadata_only=True,
                gpu_eligible=False, formal_goodput_allowed=False, actual_GPU_operations=0)


def validate_formal_capture(root, gates, capture, frontend, *, source_ref, driver):
    """Delegate complete real frames/output/causal checks to original reserve verifier."""
    driver.require(type(capture) is dict and capture.get('valid') is True and
                   type(capture.get('frames')) is list and capture['frames'],
                   'formal whole-stream CUDA observation UNKNOWN; never drop unsupported frames')
    frames = capture['frames']
    ordinals = [row.get('native_step_ordinal') for row in frames]
    driver.require(all(type(value) is int and value >= 0 for value in ordinals) and
                   ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))),
                   'unchanged complete original native ordinal stream')
    join = _load_original_reserve_join(root, gates['finite_activation'], driver=driver)
    return join._capture(capture, frontend, run_id=gates['config']['run_id'],
                         expected_ordinals=ordinals, source_ref=source_ref)

'''

def replace_once(old, new):
    global source
    assert source.count(old) == 1, old[:120]
    source = source.replace(old, new, 1)

replace_once('def execute(root, gates, reservation, *, driver):', helpers + '\ndef execute(root, gates, reservation, *, driver):')
replace_once('    collector_source_ref = preflight_collector_binding(root, gates, driver=driver)',
             '    route = preflight_runtime_route(root, gates, driver=driver)\n    collector_source_ref = preflight_collector_binding(root, gates, driver=driver)')
replace_once('                  natural_trace_bound=gates["workload"]["schema"] == "natural_off_qualification_workload_v1",',
             '                  natural_trace_bound=gates["workload"]["schema"] in ("natural_off_qualification_workload_v1", "natural_trace_workload_v1"),')
replace_once('                  fit_or_evaluation_input_allowed=False, production_release_qualified=False, cost_qualified=False,',
             '                  fit_or_evaluation_input_allowed=route["formal"], production_release_qualified=False, cost_qualified=False,\n                  actual_run_config_ref=gates.get("actual_run_config_ref"),\n                  formal_trace_binding_ref=config.get("formal_trace_binding_ref"),\n                  formal_phase_preflight=gates.get("formal_phase_preflight"),\n                  formal_initial_namespace=None if not route["formal"] else dict(\n                      storage_path=str(storage), fresh_before_runtime_creation=True, whole_partition_no_reset=True,\n                      run_id=config["run_id"], arm=config["arm"], partition=gates["formal_workload_binding"]["partition"]),\n                  formal_partition=None if not route["formal"] else gates["formal_workload_binding"]["partition"],')
old_startup = '''            startup_module = driver.load(root, finite_activation["descriptor"]["startup_ref"],
                                         "_strong_preconstruction_finite_" + str(time.monotonic_ns()))
            reactor_module = importlib.import_module("py_kvcache.reactor")
            finite_startup = startup_module.FiniteStartup(native_module=native_module, reactor_module=reactor_module,
                table=qualified_table, issuer=issuer, run_id=config["run_id"],
                budget_ns=finite_activation["independent_budget"]["internal_step_budget_ns"], root=root,
                runtime_refs=refs, driver=driver, binding_module=binding_module, validate_drained=validate_drained_snapshot,
                observation_only=config["phase"] == "development",
                reserve_covered_cell_signatures=finite_activation.get("reserve_covered_cell_signatures"))'''
replace_once(old_startup, '''            finite_startup = prepare_finite_controller(root, gates, native_module=native_module,
                table=qualified_table, issuer=issuer, binding_module=binding_module, driver=driver)''')
replace_once('driver.require((bridge is None) if config["phase"] == "qualification" else',
             'driver.require((bridge is None) if config["arm"] == "U" else')
replace_once('bridge_mode=None if bridge is None else bridge.mode, real_interference_table_installed=qualified_table is not None,\n            actual_I_strategy_activated=config["phase"] == "effect", native_U_preserved=config["phase"] != "effect",\n            qualified_finite_development_shadow=config["phase"] == "development",',
             'bridge_mode=None if bridge is None else bridge.mode, real_interference_table_installed=finite_startup is not None,\n            actual_I_strategy_activated=config["phase"] == "effect" and config["arm"] == "I",\n            native_U_preserved=not (config["phase"] == "effect" and config["arm"] == "I"),\n            qualified_finite_development_shadow=config["phase"] == "development" and config["arm"] == "I",\n            real_private_cost_table_reissued_for_identity=qualified_table is not None,\n            optional_I_controller_constructed=finite_startup is not None,')
replace_once('''            result["finite_current_owner_install"] = finite_startup.install_after_original_drain(active_capture, identity,
                                                                                                  host_observer=host_observer)''',
             '''            result["finite_running_identity_verified"] = True
            if finite_startup is not None:
                result["finite_current_owner_install"] = finite_startup.install_after_original_drain(active_capture, identity,
                                                                                                      host_observer=host_observer)
            else:
                driver.require(config["arm"] == "U" and bridge is None,
                               "same calibrated U identity with original policy off and no I owner attachment")''')
replace_once('        driver.new_json(out / "actual-request-outputs.json", result["frontend"])',
             '        if route["formal"]:\n            result["formal_frontend_identity_closure"] = driver.validate_formal_frontend_records(gates, result["frontend"])\n        driver.new_json(out / "actual-request-outputs.json", result["frontend"])\n        if route["formal"]:\n            result["actual_request_outputs_ref"] = driver.ref(root, (out / "actual-request-outputs.json").relative_to(root).as_posix())')
replace_once('        driver.new_json(out / "actual-original-full-step-capture.json", result["full_original_step_capture"])',
             '        driver.new_json(out / "actual-original-full-step-capture.json", result["full_original_step_capture"])\n        if route["formal"]:\n            result["actual_original_full_step_capture_ref"] = driver.ref(root, (out / "actual-original-full-step-capture.json").relative_to(root).as_posix())')
replace_once('        result["observation_unknown_does_not_invent_model_failure_or_performance_result"] = True',
             '''        result["observation_unknown_does_not_invent_model_failure_or_performance_result"] = True
        if route["formal"]:
            result["formal_complete_native_observation"] = validate_formal_capture(root, gates,
                result["full_original_step_capture"], result["frontend"], source_ref=refs[collector_relative], driver=driver)''')
replace_once('''                            else "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN" if config["phase"] == "development"
                            else "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS")''',
             '''                            else "PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if config["phase"] == "development" and config["arm"] == "U"
                            else "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN" if config["phase"] == "development"
                            else "PASS_FORMAL_EFFECT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if config["arm"] == "U"
                            else "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS")''')
target = HERE / 'native_runtime_v3.py'
if target.exists():
    assert target.read_bytes() == source.encode('utf-8'), 'new adapter path already differs'
else:
    target.write_bytes(source.encode('utf-8'))
print(target)
