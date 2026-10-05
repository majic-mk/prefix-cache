"""Original-runtime adapter reached only after source and original GPU guard gates.

Every module imported on CPU uses the standard library. Model, native I/O and
torch are imported only inside execute after the original active reservation.
No scheduler, allocator, completion protocol or model executor is implemented.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def capture_original_worker(worker):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    connector = get_kv_transfer_group()
    registry = connector.connector_worker.worker
    if type(registry.handlers) is not set or len(registry.handlers) != 1:
        raise ValueError("sole original native handler registry required")
    handler = next(iter(registry.handlers))
    mapping = registry.transfer_type_to_handler
    if set(mapping) != {("GPU", "SHARED_STORAGE"), ("SHARED_STORAGE", "GPU")} or not all(
            item is handler for item in mapping.values()):
        raise ValueError("original native bidirectional registration required")
    return worker, handler


def validate_drained_snapshot(snapshot):
    if type(snapshot) is not dict or snapshot.get("owner_capture") is not True:
        raise ValueError("original owner captured native snapshot required")
    native, aio = snapshot.get("native"), snapshot.get("aio")
    if type(native) is not dict or not native or not all(type(x) is int and x == 0 for x in native.values()):
        raise ValueError("native work not drained")
    if type(aio) is not dict or not all(type(aio.get(k)) is int and aio[k] == 0
                                      for k in ("outstanding", "pending", "ready", "unreaped")):
        raise ValueError("actual native AIO not drained")
    accounting = snapshot.get("stage_accounting")
    if accounting is not None:
        if accounting.get("valid") is not True or not all(
                type(row["inflight_ops"]) is int and row["inflight_ops"] == 0 and
                type(row["inflight_bytes"]) is int and row["inflight_bytes"] == 0 and
                type(row["failed_ops"]) is int and row["failed_ops"] == 0
                for row in accounting["stages"].values()):
            raise ValueError("original stage accounting not drained")
    return snapshot


def drain_original(llm, original_acquire, handler):
    # Existing worker_control submits and waits original handlers. It explicitly
    # leaves completion delivery/get_finished to the unchanged engine metadata.
    value = llm.collective_rpc(original_acquire.worker_control, timeout=45, args=("drain", None))
    if type(value) is not list or len(value) != 1:
        raise ValueError("one original native drain result required")
    deadline = time.monotonic() + 10
    while True:
        snapshot = handler.coordinator.inspect_snapshot(timeout=2)
        try:
            validate_drained_snapshot(snapshot)
            return dict(original_worker_control=value[0], owner_snapshot=snapshot,
                        completion_delivery="unchanged original engine metadata; no synthetic completion or flush request",
                        scheduler_protectors_release_proof=False)
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def verify_original_tail(report):
    for name in ("actual_aio", "actual_native"):
        values = report[name]
        selected = values if name == "actual_native" else {k: values[k] for k in ("outstanding", "pending", "ready", "unreaped")}
        if not all(type(v) is int and v == 0 for v in selected.values()):
            raise ValueError("original native post-shutdown tail not empty")
    if type(report["actual_handler_active"]) is not int or report["actual_handler_active"] != 0:
        raise ValueError("actual original active transfer handlers remain")
    if type(report.get("observation_failures")) is not int or report["observation_failures"] != 0:
        raise ValueError("original native observation failed")
    return True


def verify_loaded_native(root, refs, driver):
    rows = []
    for name, module in sorted(sys.modules.items()):
        if name in ("py_kvcache", "prefix_io_control") or name.startswith(("py_kvcache.", "prefix_io_control.")):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(root).as_posix()
            driver.require(relative in refs, "loaded native source outside frozen original closure")
            driver.check_ref(root, refs[relative])
            rows.append(dict(module=name, source_ref=refs[relative]))
    driver.require(1 <= len(rows) <= 128 and any(row["module"] == "py_kvcache.reactor" for row in rows),
                   "actual original native reactor imported")
    return rows


def original_planner_identity(llm, handler, root, refs, driver):
    # These member names must match the current frozen original InprocClient and
    # Scheduler sources; tests replay their exact values and source pin them.
    core = llm.llm_engine.engine_core.engine_core
    connector = core.scheduler.connector
    scheduler = connector.connector_scheduler
    spec = scheduler.spec
    planner = spec._planner
    driver.require(planner is not None and scheduler.manager._planner is planner,
                   "same actual original cost-admission planner in scheduler manager")
    cfg = spec.shared_file_config
    driver.require(cfg.load_planner == "on" and cfg.enable_preload is True and cfg.preload_share_staging is True,
                   "actual original planner/preload/shared staging settings")
    driver.require(type(planner).__module__ == "py_kvcache.load_planner" and
                   driver.ref(root, driver.NATIVE + "/load_planner.py") == refs[driver.NATIVE + "/load_planner.py"],
                   "actual original LoadPlanner source")
    driver.require(handler.coordinator.reactor.progress_bridge_enabled is True,
                   "retained common original admission/accounting")
    return dict(actual_planner_present=True, same_manager_planner=True, load_planner="on",
                enable_preload=True, preload_share_staging=True, planner_source_ref=refs[driver.NATIVE + "/load_planner.py"],
                connector_source_ref=refs[driver.AUTHOR + "/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"],
                scheduler_source_ref=refs[driver.AUTHOR + "/vllm/v1/core/sched/scheduler.py"],
                original_engine_source_ref=refs[driver.AUTHOR + "/vllm/v1/engine/llm_engine.py"])


def preflight_collector_binding(root, gates, *, driver):
    """Source binding only: cannot issue a table or replace the native executor.

    The existing activation verifier owns the full calibration ancestry/deadline
    gate. This repeats exact collector/delegate bytes before model imports.
    """
    config, refs = gates["config"], gates["refs"]
    directory = Path(config["runtime_ref"]["path"]).parent
    legacy = (directory / "bounded_native_full_step_collector.py").as_posix()
    if config["phase"] not in ("development", "effect"):
        driver.require(legacy in refs, "unchanged native off/shadow collector source frozen")
        driver.check_ref(root, refs[legacy])
        return refs[legacy]
    activation = gates.get("finite_activation")
    driver.require(type(activation) is dict and type(activation.get("descriptor")) is dict and
                   type(activation.get("calibration_plan")) is dict,
                   "closed finite calibration source gate before framework/model")
    descriptor, plan = activation["descriptor"], activation["calibration_plan"]
    selected = descriptor.get("collector_source_ref", descriptor.get("collector_ref"))
    driver.require(type(selected) is dict and descriptor.get("collector_ref") == selected and
                   ("collector_source_ref" not in descriptor or descriptor["collector_source_ref"] == selected),
                   "one exact closed startup collector reference")
    expected = (directory / "bounded_native_full_step_collector_v2.py").as_posix()
    driver.require(selected.get("path") == expected and
                   selected.get("sha256") == "a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0" and
                   selected.get("bytes") == 13016 and refs.get(expected) == selected and
                   activation.get("runtime_refs", {}).get(expected) == selected,
                   "exact actual V2 collector in current runtime closure")
    driver.check_ref(root, selected)
    driver.require(plan.get("collector_source_ref") == selected and
                   plan.get("source_lock_ref") == descriptor.get("calibration_source_lock_ref") and
                   plan.get("gpu_uuid") == config["gpu_uuid"] and
                   plan.get("common_runtime_domain_sha256") == gates["common_runtime_domain_sha256"],
                   "same calibration collector/device/common domain")
    declared_plan = driver.read(driver.check_ref(root, descriptor["plan_ref"]))
    driver.require(declared_plan == plan, "actual frozen calibration plan bytes before model")
    original = plan.get("original_collector_source_ref")
    driver.require(type(original) is dict and original.get("path") == legacy and
                   original.get("sha256") == "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d" and
                   original.get("bytes") == 25070 and refs.get(legacy) == original and
                   activation["runtime_refs"].get(legacy) == original,
                   "same immutable original collector delegate in runtime closure")
    driver.check_ref(root, original)
    calibration = driver.read(driver.check_ref(root, descriptor["calibration_source_lock_ref"]))
    driver.require(type(calibration.get("files")) is list and
                   all(refs.get(row["path"]) == row for row in calibration["files"]) and
                   selected in calibration["files"] and original in calibration["files"],
                   "exact inherited calibration source rows; appended runtime is allowed")
    return selected



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


def execute(root, gates, reservation, *, driver):
    config, refs = gates["config"], gates["refs"]
    driver.require(config["phase"] in ("qualification", "shadow", "development", "effect"), "bounded original-runtime phase")
    driver.require(reservation["label"] == config["run_id"], "same original guarded runtime")
    route = preflight_runtime_route(root, gates, driver=driver)
    collector_source_ref = preflight_collector_binding(root, gates, driver=driver)
    out, storage = gates["out"], gates["storage"]
    out.mkdir(parents=True, exist_ok=False)
    storage.mkdir(parents=True, exist_ok=False)
    old_environment, old_path, old_cwd = dict(os.environ), list(sys.path), Path.cwd()
    old_base = sys.modules.get("native_gpu_prefix_smoke")
    base_present = "native_gpu_prefix_smoke" in sys.modules
    llm = finder = probe = handler = runtime_tail = native_drain = active_capture = finite_startup = host_binding = None
    native_module = journal_factory = original_factory = journal = host_observer = None
    result = dict(schema="strong_native_original_workload_result_v1", status="FAILED_STRONG_NATIVE_WORKLOAD",
                  origin="actual_guarded_original_runtime", phase=config["phase"], arm=config["arm"], mode=config["mode"],
                  run_id=config["run_id"], gpu_uuid=config["gpu_uuid"], guard_reservation_id=reservation["id"],
                  runner_ref=config["runner_ref"], runtime_ref=config["runtime_ref"],
                  source_lock_ref=config["source_lock_ref"], workload_ref=config["workload_ref"], pair_config_ref=config["pair_config_ref"],
                  common_runtime_domain_sha256=gates["common_runtime_domain_sha256"],
                  standing_authorization_ref=gates["standing_authorization_ref"], framework_import_attempted=False,
                  original_engine_initialization_attempted=False, original_engine_shutdown_returned=False,
                  native_tail_drained=False, os_session_drained=False, os_session_drain_owner="unchanged original outer GPU guard after child exit",
                  workload_kind=gates["workload"].get("workload_kind", "natural_author_text_qualification"),
                  natural_trace_bound=gates["workload"]["schema"] in ("natural_off_qualification_workload_v1", "natural_trace_workload_v1"),
                  synthetic_io_or_stall_injected=False, midstream_cache_reset=False, selected_natural_request_prefix_injected=False,
                  fit_or_evaluation_input_allowed=route["formal"], production_release_qualified=False, cost_qualified=False,
                  actual_run_config_ref=gates.get("actual_run_config_ref"),
                  formal_trace_binding_ref=config.get("formal_trace_binding_ref"),
                  formal_phase_preflight=gates.get("formal_phase_preflight"),
                  formal_initial_namespace=None if not route["formal"] else dict(
                      storage_path=str(storage), fresh_before_runtime_creation=True, whole_partition_no_reset=True,
                      run_id=config["run_id"], arm=config["arm"], partition=gates["formal_workload_binding"]["partition"]),
                  formal_partition=None if not route["formal"] else gates["formal_workload_binding"]["partition"],
                  formal_effect_qualified=False, formal_goodput_allowed=False, strategy_improvement_proved=False,
                  development_diagnostic_only=gates.get("finite_activation", {}).get("development_diagnostic_only", False)
                      if gates.get("finite_activation") is not None else False,
                  actual_gpu_runs=1, completed_guard_required_before_qualification_receipt=True)
    try:
        common = driver.load(root, refs[driver.COMMON], "_strong_native_common_" + str(time.monotonic_ns()))
        base = driver.load(root, refs[driver.SMOKE], "_strong_native_base_" + str(time.monotonic_ns()))
        base.ROOT = root
        base.AUTHOR_ROOT = root / driver.AUTHOR
        runtime_tail = driver.load(root, refs[driver.TAIL], "_strong_native_tail_" + str(time.monotonic_ns()))
        sys.path[:0] = [str(root / "experiments/prefix_io_v1/scripts"),
                       str(root / driver.OVERLAY / "third_party/work/py-kvcache-p4-02-cpu"),
                       str(root / driver.OVERLAY / "third_party/work/prefix-io-p4-02-cpu/src"),
                       str(root / driver.CANDIDATE)]
        activation_module = finite_activation = None
        if config["phase"] in ("development", "effect"):
            finite_activation = gates["finite_activation"]
            driver.require(finite_activation is not None, "closed finite activation gate required")
            driver.require(not any(name == "prefix_io_control" or name.startswith("prefix_io_control.") for name in sys.modules),
                           "fresh sole qualified policy package")
            sys.path.insert(0, str(root / finite_activation["activation_source_path"]))
            activation_module = driver.activation_gate_module(root, config, refs)
        sys.modules["native_gpu_prefix_smoke"] = base
        native_drain = driver.load(root, refs[driver.DRAIN], "_strong_native_original_drain_" + str(time.monotonic_ns()))
        result["runtime_cache_environment"] = common.configure_process_caches(out, base)
        sdk = driver.load(root, refs[driver.SDK], "_strong_native_sdk_" + str(time.monotonic_ns()))
        sdk_evidence = sdk.prepare_site_sdk(root, refs, out)
        result["sdk_environment"] = sdk_evidence
        result["ninja_environment"] = common.prepare_process_ninja(root, refs, out, sdk_evidence)
        optional, absence = common.load_optional_probe(root, refs)
        result["optional_absence"] = absence
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["PYTHONHASHSEED"] = "0"
        driver.require(os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline guard")
        model_dir, model_identity = base.validate_local_model(driver.safe(root, driver.MODEL), driver.safe(root, driver.MODEL_PLAN))
        model_identity = dict(model_identity, model_directory=str(model_dir.resolve()))
        qualifier = driver.load(root, refs[driver.QUALIFIER], "_strong_native_original_finder_" + str(time.monotonic_ns()))
        finder = qualifier.BoundAuthorFinder(root, refs)
        driver.require(not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules), "fresh original author imports")
        sys.meta_path.insert(0, finder)
        os.chdir(out)
        result["framework_import_attempted"] = True
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        import vllm.utils.import_utils as import_utils
        probe = optional.install_capability_probe(import_utils, finder, root, refs)
        driver.require(qualifier.normalize_uuid(torch.cuda.get_device_properties(0).uuid) ==
                       qualifier.normalize_uuid(config["gpu_uuid"]), "actual bound hardware UUID")
        driver.require(Path(vllm.__file__).resolve().is_relative_to(root / driver.AUTHOR), "pinned original author vLLM")
        engine_config = deepcopy(gates["pair"][config["arm"]]["engine"])
        extra = engine_config["kv_transfer_config"]["kv_connector_extra_config"]
        if config["phase"] == "shadow":
            extra["prefix_io_p4_policy"]["mode"] = "shadow"
            driver.require(extra["prefix_io_p4_policy"]["cost_table"] is None,
                           "shadow carries no false qualified generalized table")
        if config["phase"] == "development" and (
                not finite_activation.get("development_diagnostic_only", False) or config["arm"] == "I"):
            # Observe the same qualified finite controller with a prospective
            # full deadline. Its wrapper returns native fallback for every
            # proposed ordinary action; no unfrozen reserve authorizes I.
            extra["prefix_io_p4_policy"]["internal_step_budget_ns"] = finite_activation["independent_budget"]["internal_step_budget_ns"]
        alias = out / base.MODEL_ID
        alias.parent.mkdir(parents=True, exist_ok=False)
        alias.symlink_to(model_dir, target_is_directory=True)
        result["engine_config"] = engine_config
        result["model_identity"] = model_identity
        binding_module = qualified_table = issuer = None
        from prefix_io_control.p4_native_window_journal import NativeWindowJournal
        native_module = importlib.import_module("py_kvcache.vllm")
        old_raw_path = driver.PREVIOUS + "/calibration_v2/run_native_cost_experiment.py"
        driver.require(old_raw_path in refs, "original sparse native owner observation source frozen")
        old_raw = driver.load(root, refs[old_raw_path], "_strong_original_sparse_owner_" + str(time.monotonic_ns()))
        journal = NativeWindowJournal(config["run_id"], refs[driver.NATIVE + "/reactor.py"]["sha256"], enabled=True, max_events=4096)
        sparse = old_raw.SparseOwnerSink(journal)
        original_factory = native_module.TransferCoordinator
        def observed_factory(**kwargs):
            driver.require(kwargs.get("progress_run_id") == config["run_id"] and kwargs.get("max_accepted_parents") == 8,
                           "same original common admission/parent identity while observing")
            previous_sink = kwargs.get("observation_sink")
            def observe_owner(reactor):
                if previous_sink is not None:
                    previous_sink(reactor)
                sparse(reactor)
            kwargs["stage_accounting"], kwargs["observation_sink"] = journal, observe_owner
            return original_factory(**kwargs)
        journal_factory = observed_factory
        native_module.TransferCoordinator = journal_factory
        if config["phase"] in ("development", "effect"):
            qualified_table, issuer = activation_module.issue(root, finite_activation, driver=driver)
            binding_module = driver.load(root, finite_activation["descriptor"]["finite_binding_ref"],
                                         "_strong_live_finite_binding_" + str(time.monotonic_ns()))
            finite_startup = prepare_finite_controller(root, gates, native_module=native_module,
                table=qualified_table, issuer=issuer, binding_module=binding_module, driver=driver)
        result["original_engine_initialization_attempted"] = True
        llm = LLM(model=base.MODEL_ID, **engine_config)
        driver.require(llm.llm_engine.vllm_config.scheduler_config.async_scheduling is False,
                       "original synchronous scheduler unchanged")
        driver.require(llm.llm_engine.log_stats is False and llm.llm_engine.logger_manager is None,
                       "same supported optional metrics switch")
        holders = []
        def capture(worker):
            original_worker, original_handler = capture_original_worker(worker)
            holders.append((original_worker, original_handler))
            return {"captured": True}
        captured = llm.collective_rpc(capture, timeout=10)
        driver.require(type(captured) is list and len(captured) == len(holders) == 1, "actual original Uni worker")
        worker, handler = holders[0]
        result["loaded_native_source_binding"] = verify_loaded_native(root, refs, driver)
        result["original_planner_identity"] = original_planner_identity(llm, handler, root, refs, driver)
        bridge = handler.coordinator.reactor._prefix_p4_bridge
        driver.require((bridge is None) if config["arm"] == "U" else
                       (bridge is not None and bridge.mode == "shadow" and bridge.policy.table is None)
                       if config["phase"] == "shadow" else
                       (bridge is not None and bridge.mode == "interference" and bridge.policy.table is qualified_table and
                        issuer.qualified_identity(qualified_table) is not None),
                       "requested native off or observation-only shadow semantics")
        result["strategy_runtime"] = dict(bridge_is_none=bridge is None,
            bridge_mode=None if bridge is None else bridge.mode, real_interference_table_installed=finite_startup is not None,
            actual_I_strategy_activated=config["phase"] == "effect" and config["arm"] == "I",
            native_U_preserved=not (config["phase"] == "effect" and config["arm"] == "I"),
            qualified_finite_development_shadow=config["phase"] == "development" and config["arm"] == "I",
            real_private_cost_table_reissued_for_identity=qualified_table is not None,
            optional_I_controller_constructed=finite_startup is not None,
            unknown_cells_native_U=True, production_batch_activation=False)
        runtime_tail.original_method(handler, "shutdown", root, refs)
        runtime_tail.original_method(handler.coordinator, "inspect_snapshot", root, refs)
        result["before_workload_native_drain"] = drain_original(llm, native_drain, handler)
        collector_relative = collector_source_ref["path"]
        driver.require(collector_relative in refs, "same complete observation source frozen for U and I")
        collector = driver.load(root, refs[collector_relative], "_strong_native_step_capture_" + str(time.monotonic_ns()))
        collector_refs = dict(refs)
        # Preserve the original collector API's historical lookup key while
        # binding the active, unchanged common native reactor bytes.
        collector_refs["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"] = refs[driver.NATIVE + "/reactor.py"]
        try:
            active_capture = collector.install(worker, common=common, root=root, refs=collector_refs,
                run_id=config["run_id"], event_class=torch.cuda.Event, selected_offsets=(), action=None,
                origin="native_gpu_recording", max_steps=4096)
        except Exception as exc:
            result["optional_full_step_capture_install_unknown"] = str(exc)[:2000]
            if config["phase"] in ("development", "effect"):
                raise
        host_path = Path(config["runtime_ref"]["path"]).parent.parent / "activation/control_observation/host_control_observer.py"
        binding_path = Path(config["runtime_ref"]["path"]).with_name("host_boundary_binding.py")
        driver.require(host_path.as_posix() in refs and binding_path.as_posix() in refs, "same compact host observation implementation frozen")
        host_module = driver.load(root, refs[host_path.as_posix()], "_strong_host_observer_" + str(time.monotonic_ns()))
        host_binding_module = driver.load(root, refs[binding_path.as_posix()], "_strong_host_metadata_hooks_" + str(time.monotonic_ns()))
        host_observer = host_module.HostControlObserver(config["run_id"], max_steps=4096, max_intervals_per_step=64)
        if config["phase"] in ("development", "effect"):
            driver.require(refs[collector_relative] == finite_activation["descriptor"]["collector_ref"],
                           "U/I collector equals exact-cell measurement source")
            identity = binding_module.verify_running_identity(worker, handler, active_capture, qualified_table,
                issuer=issuer, root=root, runtime_refs=refs, common_runtime_domain_sha256=gates["common_runtime_domain_sha256"],
                model_identity=model_identity, calibration_lock_ref=finite_activation["descriptor"]["calibration_source_lock_ref"])
            result["finite_running_identity_verified"] = True
            if finite_startup is not None:
                result["finite_current_owner_install"] = finite_startup.install_after_original_drain(active_capture, identity,
                                                                                                      host_observer=host_observer)
            else:
                driver.require(config["arm"] == "U" and bridge is None,
                               "same calibrated U identity with original policy off and no I owner attachment")
        host_binding = host_binding_module.HostBoundaryBinding(host_observer, llm=llm, worker=worker, capture=active_capture,
            root=root, refs=refs, driver=driver, finite_binding=None if finite_startup is None else finite_startup.binding)
        result["complete_observation_config"] = dict(source_ref=refs[collector_relative], max_pending_events=128,
            max_original_steps=4096, selected_offsets=[], action=None, extra_synchronize=False,
            common_U_I_observation=True, original_native_choice_preserved=True)
        sampling_common = deepcopy(gates["pair"][config["arm"]]["sampling"])
        def sampling(record):
            return SamplingParams(**dict(sampling_common, seed=record["seed"], min_tokens=record["min_tokens"], max_tokens=record["max_tokens"]))
        counts, completions = {}, set()
        with (out / "actual-token-return-events.jsonl").open("x", encoding="utf-8", newline="\n") as progress:
            def capture_progress(rows, step):
                for row in rows:
                    rid = row["request_id"]
                    first = counts.get(rid, 0)
                    for ordinal in range(first, len(row["output_token_ids"])):
                        progress.write(json.dumps(dict(request_id=rid, native_request_id=row["native_request_id"],
                            token_ordinal=ordinal, token_id=row["output_token_ids"][ordinal],
                            return_ns=row["token_return_ns"][ordinal], original_step=step), allow_nan=False) + "\n")
                    counts[rid] = len(row["output_token_ids"])
                    if row["state"] == "COMPLETED" and rid not in completions:
                        progress.write(json.dumps(dict(request_id=rid, state="COMPLETED", finish_ns=row["finished_ns"],
                                                     total_tokens=counts[rid]), allow_nan=False) + "\n")
                        completions.add(rid)
                progress.flush()
            result["frontend"] = driver.drive_original_engine(llm.llm_engine, sampling, gates["workload"]["records"],
                max_concurrency=gates["workload"]["max_concurrency"], deadline_seconds=min(120, config["seconds_limit"]),
                on_progress=host_binding.bind_frontend_progress(capture_progress, root=root, refs=refs, driver=driver))
        if route["formal"]:
            result["formal_frontend_identity_closure"] = driver.validate_formal_frontend_records(gates, result["frontend"])
        driver.new_json(out / "actual-request-outputs.json", result["frontend"])
        if route["formal"]:
            result["actual_request_outputs_ref"] = driver.ref(root, (out / "actual-request-outputs.json").relative_to(root).as_posix())
        result["full_original_step_capture"] = None if active_capture is None else active_capture.export()
        driver.new_json(out / "actual-original-full-step-capture.json", result["full_original_step_capture"])
        if route["formal"]:
            result["actual_original_full_step_capture_ref"] = driver.ref(root, (out / "actual-original-full-step-capture.json").relative_to(root).as_posix())
        result["full_step_evidence_qualified"] = (result["full_original_step_capture"] is not None and
                                                 result["full_original_step_capture"]["valid"] is True)
        result["observation_unknown_does_not_invent_model_failure_or_performance_result"] = True
        if route["formal"]:
            result["formal_complete_native_observation"] = validate_formal_capture(root, gates,
                result["full_original_step_capture"], result["frontend"], source_ref=refs[collector_relative], driver=driver)
        if active_capture is not None:
            active_capture.detach()
            active_capture = None
        driver.require(result["frontend"]["actual_request_stream_completed"] is True, "all selected original requests completed")
        result["after_workload_native_drain"] = drain_original(llm, native_drain, handler)
        if finite_startup is not None:
            result["finite_current_cell_decisions"] = finite_startup.binding.evidence()
        result["status"] = ("PASS_STRONG_NATIVE_OFF_WORKLOAD_LIFECYCLE" if config["phase"] == "qualification"
                            else "PASS_STRONG_NATIVE_SHADOW_WORKLOAD_ONLY" if config["phase"] == "shadow"
                            else "PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if config["phase"] == "development" and config["arm"] == "U"
                            else "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN" if config["phase"] == "development"
                            else "PASS_FORMAL_EFFECT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if config["arm"] == "U"
                            else "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS")
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc)[:2000], traceback=traceback.format_exc(limit=24))
    finally:
        if active_capture is not None:
            try:
                result["failed_run_full_step_capture"] = active_capture.export()
            except Exception as exc:
                result["failed_capture_export_error"] = str(exc)[:2000]
            finally:
                try:
                    active_capture.detach()
                except Exception as exc:
                    result["status"] = "FAILED_COMPLETE_OBSERVER_RESTORE"
                    result["capture_cleanup_error"] = str(exc)[:2000]
        if host_binding is not None:
            result["host_boundary_wrappers_restored"] = host_binding.detach()
        if host_observer is not None:
            try:
                observation = host_observer.export()
                path = out / "actual-host-control-observation.json"
                driver.new_json(path, observation)
                result["host_control_observation_ref"] = host_module.closed_ref(path)
                result["host_clock_scope"] = observation["clock_scope"]
                result["host_control_reserve_qualification"] = "off/shadow only observed; actual completed development guard/source/capture/controller join required"
                if finite_startup is not None:
                    actual_identity = issuer.qualified_identity(qualified_table)
                    preview_evidence = finite_startup.binding.evidence()
                    result["host_controller_coverage"] = dict(
                        kind="same_qualified_I_lookup_binding_path_observed_only_no_issue" if config["phase"] == "development" else "actual_on_path",
                        native_step_ordinals=[row["native_step_ordinal"] for row in observation["per_step"]],
                        qualified_preview_observed_ordinals=[row["native_step_ordinal"] for row in observation["per_step"] if
                            any(item["boundary_id"] == "qualified_finite_controller_preview" for item in row["intervals"])],
                        ordinary_IO_issued_by_observer=False,
                        qualified_identity_source_lock_sha256=actual_identity.source_lock_sha256,
                        qualified_cells=[list(cell) for cell in actual_identity.cells],
                        controller_boundary_ids=["frontend_progress_bookkeeping", "qualified_finite_controller_preview"],
                        preview_windows=preview_evidence["preview_windows"], attempt_count=preview_evidence["attempt_count"],
                        overflow=preview_evidence["overflow"], unknown_coverage=preview_evidence["unknown_coverage"],
                        condition_max_age_ns=bridge.policy.config.sample_max_age_ns,
                        missing_preview_is_not_fabricated=True, actual_reserve_verifier_required=True)
            except Exception as exc:
                result["host_observation_error"] = str(exc)[:2000]
        if llm is not None:
            try:
                if handler is not None and native_drain is not None:
                    result["final_before_shutdown_drain"] = drain_original(llm, native_drain, handler)
            except Exception as exc:
                result["status"] = "FAILED_FINAL_ORIGINAL_NATIVE_DRAIN"
                result["final_drain_error"] = dict(type=type(exc).__name__, message=str(exc))
            finally:
                try:
                    llm.llm_engine.engine_core.shutdown(timeout=15)
                    result["original_engine_shutdown_returned"] = True
                    if handler is not None:
                        result["post_original_shutdown"] = runtime_tail.post_shutdown_snapshot(handler)
                        result["native_tail_drained"] = verify_original_tail(result["post_original_shutdown"])
                except Exception as exc:
                    result["status"] = "FAILED_ORIGINAL_SHUTDOWN_OR_TAIL"
                    result["shutdown_error"] = dict(type=type(exc).__name__, message=str(exc))
        if probe is not None:
            try:
                result["optional_probe_restored"] = probe.detach()
                driver.require(result["optional_probe_restored"] is True, "original optional probe restoration")
            except Exception as exc:
                result["status"] = "FAILED_OPTIONAL_PROBE_RESTORE"
                result["probe_cleanup_error"] = dict(type=type(exc).__name__, message=str(exc))
        if finite_startup is not None:
            try:
                if finite_startup.binding is not None:
                    result["finite_current_cell_decisions"] = finite_startup.binding.evidence()
                result["finite_startup_restored_after_original_shutdown"] = finite_startup.detach_after_original_shutdown()
                driver.require(result["finite_startup_restored_after_original_shutdown"] is True, "finite optional adapter restoration")
            except Exception as exc:
                result["status"] = "FAILED_FINITE_ADAPTER_RESTORE"
                result["finite_cleanup_error"] = str(exc)[:2000]
        if native_module is not None and journal_factory is not None:
            if native_module.TransferCoordinator is journal_factory:
                native_module.TransferCoordinator = original_factory
            else:
                result["status"] = "FAILED_NATIVE_OBSERVATION_FACTORY_RESTORE"
        if journal is not None:
            events, frames, valid = journal.published()
            result["actual_native_stage_journal"] = dict(run_id=config["run_id"], source_sha256=refs[driver.NATIVE + "/reactor.py"]["sha256"],
                valid=valid, lost=journal.lost, last_reason=journal.last_reason,
                events=[asdict(item) for item in events], frames=[asdict(item) for item in frames], max_events=4096,
                original_owner_queue_retained=True, opportunity="ordinary_ready_legality_unknown_until_compact_ready_witness",
                no_normal_candidate_conclusion_allowed=False)
            driver.new_json(out / "actual-native-stage-journal.json", result["actual_native_stage_journal"])
        if finder is not None and finder in sys.meta_path:
            sys.meta_path.remove(finder)
        if base_present:
            sys.modules["native_gpu_prefix_smoke"] = old_base
        else:
            sys.modules.pop("native_gpu_prefix_smoke", None)
        sys.path[:] = old_path
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_environment)
        try:
            for row in (config["runner_ref"], config["runtime_ref"], config["workload_ref"], config["pair_config_ref"]):
                driver.check_ref(root, row)
            result["immutable_runner_config_workload_unchanged"] = True
        except Exception as exc:
            result["status"] = "FAILED_IMMUTABLE_RUNTIME_INPUT"
            result["postrun_source_error"] = dict(type=type(exc).__name__, message=str(exc))
        driver.new_json(out / "strong-native-workload-result.json", result)
    return result
