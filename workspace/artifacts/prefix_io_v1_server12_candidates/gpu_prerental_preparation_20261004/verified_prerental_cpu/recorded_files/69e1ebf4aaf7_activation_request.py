"""Closed finite activation inputs; CPU preparation cannot fabricate evidence."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import time

SCHEMA = "finite_gpu_cell_activation_request_v1"
FIELDS = {"schema", "plan_ref", "measurements_ref", "completed_guard_ref", "intent_ref",
          "calibration_source_lock_ref", "collector_ref", "independent_deadline_ref", "development_reserve_ref",
          "development_budget_ref", "protocol_source_ref", "issuer_source_ref", "cost_table_source_ref",
          "finite_binding_ref", "startup_ref", "reserve_verification_ref", "reserve_join_source_ref",
          "development_plan_ref", "development_guard_ref", "development_observation_ref", "development_native_result_ref"}
POST_DEVELOPMENT = {"development_reserve_ref", "development_budget_ref", "reserve_verification_ref",
                    "development_plan_ref", "development_guard_ref", "development_observation_ref", "development_native_result_ref"}


def verify(root, row, *, refs, pair, gpu_uuid, driver, phase="effect"):
    value = driver.read(driver.check_ref(root, row))
    driver.require(type(value) is dict and set(value) == FIELDS and value["schema"] == SCHEMA,
                   "closed exact finite GPU activation descriptor")
    driver.require(refs.get(row["path"]) == row, "activation request in frozen runtime closure")
    for key in FIELDS - {"schema"}:
        if phase == "development" and key in POST_DEVELOPMENT:
            driver.require(value[key] is None, "development shadow cannot claim a measured reserve before this run")
            continue
        driver.check_ref(root, value[key])
        driver.require(refs.get(value[key]["path"]) == value[key], "activation leaf in complete runtime closure: " + key)
    plan = driver.read(driver.check_ref(root, value["plan_ref"]))
    driver.require(plan.get("schema") == "strong_gpu_exact_cell_prelaunch_plan_v1" and
                   plan.get("source_lock_ref") == value["calibration_source_lock_ref"] and
                   plan.get("gpu_uuid") == gpu_uuid and
                   plan.get("common_runtime_domain_sha256") == driver.common_domain_sha(pair),
                   "actual same-domain finite strong calibration plan")
    for key in ("collector_source_ref", "issuer_source_ref", "cost_table_source_ref"):
        source = value["collector_ref"] if key == "collector_source_ref" else value[key]
        driver.require(plan.get(key) == source, "same active finite calibration leaf: " + key)
    calibration = driver.read(driver.check_ref(root, value["calibration_source_lock_ref"]))
    driver.require(type(calibration.get("files")) is list and all(refs.get(item["path"]) == item
                   for item in calibration["files"]), "exact calibration-source ancestry; no old singleton reinterpretation")
    protocol = driver.load(root, value["protocol_source_ref"], "_finite_independent_budget_" + str(time.monotonic_ns()))
    deadline = driver.read(driver.check_ref(root, value["independent_deadline_ref"]))
    if phase == "development":
        driver.require(deadline.get("schema") == "independent_deadline_declaration_v1" and
                       deadline.get("origin") == "independent_requirement_before_development" and
                       type(deadline.get("full_control_window_deadline_ns")) is int and
                       deadline["full_control_window_deadline_ns"] > 0, "independent prospective development shadow deadline")
        authority_ref = protocol.verify_closed_ref(deadline.get("authority_ref"), "independent authority")
        authority = driver.read(Path(authority_ref["path"]))
        driver.require(authority.get("schema") == "independent_deadline_authority_v1" and
                       authority.get("origin") == "independent_requirement_before_development" and
                       authority.get("full_control_window_deadline_ns") == deadline["full_control_window_deadline_ns"] and
                       authority.get("service_SLO") == deadline.get("service_SLO"), "actual independent authority bytes")
        protocol.clock_scope(deadline.get("clock_scope"))
        protocol.validate_service_SLO(deadline.get("service_SLO"))
        budget = dict(internal_step_budget_ns=deadline["full_control_window_deadline_ns"],
                      observation_only=True, ordinary_I_authorized=False, reserve_not_measured_yet=True)
    else:
        reserve = driver.read(driver.check_ref(root, value["development_reserve_ref"]))
        budget = protocol.freeze_development_budget(deadline, reserve)
        driver.require(driver.read(driver.check_ref(root, value["development_budget_ref"])) == budget,
                       "independent deadline minus actual measured control reserve frozen before on")
    extra = pair["I"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    driver.require((phase == "development" or extra["prefix_io_p4_policy"]["internal_step_budget_ns"] == budget["internal_step_budget_ns"]) and
                   type(budget["internal_step_budget_ns"]) is int and budget["internal_step_budget_ns"] > 0,
                   "same independent ordinary interference allowance")
    directory = Path(value["issuer_source_ref"]["path"]).parent
    driver.require(directory.name == "prefix_io_control" and directory.parent.name == "source" and
                   value["cost_table_source_ref"]["path"] == (directory / "p4_cost_table.py").as_posix(),
                   "one active private package; no alias issuer instance")
    for path in (directory / "__init__.py", directory / "p4_bridge.py", directory / "p4_policy.py", directory / "p4_options.py"):
        driver.require(path.as_posix() in refs, "complete original finite policy/startup package")
        driver.check_ref(root, refs[path.as_posix()])
    return dict(descriptor=value, independent_budget=budget, activation_source_path=directory.parent.as_posix(),
                calibration_plan=plan, actual_table_issued=False, formal_effect_qualified=False,
                phase=phase, runtime_refs=refs,
                expected_common_domain_sha256=driver.common_domain_sha(pair))


def issue(root, gates, *, driver):
    """Called after real original guard, with the active package imported once."""
    from prefix_io_control import gpu_cell_issuer as issuer
    value, plan = gates["descriptor"], gates["calibration_plan"]
    driver.require(Path(issuer.__file__).resolve() == driver.check_ref(root, value["issuer_source_ref"]).resolve(),
                   "actual sole private issuer package")
    table = issuer.issue_verified_gpu_table(root,
        plan_ref=value["plan_ref"], measurements_ref=value["measurements_ref"],
        guard_ref=value["completed_guard_ref"], intent_ref=value["intent_ref"],
        expected_plan_ref=value["plan_ref"], expected_guard_ref=value["completed_guard_ref"],
        expected_intent_ref=value["intent_ref"],
        expected_runtime_domain_sha256=gates["expected_common_domain_sha256"], expected_gpu_uuid=plan["gpu_uuid"],
        expected_source_lock_sha256=value["calibration_source_lock_ref"]["sha256"],
        expected_collector_source_ref=value["collector_ref"])
    driver.require(issuer.qualified_identity(table) is not None and table.production_qualified is True,
                   "real independently validated finite raw cells")
    if gates["phase"] == "effect":
        join_ref = value["reserve_join_source_ref"]
        host_relative = Path(join_ref["path"]).with_name("host_control_observer.py").as_posix()
        driver.require(host_relative in gates["runtime_refs"], "actual native-reserve host verifier dependency frozen")
        previous, present = sys.modules.get("host_control_observer"), "host_control_observer" in sys.modules
        try:
            host = driver.load(root, gates["runtime_refs"][host_relative], "_finite_read_only_reserve_host_" + str(time.monotonic_ns()))
            sys.modules["host_control_observer"] = host
            join = driver.load(root, join_ref, "_finite_read_only_actual_reserve_" + str(time.monotonic_ns()))
        finally:
            if present:
                sys.modules["host_control_observer"] = previous
            else:
                sys.modules.pop("host_control_observer", None)
        def absolute(row):
            driver.check_ref(root, row)
            return dict(row, path=(root / row["path"]).as_posix())
        receipt = driver.read(driver.check_ref(root, value["development_reserve_ref"]))
        source = driver.project_ref(root, receipt["native_result_ref"])
        driver.require(gates["runtime_refs"].get(source["path"]) == source, "actual qualified reserve source frozen")
        replay = join.verify_existing_native_reserve(root,
            reserve_source_ref=absolute(source), reserve_receipt_ref=absolute(value["development_reserve_ref"]),
            native_verification_ref=absolute(value["reserve_verification_ref"]),
            plan_ref=absolute(value["development_plan_ref"]), expected_plan_ref=absolute(value["development_plan_ref"]),
            guard_ref=absolute(value["development_guard_ref"]), expected_guard_ref=absolute(value["development_guard_ref"]),
            observation_ref=absolute(value["development_observation_ref"]), native_result_ref=absolute(value["development_native_result_ref"]),
            qualified_controller_table=table, issuer=issuer)
        driver.require(replay.get("status") == "PASS_READ_ONLY_NATIVE_RESERVE_REPLAY" and
                       replay.get("frozen_budget") == gates["independent_budget"] and
                       replay.get("native_qualification_verifier_required_separately") is False,
                       "independent real native reserve replay required before on; CPU algebra cannot authorize I")
        covered = replay.get("covered_cell_signatures")
        identity = issuer.qualified_identity(table)
        driver.require(type(covered) is list and covered and
                       all(type(cell) is list and tuple(cell) in identity.cells for cell in covered) and
                       len({tuple(cell) for cell in covered}) == len(covered),
                       "on reserve applies only to the real replay's observed exact-cell subset")
        gates["reserve_covered_cell_signatures"] = tuple(tuple(cell) for cell in covered)
    return table, issuer
