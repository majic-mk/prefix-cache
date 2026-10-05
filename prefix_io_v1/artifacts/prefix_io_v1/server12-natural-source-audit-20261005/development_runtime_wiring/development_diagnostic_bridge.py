"""Development-only input/activation dispatch around exact original consumers.

No executor, device, cache owner, table constructor, model or backend is here.
An actual no-SLO natural development manifest still passes all original source,
author/tokenizer/family/model/domain/namespace checks. Its private calibration
table is issued by the unchanged original runtime issuer after the real guard.
Effect/evaluation and ordinary I are never accepted by these consumers.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace


FORMAL_SHA = "bab370bf83cf4736b03dc3fdc80f07b6efc40b15425ed43e3581fbcfe14f2d9b"
ACTIVATION_SHA = "30d62062a7297e90a18ff9691ce08be4d286a29266817d1a114b3bd9a4c9065a"
RAW_VALIDATOR_SHA = "675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
BINDING_SCHEMA = "development_natural_trace_binding_v1"
OUTPUT_SCHEMA = "development_natural_token_id_partition_CPU_binding_v1"
ACTIVATION_SCHEMA = "development_diagnostic_gpu_activation_request_v1"
ROLES = {("development", "off", "U"), ("development", "shadow", "I")}
PURE_IMPORTS = {"__future__", "argparse", "ast", "copy", "hashlib", "json", "math", "random",
                "re", "dataclasses", "pathlib", "typing"}
INPUT_REF_FIELDS = {"manifest_ref", "dataset_ref", "author_trace_ref", "author_common_ref",
                    "protocol_source_ref", "declaration_ref", "tokenizer_receipt_ref", "model_manifest_ref",
                    "namespace_contract_ref", "strong_pair_validator_ref"}
ACTIVATION_REF_FIELDS = {"plan_ref", "measurements_ref", "completed_guard_ref", "intent_ref",
    "calibration_source_lock_ref", "collector_ref", "development_reserve_ref", "development_budget_ref",
    "protocol_source_ref", "issuer_source_ref", "cost_table_source_ref", "finite_binding_ref", "startup_ref",
    "reserve_verification_ref", "reserve_join_source_ref", "development_plan_ref", "development_guard_ref",
    "development_observation_ref", "development_native_result_ref"}
POST_DEVELOPMENT = {"development_reserve_ref", "development_budget_ref", "reserve_verification_ref",
                   "development_plan_ref", "development_guard_ref", "development_observation_ref", "development_native_result_ref"}
DIAGNOSTIC_BUDGET_FIELDS = {"internal_step_budget_ns", "engineering_pair_ref", "origin", "service_SLO",
    "observation_only", "ordinary_I_authorized", "reserve_not_measured_yet", "preview_is_service_safety_claim",
    "effect_budget_reuse_allowed", "formal_goodput_allowed"}


def require(value, reason):
    if not value:
        raise ValueError("DEVELOPMENT_DIAGNOSTIC_REJECTED: " + reason)


def require_role(config):
    require(type(config) is dict and (config.get("phase"), config.get("mode"), config.get("arm")) in ROLES,
            "development Uoff/Ishadow only; no evaluation/effect/ordinary I authority")
    require(config.get("service_SLO") is None and
            all(config.get(key) is None for key in
                ("independent_deadline_ref", "authority_ref", "full_control_window_deadline_ns")),
            "no external service SLO, authority or deadline is created or relabeled")
    return config["phase"], config["mode"], config["arm"]


def prepare_binding_descriptor(root, input_refs, *, refs, config, driver):
    """Create only a byte-closed descriptor from real caller-supplied leaves.

    The complete original replay still must run after the descriptor is frozen.
    Missing assets stay UNBOUND; no deadline, tokenizer receipt or namespace is
    manufactured here. The caller owns append-only writing and source freezing.
    """
    require_role(config)
    require(type(input_refs) is dict and set(input_refs) == INPUT_REF_FIELDS,
            "UNBOUND: exact real natural input reference fields; no external deadline field")
    for key, row in input_refs.items():
        driver.require(type(row) is dict and refs.get(row.get("path")) == row,
                       "UNBOUND actual input leaf: " + key)
        driver.check_ref(root, row)
    return dict(descriptor=dict(schema=BINDING_SCHEMA, **deepcopy(input_refs)),
                schema="CPU_development_binding_descriptor_preparation_v1", ready_for_GPU=False,
                original_full_input_replay_required=True, actual_GPU_operations=0)


def prepare_activation_descriptor(root, activation_refs, *, refs, config, driver):
    """Close genuine calibration/source leaves; no deadline or reserve created."""
    require_role(config)
    require(type(activation_refs) is dict and set(activation_refs) == ACTIVATION_REF_FIELDS,
            "UNBOUND: exact real calibration/private-source references; no external deadline field")
    for key, row in activation_refs.items():
        if key in POST_DEVELOPMENT:
            require(row is None, "diagnostic cannot claim a measured reserve or formal effect budget")
        else:
            driver.require(type(row) is dict and refs.get(row.get("path")) == row,
                           "UNBOUND actual calibration leaf: " + key)
            driver.check_ref(root, row)
    return dict(descriptor=dict(schema=ACTIVATION_SCHEMA, **deepcopy(activation_refs)),
                schema="CPU_development_activation_descriptor_preparation_v1", ready_for_GPU=False,
                original_calibration_guard_and_private_issue_replay_required=True, actual_GPU_operations=0)


def is_diagnostic(root, config, refs, *, driver):
    row = config.get("formal_trace_binding_ref") if type(config) is dict else None
    if row is None:
        return False
    driver.require(type(row) is dict and refs.get(row.get("path")) == row,
                   "actual frozen input descriptor; no schema flag authority")
    descriptor = driver.read(driver.check_ref(root, row))
    if descriptor.get("schema") != BINDING_SCHEMA:
        return False
    require_role(config)
    return True


def _pinned_source(root, row, refs, expected, *, driver):
    driver.require(type(row) is dict and refs.get(row.get("path")) == row and row.get("sha256") == expected,
                   "exact unchanged original adapter dependency")
    raw = driver.check_ref(root, row).read_bytes()
    driver.require(hashlib.sha256(raw).hexdigest() == expected and len(raw) == row["bytes"],
                   "actual original bytes before bounded AST dispatch")
    return raw


def source_only_pure_loader(module):
    """Retain original F leaf/import rules but execute only its frozen bytes."""
    def load(root, row, refs, expected_sha):
        row, path = module.closed(root, row, refs)
        module.require(row["sha256"] == expected_sha and path.suffix == ".py" and row["bytes"] <= 1024**2,
                       "exact existing pure CPU protocol/helper source")
        raw = path.read_bytes()
        module.require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                       "actual source-only helper bytes")
        tree = ast.parse(raw)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module.require(all(alias.name in PURE_IMPORTS for alias in node.names), "pure helper import boundary")
            elif isinstance(node, ast.ImportFrom):
                module.require(node.module in PURE_IMPORTS and node.level == 0, "pure helper import boundary")
        name = "_actual_source_only_development_helper_" + str(time.monotonic_ns())
        spec = module.importlib.util.spec_from_file_location(name, path)
        loaded = module.importlib.util.module_from_spec(spec)
        module.require(name not in sys.modules, "fresh pure CPU helper")
        sys.modules[name] = loaded
        try:
            exec(compile(raw, str(path), "exec", dont_inherit=True), loaded.__dict__)
            module.closed(root, row, refs)
            return loaded
        finally:
            sys.modules.pop(name, None)
    return load


def _development_input(module, raw):
    original = next(node for node in ast.parse(raw).body
                    if isinstance(node, ast.FunctionDef) and node.name == "validate_formal_workload")
    require(len(original.body) == 43 and original.body[26].lineno == 321 and original.body[35].lineno == 335,
            "exact original input/deadline boundaries")
    function = deepcopy(original)
    function.name = "_actual_development_input_replay"
    function.body = function.body[:26] + function.body[35:]
    replaced = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and node.value == "formal_natural_trace_binding_v1":
            node.value = BINDING_SCHEMA
            replaced += 1
    require(replaced == 1, "one distinct development descriptor schema")
    returned = function.body[-1]
    require(isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call), "original input metadata only")
    for item in returned.value.keywords:
        if item.arg == "schema": item.value = ast.Constant(OUTPUT_SCHEMA)
        elif item.arg == "origin": item.value = ast.Constant("actual_byte_closed_original_development_input_replay_CPU_only")
        elif item.arg in ("independent_deadline_ref", "service_SLO"): item.value = ast.Constant(None)
    module.BINDING_FIELDS = module.BINDING_FIELDS - {"independent_deadline_ref"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "<exact_original_development_input_dispatch>", "exec", dont_inherit=True), module.__dict__)
    return module.__dict__[function.name]


def bind_input(module, *, root, source_ref, config, refs, driver):
    """Called after original current-arm/actual closed-peer namespace bind."""
    require_role(config)
    raw = _pinned_source(root, source_ref, refs, FORMAL_SHA, driver=driver)
    module.load_pure = source_only_pure_loader(module)
    actual = _development_input(module, raw)
    bound_role = require_role(config)
    def inputs(document, *, root, workload_ref, binding_ref, refs, pair, partition):
        require(require_role(config) == bound_role and partition == "development",
                "only the full prospective development partition")
        result = actual(document, root=root, workload_ref=workload_ref, binding_ref=binding_ref,
                        refs=refs, pair=pair, partition=partition)
        require(result.get("schema") == OUTPUT_SCHEMA and result.get("service_SLO") is None and
                result.get("independent_deadline_ref") is None and result.get("gpu_eligible") is False,
                "development metadata remains no GPU/effect authority")
        return result
    module.validate_formal_workload = inputs
    def phase(actual_config, *, root, refs, pair, formal_workload, finite_activation=None, development_replay=None):
        require(require_role(actual_config) == bound_role and actual_config == config,
                "same real development config and bounded role")
        require(development_replay is None, "diagnostic has no effect native-reserve qualification")
        actual_binding = inputs(module.json_leaf(root, actual_config["workload_ref"], refs)[1], root=root,
            workload_ref=actual_config["workload_ref"], binding_ref=actual_config["formal_trace_binding_ref"],
            refs=refs, pair=pair, partition="development")
        module.require(module.exact(actual_binding, formal_workload), "full real development replay; no forged IDs/family/namespace")
        module.require(actual_config["run_id"] == pair[actual_config["arm"]]["engine"]["kv_transfer_config"]
                       ["kv_connector_extra_config"]["prefix_io_parent_admission"]["run_id"], "actual original run/parent identity")
        require(type(finite_activation) is dict and finite_activation.get("development_diagnostic_only") is True and
                finite_activation.get("phase") == "development" and finite_activation.get("actual_table_issued") is False,
                "source-checked diagnostic activation, no CPU-created table authority")
        descriptor = module.json_leaf(root, actual_config["activation_ref"], refs)[1]
        require(module.exact(descriptor, finite_activation.get("descriptor")) and
                descriptor.get("schema") == ACTIVATION_SCHEMA and "independent_deadline_ref" not in descriptor,
                "actual separate development activation bytes; no invented deadline")
        verify_diagnostic_budget(actual_config, finite_activation.get("independent_budget"))
        actual_activation = driver.activation_gate_module(root, actual_config, refs).verify(
            root, actual_config["activation_ref"], refs=refs, pair=pair,
            gpu_uuid=actual_config["gpu_uuid"], driver=driver, phase="development")
        require(module.exact(actual_activation, finite_activation),
                "fresh original calibration/source/guard and frozen engineering budget replay; no forged finite gate")
        plan = finite_activation["calibration_plan"]
        require(plan.get("common_runtime_domain_sha256") == actual_binding["common_runtime_domain_sha256"] and
                plan.get("gpu_uuid") == actual_config["gpu_uuid"], "same real calibrated domain/device")
        return dict(schema="formal_trace_phase_CPU_preflight_v1", role=list(bound_role), partition="development",
            input_bindings_validated=True, same_strong_U_I_domain=True, independent_service_SLO=None,
            development_diagnostic_only=True, ordinary_I_authorized=False, effect_budget_reuse_allowed=False,
            cpu_metadata_only=True, gpu_eligible=False, formal_goodput_allowed=False, actual_GPU_operations=0,
            required_runtime_gates=["original_guard_and_live_device", "original_private_issuer_finite_cost_coverage",
                "native_shutdown_and_session_drain", "all_development_requests_and_IO_accounted"])
    module.validate_formal_phase = phase
    return module


def diagnostic_budget(config, pair):
    role = require_role(config)
    value = None
    if role[2] == "I":
        value = pair["I"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
        value = value["prefix_io_p4_policy"]["internal_step_budget_ns"]
        require(type(value) is int and value > 0,
                "UNBOUND_SHADOW_ENGINEERING_PREVIEW_BUDGET: freeze a real development engineering setting; no value invented")
    result = dict(internal_step_budget_ns=value, engineering_pair_ref=config["pair_config_ref"],
        origin="frozen_actual_pair_engineering_preview_only" if role[2] == "I" else "original_U_off_no_I_preview_budget",
        service_SLO=None, observation_only=True, ordinary_I_authorized=False,
        reserve_not_measured_yet=True, preview_is_service_safety_claim=False,
        effect_budget_reuse_allowed=False, formal_goodput_allowed=False)
    verify_diagnostic_budget(config, result)
    return result


def verify_diagnostic_budget(config, budget):
    role = require_role(config)
    require(type(budget) is dict and set(budget) == DIAGNOSTIC_BUDGET_FIELDS and
            budget.get("origin") == ("original_U_off_no_I_preview_budget" if role[2] == "U"
                                     else "frozen_actual_pair_engineering_preview_only") and
            budget.get("observation_only") is True and
            budget.get("ordinary_I_authorized") is False and budget.get("reserve_not_measured_yet") is True and
            budget.get("preview_is_service_safety_claim") is False and budget.get("effect_budget_reuse_allowed") is False and
            budget.get("formal_goodput_allowed") is False and budget.get("service_SLO") is None and
            budget.get("engineering_pair_ref") == config["pair_config_ref"], "diagnostic budget is no service/reserve/ordinary-I authority")
    value = budget.get("internal_step_budget_ns")
    require(value is None if role[2] == "U" else type(value) is int and value > 0,
            "U has no I preview; shadow has an existing positive engineering threshold")
    return budget


def _activation_consumer(original, raw, config):
    function = next(node for node in ast.parse(raw).body
                    if isinstance(node, ast.FunctionDef) and node.name == "verify")
    require(len(function.body) == 18 and function.body[9].lineno == 41 and function.body[14].lineno == 67,
            "exact unchanged original activation/deadline boundary")
    derived = deepcopy(function)
    derived.name = "_actual_development_activation_replay"
    # All calibration/leaf/source/private-package checks remain exact. The
    # external protocol/deadline/authority/budget stanza alone is replaced by
    # no-SLO engineering observation metadata; U requires no preview number.
    derived.body = derived.body[:9] + ast.parse("budget = _diagnostic_budget(config, pair)").body + derived.body[14:]
    environment = dict(original.__dict__)
    environment.update(SCHEMA=ACTIVATION_SCHEMA, FIELDS=original.FIELDS - {"independent_deadline_ref"},
                       _diagnostic_budget=diagnostic_budget, config=config)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[])),
                 "<exact_original_development_activation_dispatch>", "exec", dont_inherit=True), environment)
    return environment[derived.name]


def bind_activation(original, *, root, source_ref, config, refs, driver):
    require_role(config)
    raw = _pinned_source(root, source_ref, refs, ACTIVATION_SHA, driver=driver)
    actual = _activation_consumer(original, raw, config)
    def verify(root, row, *, refs, pair, gpu_uuid, driver, phase="development"):
        require_role(config)
        require(phase == "development", "development activation cannot be reused by effect")
        pair_ref = config["pair_config_ref"]
        driver.require(refs.get(pair_ref["path"]) == pair_ref and
                       driver.read(driver.check_ref(root, pair_ref))["configurations"] == pair,
                       "actual frozen pair bytes; engineering threshold is not a supplied fixture")
        result = actual(root, row, refs=refs, pair=pair, gpu_uuid=gpu_uuid, driver=driver, phase=phase)
        plan, value = result["calibration_plan"], result["descriptor"]
        require(plan.get("evidence_origin") == "native_runtime_preregistered" and
                plan.get("cpu_preparation_only") is False and plan.get("synthetic_fixture") is False,
                "actual original calibration plan; no old singleton or mock cost qualification")
        measured = driver.read(driver.check_ref(root, value["measurements_ref"]))
        require(measured.get("schema") == "strong_gpu_exact_cell_raw_measurements_v1" and
                measured.get("origin") == "native_gpu_recording" and measured.get("synthetic_fixture") is False and
                measured.get("plan_ref") == value["plan_ref"], "actual original raw GPU costs, not mock/self-declared table flags")
        validator_ref = plan.get("validation_source_ref")
        driver.require(type(validator_ref) is dict and refs.get(validator_ref.get("path")) == validator_ref and
                       validator_ref.get("sha256") == RAW_VALIDATOR_SHA, "original actual calibration/guard verifier source")
        validator = driver.load(root, validator_ref, driver.development_module_name("_development_actual_original_guard_"))
        validator.validate_guard(driver.read(driver.check_ref(root, value["completed_guard_ref"])),
            gpu_uuid=gpu_uuid, job_id=plan["job_id"], wrapper_path=plan["wrapper_source_ref"]["path"])
        return dict(result, development_diagnostic_only=True, ordinary_I_authorized=False,
                    effect_budget_reuse_allowed=False, calibration_raw_origin_verified=True)
    # issue is the SAME original function. It still calls the sole real private
    # issuer with full raw cells/CUDA clocks/guard/output/source replay after the
    # actual guard. No conditional/mock table can be attached here.
    return SimpleNamespace(verify=verify, issue=original.issue, __file__=original.__file__)
