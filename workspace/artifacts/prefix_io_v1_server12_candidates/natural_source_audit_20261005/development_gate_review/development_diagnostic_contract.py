"""CPU prototype of a development-only natural input consumer.

No engine, tokenizer, private table, GPU launch or runtime activation is here.
The exact original input function retains every real source/tokenizer/family/
manifest/namespace check. Only its external deadline/authority stanza is omitted
for development Uoff or Ishadow; effect callers are rejected before loading it.
This file is a proposal prototype, not an installed GPU entry point.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib


FORMAL_SHA = "bab370bf83cf4736b03dc3fdc80f07b6efc40b15425ed43e3581fbcfe14f2d9b"
DEVELOPMENT_SCHEMA = "development_natural_trace_binding_v1"
OUTPUT_SCHEMA = "development_natural_token_id_partition_CPU_binding_v1"
ALLOWED_ROLES = {("development", "off", "U"), ("development", "shadow", "I")}


def require(condition, reason):
    if not condition:
        raise ValueError("DEVELOPMENT_DIAGNOSTIC_REJECTED: " + reason)


def check_role(config):
    require(type(config) is dict and (config.get("phase"), config.get("mode"), config.get("arm")) in ALLOWED_ROLES,
            "development Uoff/Ishadow only; effect and ordinary on never enter this adapter")
    require(config.get("service_SLO") is None, "diagnostic carries no service SLO")
    require(all(config.get(key) is None for key in
                ("independent_deadline_ref", "authority_ref", "full_control_window_deadline_ns")),
            "diagnostic does not construct or relabel an external service authority/deadline")
    return config["phase"], config["mode"], config["arm"]


def preview_budget_from_existing_pair(config, pair):
    """Use only an existing frozen engineering setting, never invent a number."""
    check_role(config)
    value = pair["I"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["prefix_io_p4_policy"]["internal_step_budget_ns"]
    require(type(value) is int and value > 0, "existing fixed engineering preview threshold required")
    return dict(internal_step_budget_ns=value, origin="existing_frozen_pair_engineering_preview_threshold",
                service_SLO=None, independent_deadline_ref=None, observation_only=True,
                ordinary_I_authorized=False, reserve_not_measured_yet=True,
                preview_is_service_safety_claim=False, formal_goodput_allowed=False,
                effect_budget_reuse_allowed=False, gpu_eligible=False)


def build_consumer(original_source, *, config):
    """Derive one bounded input function from the byte-pinned original source.

    Nine known deadline/authority statements are the only removed statements.
    The function is not an executor. Its full original freeze_trace replay,
    all receipt/source/model/family checks and namespace callback remain intact.
    A new descriptor/output schema prevents this result from being consumed by
    the unchanged evaluation/finite activation contract.
    """
    check_role(config)
    require(type(original_source) is bytes and hashlib.sha256(original_source).hexdigest() == FORMAL_SHA,
            "exact original formal input source bytes")
    tree = ast.parse(original_source)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_formal_workload")
    require(len(original.body) == 43 and original.body[26].lineno == 321 and original.body[35].lineno == 335,
            "exact source-pinned external-deadline stanza boundaries")
    removed = original.body[26:35]
    require(len(removed) == 9 and "independent_deadline_ref" in ast.unparse(removed[0]) and
            "authority" in ast.unparse(removed[-1]), "only original external deadline/authority stanza")
    function = deepcopy(original)
    function.name = "_development_source_input_only"
    function.body = function.body[:26] + function.body[35:]
    changed_schema = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and node.value == "formal_natural_trace_binding_v1":
            node.value = DEVELOPMENT_SCHEMA
            changed_schema += 1
    require(changed_schema == 1, "one descriptor schema replacement")
    returned = function.body[-1]
    require(isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call), "original bounded input metadata return")
    for keyword in returned.value.keywords:
        if keyword.arg == "schema": keyword.value = ast.Constant(OUTPUT_SCHEMA)
        elif keyword.arg == "origin": keyword.value = ast.Constant("actual_byte_closed_original_development_input_replay_CPU_only")
        elif keyword.arg in ("independent_deadline_ref", "service_SLO"): keyword.value = ast.Constant(None)
    # Never accept a caller-provided helper namespace or an unverified .pyc.
    # The prototype uses the unchanged original namespace checker; connecting
    # the existing current-arm/guard-closed-peer adapter is a separate, still
    # unimplemented production integration step.
    namespace = {"__name__": "_byte_pinned_original_development_input_helpers"}
    exec(compile(original_source, "<byte_pinned_original_formal_input_helpers>", "exec", dont_inherit=True), namespace)
    namespace["BINDING_FIELDS"] = namespace["BINDING_FIELDS"] - {"independent_deadline_ref"}
    compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(compiled, "<byte_pinned_development_input_consumer>", "exec", dont_inherit=True), namespace)
    consumer = namespace[function.name]
    return consumer, dict(schema="development_input_adapter_derivation_CPU_v1", original_source_sha256=FORMAL_SHA,
        original_body_statements=43, removed_external_deadline_statements=9, retained_original_statements=34,
        original_engine_functions_modified=0, actual_GPU_operations=0, GPU_launch_allowed=False,
        descriptor_schema=DEVELOPMENT_SCHEMA, result_schema=OUTPUT_SCHEMA,
        evaluation_or_effect_contract_modified=False,
        actual_runtime_namespace_adapter_integrated=False,
        existing_original_pure_helper_loader_retained=True)


def validate_development_inputs(document, *, root, workload_ref, binding_ref, refs, pair,
                                config, original_source):
    check_role(config)
    # Real byte-closed input is still mandatory. No source/model library is
    # imported and no input receipt is made when the actual manifest is absent.
    if document is None or workload_ref is None or binding_ref is None:
        return dict(schema=OUTPUT_SCHEMA, status="UNBOUND_NO_ACTUAL_DEVELOPMENT_INPUT",
                    service_SLO=None, independent_deadline_ref=None, gpu_eligible=False,
                    formal_goodput_allowed=False, actual_GPU_operations=0)
    consumer, audit = build_consumer(original_source, config=config)
    result = consumer(document, root=root, workload_ref=workload_ref, binding_ref=binding_ref,
                      refs=refs, pair=pair, partition="development")
    require(result["schema"] == OUTPUT_SCHEMA and result["partition"] == "development" and
            result["gpu_eligible"] is False and result["formal_goodput_allowed"] is False and
            result["service_SLO"] is None and result["independent_deadline_ref"] is None,
            "CPU development metadata cannot become evaluation/ordinary-I authority")
    require(all(type(row["min_tokens"]) is int and row["min_tokens"] == 128 and
                type(row["max_tokens"]) is int and row["max_tokens"] == 128 for row in result["records"]),
            "current complete native capture contract retains128 outputs")
    return dict(result, derivation_audit=audit, effect_budget_reuse_allowed=False,
                actual_runtime_source_integration_complete=False)
