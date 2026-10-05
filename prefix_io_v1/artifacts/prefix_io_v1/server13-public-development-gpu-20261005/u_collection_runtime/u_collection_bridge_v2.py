"""CPU source/input boundary for an uncalibrated original development U run.

This consumer can collect genuine current-device native observations only.
It cannot issue a private cost table, calibrate an old GPU, authorize I, or
turn source inheritance into a current-machine full-source hash claim.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path


SCHEMA = "uncalibrated_original_U_development_collection_v1"
GATE_SCHEMA = "uncalibrated_original_U_collection_CPU_gate_v1"
PROOF_SCHEMA = "migrated_u_collection_source_proof_v1"
ANCESTOR_LOCK = dict(
    path="artifacts/prefix_io_v1/server12-natural-source-audit-20261005/PRERENT_SOURCE_LOCK_V14.json",
    bytes=1164001, sha256="427c796365d31a5ff7a725f2bfa52641a74a9c609b67fda918fa23834af343b4")
ANCESTOR_PROOF = dict(
    path="artifacts/prefix_io_v1/server12-natural-source-audit-20261005/PRERENT_SOURCE_PROOF_V14.json",
    bytes=1558, sha256="e5392891a83f60c436e708aac85ebcca99717a52a185183ef573692c3b2cf8fd")
FALSE_FIELDS = {"table_issued", "cost_qualified", "ordinary_I_authorized", "formal_goodput_allowed"}
DESCRIPTOR_FIELDS = {"schema", "gpu_uuid", "common_runtime_domain_sha256", "workload_ref",
    "formal_trace_binding_ref", "pair_config_ref", "collector_ref", "original_collector_source_ref",
    "reserve_join_source_ref", "native_journal_source_ref", "site_sdk_adapter_ref", "site_sdk_migration_ref"} | FALSE_FIELDS
PROOF_FIELDS = {"schema", "status", "ancestor_source_lock_ref", "ancestor_source_proof_ref",
    "source_lock_ref", "source_count", "ancestor_source_count", "targeted_source_refs",
    "required_source_refs", "actual_gpu_runs", "full_source_verified", "current_host_whole_source_hash_performed"}
COLLECTOR_SHA = "a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0"
ORIGINAL_COLLECTOR_SHA = "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"
RESERVE_JOIN_SHA = "92996cb217bb11a85e237991cb82b37c1b1d10e26f65889338f030a0d46342ff"
HOST_SHA = "04b618db4e816f30231164a503adcdc2a55369f603350f107c0e48c90ec985ec"
JOURNAL_SHA = "3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3"
SITE_SDK_ADAPTER_SHA = "3fd20b032c16178da8cd2a46e89a74f93cec8ff7464fbf4e5fe689e6be5b1d0f"
SDK_MIGRATION_AUDIT = dict(
    path="artifacts/prefix_io_v1/server13-public-development-gpu-20261005/ACTUAL_DRIVER_MIGRATION_AUDIT_01.json",
    bytes=1413, sha256="9874640f0ab80d847d6adcf77dd71c7f6e5c540ed782d855473cf0cfa3818a3c")


def require(value, reason):
    if not value:
        raise ValueError("UNCALIBRATED_U_COLLECTION_REJECTED: " + reason)


def require_role(config):
    require(type(config) is dict and (config.get("phase"), config.get("mode"), config.get("arm")) ==
            ("development", "off", "U"), "only original development U/off collection")
    require(type(config.get("collection_ref")) is dict and "activation_ref" not in config and
            type(config.get("formal_trace_binding_ref")) is dict and
            config.get("formal_peer_closed_ref") is None and config.get("off_qualification_ref") is None,
            "new real collection reference; no old calibration or prerequisite promotion")
    require(all(config.get(key) is None for key in ("service_SLO", "independent_deadline_ref", "authority_ref",
            "full_control_window_deadline_ns")), "collection never invents a service/deadline authority")
    return "development", "off", "U"


def _closed(root, row, refs, *, driver):
    require(type(row) is dict and refs.get(row.get("path")) == row, "actual leaf in current frozen source closure")
    return driver.check_ref(root, row)


def gate(root, config, refs, pair, *, driver):
    """Read-only source/domain gate; safe before AND after namespace creation.

    Namespace/input freshness is a separate original replay before the model.
    In particular, this does not import an activation/private issuer module.
    """
    if "collection_ref" not in config:
        return None
    require_role(config)
    value = driver.read(_closed(root, config["collection_ref"], refs, driver=driver))
    require(type(value) is dict and set(value) == DESCRIPTOR_FIELDS and value.get("schema") == SCHEMA,
            "exact new uncalibrated U collection descriptor, never an activation receipt")
    require(all(value.get(key) is False for key in FALSE_FIELDS), "collection has no table/cost/I/goodput authority")
    require(value.get("gpu_uuid") == config["gpu_uuid"] and
            value.get("common_runtime_domain_sha256") == driver.common_domain_sha(pair),
            "actual current-device unchanged original common domain")
    for key in ("workload_ref", "formal_trace_binding_ref", "pair_config_ref"):
        require(value.get(key) == config[key], "same real collection/config input: " + key)
        _closed(root, value[key], refs, driver=driver)
    require(driver.read(driver.check_ref(root, config["pair_config_ref"]))["configurations"] == pair,
            "actual frozen original strong pair, not supplied fixture metadata")
    binding = driver.read(driver.check_ref(root, value["formal_trace_binding_ref"]))
    require(binding.get("schema") == "development_natural_trace_binding_v1" and
            "independent_deadline_ref" not in binding, "real separate no-SLO development input consumer")
    directory = Path(config["runtime_ref"]["path"]).parent
    expected = {
        "collector_ref": ((directory / "bounded_native_full_step_collector_v2.py").as_posix(), 13016, COLLECTOR_SHA),
        "original_collector_source_ref": ((directory / "bounded_native_full_step_collector.py").as_posix(), 25070,
                                           ORIGINAL_COLLECTOR_SHA),
        "reserve_join_source_ref": ((directory.parent / "activation/control_observation/reserve_join.py").as_posix(),
                                    32772, RESERVE_JOIN_SHA),
        "native_journal_source_ref": ((directory.parent / "activation/source/prefix_io_control/p4_native_window_journal.py").as_posix(),
                                      9647, JOURNAL_SHA),
    }
    for key, (path, size, sha) in expected.items():
        require(value.get(key) == dict(path=path, bytes=size, sha256=sha), "same exact original observation source: " + key)
        _closed(root, value[key], refs, driver=driver)
    host_path = Path(value["reserve_join_source_ref"]["path"]).with_name("host_control_observer.py").as_posix()
    require(refs.get(host_path) == dict(path=host_path, bytes=18796, sha256=HOST_SHA),
            "same original pure host-observation source")
    driver.check_ref(root, refs[host_path])
    sdk_relative = (directory.parent / "sdk_migration/site_sdk_migration.py").as_posix()
    sdk_row = value.get("site_sdk_adapter_ref")
    require(type(sdk_row) is dict and sdk_row.get("path") == sdk_relative and
            sdk_row.get("sha256") == SITE_SDK_ADAPTER_SHA,
            "exact new-U scoped SDK migration adapter; never the old 580-host qualification")
    _closed(root, sdk_row, refs, driver=driver)
    require(value.get("site_sdk_migration_ref") == SDK_MIGRATION_AUDIT,
            "actual read-only current 595 driver migration audit, not a new compiler receipt")
    _closed(root, value["site_sdk_migration_ref"], refs, driver=driver)
    policy = pair["U"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["prefix_io_p4_policy"]
    require(policy.get("mode") == "off", "unchanged original U policy disabled")
    return dict(schema=GATE_SCHEMA, descriptor=deepcopy(value), runtime_refs=refs,
                gpu_uuid=config["gpu_uuid"], common_runtime_domain_sha256=driver.common_domain_sha(pair),
                collector_ref=deepcopy(value["collector_ref"]), table_issued=False, cost_qualified=False,
                ordinary_I_authorized=False, formal_goodput_allowed=False)


def bind_phase(module, *, config, refs, driver):
    """Use the existing real input consumer, without its old calibration phase.

    The caller already bound exact original F, real namespace and development
    input replay. All input/author/tokenizer/family/model checks run unchanged.
    """
    require_role(config)
    inputs = module.validate_formal_workload
    def phase(actual_config, *, root, refs, pair, formal_workload, finite_activation=None, development_replay=None):
        require(require_role(actual_config) == require_role(config) and actual_config == config,
                "same actual source-closed U collection role/config")
        require(finite_activation is None and development_replay is None,
                "uncalibrated collection cannot consume old private table or reserve qualification")
        actual = inputs(module.json_leaf(root, actual_config["workload_ref"], refs)[1], root=root,
            workload_ref=actual_config["workload_ref"], binding_ref=actual_config["formal_trace_binding_ref"],
            refs=refs, pair=pair, partition="development")
        module.require(module.exact(actual, formal_workload), "full real partition/IDs/family/model/namespace replay")
        require(actual.get("schema") == "development_natural_token_id_partition_CPU_binding_v1" and
                actual.get("service_SLO") is None and actual.get("independent_deadline_ref") is None and
                actual.get("gpu_eligible") is False and actual.get("partition") == "development" and
                actual.get("complete_selected_record_count") == 5,
                "complete prospective five-row development partition; no selected subset")
        require(actual_config["run_id"] == pair["U"]["engine"]["kv_transfer_config"]
                ["kv_connector_extra_config"]["prefix_io_parent_admission"]["run_id"], "actual original run/journal identity")
        gate(root, actual_config, refs, pair, driver=driver)
        return dict(schema="formal_trace_phase_CPU_preflight_v1", role=["development", "off", "U"],
            partition="development", input_bindings_validated=True, same_strong_U_I_domain=True,
            independent_service_SLO=None, development_diagnostic_only=True, uncalibrated_u_collection=True,
            table_issued=False, cost_qualified=False, ordinary_I_authorized=False, effect_budget_reuse_allowed=False,
            cpu_metadata_only=True, gpu_eligible=False, formal_goodput_allowed=False, actual_GPU_operations=0,
            required_runtime_gates=["original_guard_and_live_device", "original_full_CUDA_native_host_capture",
                "native_shutdown_and_session_drain", "all_development_requests_and_IO_accounted"])
    module.validate_formal_phase = phase
    return module


def verify_migrated_source_proof(root, config, refs, proof, *, driver):
    """Verify exact historical full proof plus targeted current-machine bytes.

    No model leaf is rehashed here. Current new leaves and the existing runner's
    REQUIRED list are hashed, while every historical V14 row remains identical.
    This boundary is explicitly unusable by I/effect or the ordinary full proof.
    """
    require_role(config)
    require(type(proof) is dict and set(proof) == PROOF_FIELDS and proof.get("schema") == PROOF_SCHEMA and
            proof.get("status") == "PASS_INHERITED_V14_PLUS_TARGETED_CPU_BYTES" and
            proof.get("source_lock_ref") == config["source_lock_ref"], "exact distinct migrated source proof")
    require(proof.get("ancestor_source_lock_ref") == ANCESTOR_LOCK and
            proof.get("ancestor_source_proof_ref") == ANCESTOR_PROOF, "exact actually verified V14 ancestry bytes")
    old_lock = driver.read(driver.check_ref(root, ANCESTOR_LOCK))
    old_proof = driver.read(driver.check_ref(root, ANCESTOR_PROOF))
    require(old_proof.get("schema") == "strong_trace_source_proof_v1" and
            old_proof.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and
            old_proof.get("source_lock_ref") == ANCESTOR_LOCK and
            old_proof.get("source_count") == 5014 and type(old_proof.get("source_count")) is int and
            old_proof.get("actual_gpu_runs") == 0 and type(old_proof.get("actual_gpu_runs")) is int,
            "genuine prior full CPU byte proof, never current GPU authority")
    rows = old_lock.get("files")
    require(type(rows) is list and len(rows) == 5014, "complete prior V14 source rows")
    inherited = {}
    for row in rows:
        require(type(row) is dict and type(row.get("path")) is str and row["path"] not in inherited,
                "unique inherited original source row")
        require(refs.get(row["path"]) == row, "all inherited source/model/SDK rows remain byte-identical")
        inherited[row["path"]] = row
    require(proof.get("source_count") == len(refs) and type(proof.get("source_count")) is int and
            proof.get("ancestor_source_count") == 5014 and type(proof.get("ancestor_source_count")) is int and
            proof.get("actual_gpu_runs") == 0 and type(proof.get("actual_gpu_runs")) is int and
            proof.get("full_source_verified") is False and proof.get("current_host_whole_source_hash_performed") is False,
            "honest inherited versus current targeted verification; no invented whole-host hash")
    added = [refs[path] for path in sorted(set(refs) - set(inherited))]
    required = [refs[path] for path in driver.REQUIRED]
    require(type(proof.get("targeted_source_refs")) is list and proof["targeted_source_refs"] == added and
            type(proof.get("required_source_refs")) is list and proof["required_source_refs"] == required,
            "complete added/current-required targeted byte denominator and order")
    for row in added + required:
        driver.check_ref(root, row)
    return dict(schema="actual_inherited_V14_targeted_CPU_source_replay_v1",
        ancestor_source_count=5014, source_count=len(refs), targeted_added_source_count=len(added),
        required_source_count=len(required), current_host_whole_source_hash_performed=False,
        full_source_verified=False, cost_qualified=False, ordinary_I_authorized=False, actual_GPU_operations=0)
