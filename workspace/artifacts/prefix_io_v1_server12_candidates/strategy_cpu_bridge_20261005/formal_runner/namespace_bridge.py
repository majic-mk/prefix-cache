"""CPU adapters for the unchanged formal trace validator's partition lifetime.

Only an original guard-closed, byte-closed peer can occupy its namespace.
The selected arm is always fresh. No path is removed, reset or substituted.
The original protocol permits SLO=None for development diagnostics; evaluation
retains the original mandatory independent service targets.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time


def verify_closed_peer(root, config, refs, pair, *, partition, peer_arm, peer_storage, manifest, driver):
    row = config.get("formal_peer_closed_ref")
    driver.require(type(row) is dict and refs.get(row.get("path")) == row,
                   "actual source-closed U peer proof required; namespace existence is no qualification")
    parent = driver.read(driver.check_ref(root, row))
    return verify_closed_peer_document(parent, root, config, refs, pair, partition=partition,
        peer_arm=peer_arm, peer_storage=peer_storage, manifest=manifest, driver=driver)


def verify_closed_peer_document(parent, root, config, refs, pair, *, partition, peer_arm,
                                peer_storage, manifest, driver):
    """Verify an in-memory parent before its single append, with all raw leaves.

    Only the parent itself need not exist yet. No raw child/guard/config/source,
    output/CUDA/native-tail checks are deferred or replaced by flags.
    """
    driver.require(type(parent) is dict, "actual candidate parent document")
    driver.require((partition == "development" and peer_arm == "U") or
                   (partition == "evaluation" and peer_arm in ("U", "I")), "bounded formal native peer role")
    status = ("PASS_FORMAL_DEVELOPMENT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if partition == "development"
              else "PASS_FORMAL_EFFECT_U_WORKLOAD_ONLY_REQUIRES_GUARD_CLOSURE" if peer_arm == "U"
              else "PASS_FINITE_QUALIFIED_I_WORKLOAD_LIFECYCLE_REQUIRES_EFFECT_ANALYSIS")
    mode = "off" if peer_arm == "U" else "on"
    phase = "development" if partition == "development" else "effect"
    driver.require(parent.get("status") == status and parent.get("phase") == phase and
                   parent.get("arm") == peer_arm and parent.get("mode") == mode and
                   parent.get("formal_partition") == partition and
                   parent.get("workload_ref") == config["workload_ref"] and
                   parent.get("formal_trace_binding_ref") == config["formal_trace_binding_ref"] and
                   parent.get("gpu_uuid") == config["gpu_uuid"] and
                   parent.get("common_runtime_domain_sha256") == driver.common_domain_sha(pair) and
                   parent.get("formal_effect_qualified") is False and
                   parent.get("original_engine_shutdown_returned") is True and
                   parent.get("native_tail_drained") is True and parent.get("os_session_drained") is True,
                   "same-partition actual U lifecycle/domain/guard closure before I")
    initial = parent.get("formal_initial_namespace")
    driver.require(type(initial) is dict and set(initial) == {
        "storage_path", "fresh_before_runtime_creation", "whole_partition_no_reset", "run_id", "partition", "arm"} and
        initial["storage_path"] == peer_storage and initial["fresh_before_runtime_creation"] is True and
        initial["whole_partition_no_reset"] is True and initial["run_id"] == parent.get("run_id") and
        initial["partition"] == partition and initial["arm"] == peer_arm,
        "actual U partition began fresh and retained its original namespace")
    leaves = {}
    for key in ("child_result_ref", "completed_guard_ref", "actual_run_config_ref"):
        ref = parent.get(key)
        driver.require(type(ref) is dict and refs.get(ref.get("path")) == ref,
                       "actual U parent/raw child/guard/config byte closure: " + key)
        leaves[key] = driver.read(driver.check_ref(root, ref))
    child, guard, prior_config = (leaves[k] for k in ("child_result_ref", "completed_guard_ref", "actual_run_config_ref"))
    driver.require(type(child) is dict and child.get("os_session_drained") is False and
                   all(parent.get(key) == value for key, value in child.items() if key != "os_session_drained") and
                   set(parent) - set(child) <= {"completed_guard_ref", "child_result_ref", "formal_parent_closure_schema"},
                   "parent must retain actual child bytes; only old guard closes the OS session")
    driver.require(prior_config.get("schema") == driver.SCHEMA and prior_config.get("run_id") == parent["run_id"] and
                   prior_config.get("phase") == phase and prior_config.get("arm") == peer_arm and
                   prior_config.get("mode") == mode and
                   all(prior_config.get(key) == parent.get(key) for key in
                       ("runner_ref", "runtime_ref", "source_lock_ref", "workload_ref", "pair_config_ref", "formal_trace_binding_ref")) and
                   prior_config.get("gpu_uuid") == config["gpu_uuid"], "actual guarded U config identities")
    prior_rows = driver.source_rows(root, prior_config["source_lock_ref"], full=False)
    driver.require(all(refs.get(path) == leaf for path, leaf in prior_rows.items()),
                   "all immutable U execution source rows inherited without replacement")
    for key in ("runner_ref", "runtime_ref", "workload_ref", "pair_config_ref", "formal_trace_binding_ref", "permissions_ref", "activation_ref"):
        leaf = prior_config.get(key)
        driver.require(type(leaf) is dict and refs.get(leaf.get("path")) == leaf, "actual U config leaf closure: " + key)
        driver.check_ref(root, leaf)
    driver.require(manifest == driver.read(driver.check_ref(root, prior_config["workload_ref"])),
                   "peer verifier consumes the exact actual complete manifest bytes")
    prior_pair = driver.read(driver.check_ref(root, prior_config["pair_config_ref"]))["configurations"]
    driver.require(driver.common_domain_sha(prior_pair) == driver.common_domain_sha(pair) and
                   prior_pair[peer_arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"] == peer_storage,
                   "actual U ran this retained peer namespace and common original domain")
    validator_path = driver.PREVIOUS + "/calibration_v2/native_conditional_cost.py"
    driver.require(validator_path in refs, "original closed-guard validator frozen")
    validator = driver.load(root, refs[validator_path], "_formal_namespace_guard_" + str(time.monotonic_ns()))
    validator.validate_guard(guard, gpu_uuid=config["gpu_uuid"], job_id=parent["run_id"],
                             wrapper_path=prior_config["runner_ref"]["path"])
    driver.require(guard.get("command") == driver.child_command(root, parent["actual_run_config_ref"]["path"],
                   prior_config["runner_ref"]["path"]) and guard.get("permissions") == prior_config["permissions_ref"] and
                   guard.get("reservation_id") == parent.get("guard_reservation_id"),
                   "original guard command/config/permissions/reservation matched actual U child")
    runtime = driver.load(root, prior_config["runtime_ref"], "_formal_namespace_raw_U_" + str(time.monotonic_ns()))
    runtime.verify_formal_peer_native_prerequisite(root, parent, config=prior_config, refs=refs, driver=driver)
    selected = [record for record in manifest["records"] if record["split"] == partition]
    driver.validate_formal_frontend_records(
        dict(config=config, formal_workload_binding=dict(records=selected, partition=partition,
             original_manifest_workload_sha256=manifest["workload_sha256"])), parent["frontend"])
    return parent


def bind(module, *, config, refs, pair, driver):
    """Install only scoped metadata adapters in this fresh private F instance."""
    role = (config["phase"], config["mode"], config["arm"])
    partition = driver.FORMAL_ROLES[role]
    old_service = module.require_independent_service_SLO
    def service(value):
        if value is None and partition == "development":
            return None
        return old_service(value)
    module.require_independent_service_SLO = service

    def namespace(document, *, root, manifest, pair, partition):
        fields = {"schema", "workload_sha256", "model_manifest_sha256", "tokenizer_receipt_digest",
                  "common_runtime_domain_sha256", "initial_cache_state", "no_per_request_reset", "partition_namespaces"}
        module.require(type(document) is dict and set(document) == fields and
                       document["schema"] == "formal_trace_namespace_contract_v1", "closed formal namespace contract")
        for key in ("workload_sha256", "model_manifest_sha256", "tokenizer_receipt_digest"):
            module.require(document[key] == manifest[key], "namespace bound to these exact formal input identities")
        module.require(document["common_runtime_domain_sha256"] == module.common_domain_sha(pair) and
                       document["initial_cache_state"] == module.INITIAL and document["no_per_request_reset"] is True,
                       "same actual strong U/I domain and whole-partition lifetime")
        namespaces = document["partition_namespaces"]
        module.require(type(namespaces) is dict and set(namespaces) == set(module.PARTITIONS), "all three independent namespace partitions")
        seen = set()
        for name in module.PARTITIONS:
            arms = namespaces[name]
            module.require(type(arms) is dict and set(arms) == {"U", "I"}, "distinct namespaces for both existing arms")
            for arm, absolute in arms.items():
                module.require(type(absolute) is str and Path(absolute).is_absolute(), "actual absolute namespace")
                try:
                    relative = Path(absolute).relative_to(Path(root).resolve(strict=True)).as_posix()
                except ValueError:
                    module.require(False, "namespace outside project")
                module.require(relative.startswith("experiments/prefix_io_v1/runs/"), "original bounded private runs only")
                path = module.safe(root, relative)
                module.require(absolute not in seen, "partition/arm namespace alias")
                seen.add(absolute)
                if name == partition:
                    actual = pair[arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"]
                    module.require(absolute == actual, "selected whole-partition actual U/I namespaces")
                    if arm == config["arm"]:
                        module.require(not path.exists(), "fresh selected current-arm whole-partition namespace")
                    elif config.get("formal_peer_closed_ref") is not None:
                        module.require(path.is_dir(), "completed original peer namespace must be retained")
                        verify_closed_peer(root, config, refs, pair, partition=partition, peer_arm=arm,
                                           peer_storage=absolute, manifest=manifest, driver=driver)
                    else:
                        module.require(not path.exists(), "both original first-arm paired namespaces begin fresh")
        return namespaces[partition]
    module.validate_namespace = namespace
    return module
