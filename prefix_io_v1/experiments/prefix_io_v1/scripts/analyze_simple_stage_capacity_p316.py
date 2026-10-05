"""Capacity provenance/geometry wrapper around the frozen P316 cohort audit.

This module adds exact registered-domain and real permit checks. The original
golden/source-preservation/drain/accounting/CPU/pressure guards run unchanged.
Reading recorded source proof here does not rehash cache payloads or infer release.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_simple_stage_p3_p316 import audit as baseline_audit
from concurrent_capacity_contract_p316 import (
    DOMAIN_IDS, CANDIDATE_SHA256, QUANTUM, validate_declaration,
    validate as validate_manifest, expected_connector_variant)
from qualify_capacity_p316 import verify_gate
from storage_source_registration import KEYS, ORIGIN_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[3]
DRIVER = "run_concurrent_capacity_p316.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, value):
    result = (Path(root) / Path(value)).resolve()
    require(result.is_relative_to(Path(root).resolve()),
            "capacity audit evidence outside project")
    return result


def option(command, name):
    found = [i for i, value in enumerate(command) if value == name]
    require(len(found) == 1 and found[0] + 1 < len(command),
            "capacity receipt option missing or duplicated: " + name)
    return command[found[0] + 1]


def cohort_receipt(root, result_path, manifest_path, qualification_path,
                   domain_id, config):
    path = result_path.parent.parent / "result.json"
    receipt = json.loads(path.read_text())
    label = result_path.parent.parent.name
    require(receipt.get("label") == label and
            receipt.get("gpu_job_attempted") is True and
            type(receipt.get("exit")) is int and receipt["exit"] == 0 and
            type(receipt.get("child_exit")) is int and receipt["child_exit"] == 0 and
            receipt.get("timed_out") is False and receipt.get("error") is None and
            receipt.get("interrupted_signal") is None and
            receipt.get("session_drained") is True and
            receipt.get("session_members_after_cleanup") == [] and
            receipt.get("gpu_uuid") == config["gpu_uuid"],
            "capacity cohort lacks completed real guarded GPU receipt")
    command = receipt.get("command")
    require(type(command) is list and all(type(x) is str for x in command) and
            sum(Path(x).name == DRIVER for x in command) == 1,
            "capacity cohort was not run through the new capacity driver")
    require(option(command, "--capacity-domain") == domain_id and
            local(root, option(command, "--output")) == result_path.parent and
            local(root, option(command, "--manifest")) == manifest_path and
            local(root, option(command, "--qualification")) == qualification_path and
            "--native-cpu-probe" in command,
            "capacity cohort command/input/CPU provenance differs")
    require(command.count("--native-cpu-probe") == 1,
            "duplicate native CPU probe flag")
    return dict(label=label, path=str(path.relative_to(root)), sha256=digest(path))


def source_provenance(root, report, reference_plan, result_path):
    proof = report.get("source_registration", {})
    require(proof.get("source_tree_exact") is True and
            proof.get("full_source_hashes_verified") is True and
            proof.get("source_materialization_performed") is False and
            proof.get("native_cache_engine_modified") is False and
            proof.get("gpu_operations_performed") is False and
            proof.get("file_count") == 3048 and
            proof.get("gpu_uuid") == reference_plan["gpu_uuid"] and
            Path(proof.get("output_path", "")).resolve() == result_path.parent,
            "recorded PRIMARY source validation missing or mismatched")
    registration_path = local(root, proof["registration_path"])
    raw = json.loads(registration_path.read_text())
    require(type(raw) is dict and set(raw) == set(KEYS) and
            type(raw.get("schema_version")) is int and raw["schema_version"] == 1 and
            proof["registration_sha256"] == digest(registration_path),
            "registered PRIMARY metadata changed")
    for key in KEYS - {"schema_version"}:
        require(raw[key] == proof.get(key),
                "recorded PRIMARY registration differs: " + key)
    source_manifest = local(root, raw["source_manifest"])
    require(raw["origin_manifest_sha256"] == ORIGIN_MANIFEST_SHA256 and
            raw["source_manifest_sha256"] == ORIGIN_MANIFEST_SHA256 and
            digest(source_manifest) == ORIGIN_MANIFEST_SHA256,
            "frozen PRIMARY source manifest differs")
    groups = reference_plan["external_groups"]
    require(len(groups) == 1 and
            Path(groups[0]["storage_path"]).resolve() ==
            Path(raw["source_root"]).resolve(),
            "PRIMARY source differs from this capacity qualification")
    return dict(registration_path=str(registration_path.relative_to(root)),
                registration_sha256=proof["registration_sha256"],
                source_manifest_sha256=ORIGIN_MANIFEST_SHA256,
                source_content_proof_sha256=proof["source_content_proof_sha256"],
                proof_scope="recorded startup source proof plus original full post-run preservation guard; no new payload hash or release credit")


def native_geometry(report, domain):
    install = report["native_kv"]
    actual = install["actual_gpu_kv_bytes"]
    require(type(actual) is int and 0 < actual <= 2147483648 and
            type(install["groups"]) is int and install["groups"] == 1 and
            type(install["layers"]) is int and install["layers"] == 28 and
            type(install["num_blocks"]) is int and install["num_blocks"] > 0,
            "actual capacity model KV geometry differs")
    observed = []
    for name in ("native_kv", "cohort_probe_start", "probe"):
        point = report[name]
        controls = point["simple_native_control"]
        require(type(controls) is list and len(controls) == 1,
                "capacity model requires one actual native handler")
        for control in controls:
            require(control.get("owner_capture") is True and
                    control.get("native_shutdown_read") is False and
                    control.get("physical_drain_inferred") is False and
                    control.get("gpu_release_credit") is False,
                    "live geometry not captured by native owner")
            capacity = control["staging_capacity"]
            require(capacity.get("valid") is True and
                    capacity.get("observation_valid") is True and
                    capacity.get("error") is None and
                    capacity.get("physical_release_credit") is False and
                    type(capacity.get("slot_count")) is int and
                    capacity["slot_count"] == domain["slot_count"] and
                    type(capacity.get("io_size")) is int and capacity["io_size"] == QUANTUM and
                    type(control.get("native_io_size")) is int and
                    control["native_io_size"] == QUANTUM,
                    "actual capacity slot/I/O geometry unknown or mismatched")
        if name != "native_kv":
            require(type(point.get("kv_budget_bytes")) is int and
                    point["kv_budget_bytes"] == 2147483648 and
                    type(point.get("staging_budget_bytes")) is int and
                    point["staging_budget_bytes"] == domain["staging_bytes"],
                    "persisted native probe limits differ from capacity domain")
        observed.append(dict(point=name, slot_count=capacity["slot_count"],
                             io_size=capacity["io_size"], owner_capture=True))
    for name in ("probe", "final_probe"):
        point = report[name]
        require(type(point.get("staging_budget_bytes")) is int and
                point["staging_budget_bytes"] == domain["staging_bytes"] and
                type(point.get("kv_budget_bytes")) is int and
                point["kv_budget_bytes"] == 2147483648,
                "tail/final probe budget differs")
        handlers = point["handlers"]
        require(type(handlers) is list and len(handlers) == 1,
                "actual capacity staging handler missing")
        for handler in handlers:
            require(type(handler.get("staging_bytes")) is int and
                    0 < handler["staging_bytes"] <= domain["staging_bytes"] and
                    handler.get("pinned") is True and
                    handler.get("progress_enabled") is True and
                    handler.get("observation_failures") == 0,
                    "actual capacity staging exceeds budget or common observation failed")
    for control in report["final_probe"]["simple_native_control"]:
        admission = control["admission"]
        require(admission.get("count_valid") is True and
                admission.get("native_drain_unknown") is False and
                type(admission.get("accepted_parents")) is int and
                admission["accepted_parents"] == 0 and
                control.get("physical_drained") is True,
                "capacity whole-parent/native drain unknown")
    return dict(actual_gpu_kv_bytes=actual, kv_budget_bytes=2147483648,
                staging_budget_bytes=domain["staging_bytes"],
                slot_count=domain["slot_count"], io_quantum_bytes=QUANTUM,
                live_owner_observations=observed,
                physical_release_credit=False)


def audit(path, reference, manifest, *, capacity_domain_id, qualification,
          manifest_path, root=None):
    project = ROOT if root is None else Path(root).resolve()
    result_path = local(project, path)
    manifest_file = local(project, manifest_path)
    qualification_file = local(project, qualification)
    report = json.loads(result_path.read_text())
    require(manifest == json.loads(manifest_file.read_text()) and
            manifest == json.loads((result_path.parent / "manifest.json").read_text()) and
            report["manifest_sha256"] == digest(manifest_file),
            "capacity analysis did not use the exact frozen run manifest")
    domain = validate_declaration(manifest, capacity_domain_id)
    require(manifest["profile"] == "mixed_readwrite" and
            validate_declaration(report, capacity_domain_id) == domain,
            "capacity replay profile/domain differs")
    permit = verify_gate(qualification_file, capacity_domain_id=capacity_domain_id,
                         manifest_path=manifest_file, root=project)
    require(report["qualification"] == permit,
            "run did not use this independently qualified capacity permit")
    cost_plan = json.loads(local(project, permit["cost_plan"]).read_text())
    reference_plan = json.loads(local(project, cost_plan["reference_plan"]).read_text())
    candidate_path = local(project, cost_plan["candidate"])
    candidate = json.loads(candidate_path.read_text())
    require(report["curves_sha256"] == digest(candidate_path) == CANDIDATE_SHA256,
            "capacity cohort curve candidate changed")
    expected_engine = validate_manifest(manifest, candidate, reference_plan["gpu_uuid"],
                                        root=project)
    config_path = result_path.parent / "frozen-config.json"
    config = json.loads(config_path.read_text())
    require(config["manifest"] == manifest and
            config["gpu_uuid"] == reference_plan["gpu_uuid"] and
            config["pythonhashseed"] == "0" and
            report["engine_delta"] == {"max_num_seqs": 2} and
            report.get("common_parent_admission") is True and
            report.get("four_stage_hook_available") is True and
            report.get("native_cpu_probe_requested") is True and
            report.get("native_observation") == "on",
            "capacity actual configuration/common observation differs")
    options = report["simple_stage_options"]
    require(report["policy_mode"] in ("off", "shadow", "fixed", "pressure") and
            options["prefix_io_stage_policy"]["mode"] == report["policy_mode"],
            "capacity policy declaration differs")
    extra = expected_connector_variant(
        candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"],
        capacity_domain_id)
    extra.update(shared_storage_path=str(result_path.parent / "storage"),
                 load_planner="on", iodepth=8,
                 prefix_cache_break_even_path=str(candidate_path),
                 prefix_io_observation_mode="shadow",
                 prefix_io_observation_run_id=result_path.parent.parent.name,
                 prefix_io_observation_interval_ns=10_000_000)
    extra.update(options)
    expected_engine = dict(expected_engine, kv_transfer_config=dict(
        kv_connector="OffloadingConnector", kv_role="kv_both",
        kv_connector_extra_config=extra))
    require(config["engine"] == expected_engine,
            "actual engine/connector changed outside registered capacity/common policy")
    receipt = cohort_receipt(project, result_path, manifest_file, qualification_file,
                             capacity_domain_id, config)
    source = source_provenance(project, report, reference_plan, result_path)
    geometry = native_geometry(report, domain)
    # The frozen audit retains every original full-output, source-preservation,
    # physical-drain, accepted-accounting, CPU and pressure-known guard.
    value = baseline_audit(result_path, reference, manifest)
    value.update(capacity_domain_id=capacity_domain_id, capacity_domain=domain,
                 capacity_gate_audited=True,
                 capacity_qualification=dict(
                     path=str(qualification_file.relative_to(project)),
                     sha256=digest(qualification_file), status=permit["status"],
                     six_phase_guarded_gpu_receipts=permit["guarded_gpu_receipts"]),
                 capacity_cohort_guarded_gpu_receipt=receipt,
                 native_geometry=geometry, capacity_source_provenance=source,
                 manifest_path=str(manifest_file.relative_to(project)),
                 manifest_sha256=digest(manifest_file))
    return value


def main():
    ap = argparse.ArgumentParser()
    for name in ("result", "reference", "manifest", "qualification", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--capacity-domain", choices=DOMAIN_IDS, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    value = audit(args.result, args.reference, manifest,
                  capacity_domain_id=args.capacity_domain,
                  qualification=args.qualification, manifest_path=args.manifest)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in value.items()
                      if k not in ("comparisons", "final_control")}, indent=2))


if __name__ == "__main__":
    main()
