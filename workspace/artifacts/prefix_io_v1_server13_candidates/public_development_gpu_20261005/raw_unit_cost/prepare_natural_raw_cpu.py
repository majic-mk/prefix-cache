"""Append raw-only A/B metadata and inherited/targeted source proof; no GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(module))
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--ssd-manifest", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--storage-template", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--job-prefix", default="server13-natural-unit")
    args = parser.parse_args(argv)
    root = args.project.resolve()
    M = source(Path(__file__).with_name("natural_unit_raw_runner.py"), "_raw_CPU_factory_"+str(time.monotonic_ns()))
    R = M.driver(root)
    a = "artifacts/prefix_io_v1/server13-public-development-gpu-20261005"
    parent_ref = R.ref(root, a+"/ACTUAL_UOFF03_COLLECTION_CLOSED_02.json")
    parent = R.read(R.check_ref(root, parent_ref))
    config = R.read(R.check_ref(root, parent["actual_run_config_ref"]))
    refs = R.source_rows(root, config["source_lock_ref"], full=False)
    current = dict(refs)
    paths = [M.P+"/raw_unit_cost/natural_unit_raw_runner.py", M.P+"/raw_unit_cost/raw_sdk_binding.py",
        M.P+"/raw_unit_cost/prepare_natural_raw_cpu.py", args.ssd_manifest, args.geometry, parent_ref["path"],
        parent["completed_guard_ref"]["path"], parent["actual_native_result_ref"]["path"],
        parent["actual_run_config_ref"]["path"], parent["closure_source_ref"]["path"]]
    paths += [row["path"] for row in parent["actual_raw_source_refs"]]
    for path in paths:
        row = R.ref(root, path)
        M.require(path not in current or current[path] == row, "no change to inherited source bytes: "+path)
        current[path] = row
    prefix = args.output_prefix.rstrip("/")
    lock_path = prefix+"/RAW_UNIT_SOURCE_LOCK_01.json"
    R.new_json(R.safe(root, lock_path), dict(schema="current_raw_unit_inherited_source_lock_v1",
        files=[current[path] for path in sorted(current)], actual_GPU_operations=0,
        whole_model_source_rehash_performed=False, raw_only=True))
    lock_ref = R.ref(root, lock_path)
    bridge_path = M.P+"/u_collection_bridge/u_collection_bridge_v3.py"
    B = source(R.check_ref(root, current[bridge_path]), "_raw_CPU_ancestry_"+str(time.monotonic_ns()))
    ancestor = R.read(R.check_ref(root, B.ANCESTOR_LOCK))
    old_paths = {row["path"] for row in ancestor["files"]}
    proof = dict(schema="current_gpu_raw_unit_inherited_source_proof_v1", status="PASS_INHERITED_V14_PLUS_TARGETED_CPU_BYTES",
        ancestor_source_lock_ref=B.ANCESTOR_LOCK, ancestor_source_proof_ref=B.ANCESTOR_PROOF,
        source_lock_ref=lock_ref, source_count=len(current), ancestor_source_count=len(old_paths),
        targeted_source_refs=[current[path] for path in sorted(set(current)-old_paths)],
        required_source_refs=[current[path] for path in R.REQUIRED], actual_gpu_runs=0,
        full_source_verified=False, current_host_whole_source_hash_performed=False)
    proof_path = prefix+"/RAW_UNIT_SOURCE_PROOF_01.json"
    R.new_json(R.safe(root, proof_path), proof)
    proof_ref = R.ref(root, proof_path)
    created = []
    for index, arm in enumerate(("A", "B")):
        job_id = args.job_prefix+"-"+arm+"01"
        plan = M.make_cpu_plan(root, current, source_lock_ref=lock_ref, pair_ref=config["pair_config_ref"],
            input_manifest_ref=current[args.ssd_manifest], geometry_ref=current[args.geometry],
            runner_ref=current[M.P+"/raw_unit_cost/natural_unit_raw_runner.py"], parent_ref=parent_ref,
            guard_ref=parent["completed_guard_ref"], gpu_uuid=parent["gpu_uuid"], job_id=job_id)
        plan_path = prefix+"/NATURAL_RAW_UNIT_PLAN_"+arm+"01.json"
        R.new_json(R.safe(root, plan_path), plan)
        run = dict(schema="current_gpu_natural_calibration_raw_unit_config_v1", job_id=job_id,
            gpu_uuid=parent["gpu_uuid"], seconds_limit=300, window_index=index,
            source_lock_ref=lock_ref, source_proof_ref=proof_ref, plan_ref=R.ref(root, plan_path),
            permissions_ref=config["permissions_ref"], actual_u_collection_parent_ref=parent_ref,
            actual_u_collection_guard_ref=parent["completed_guard_ref"],
            output_relative="experiments/prefix_io_v1/runs/"+job_id+"/details",
            storage_template_relative=args.storage_template, input_manifest_ref=current[args.ssd_manifest],
            runner_ref=current[M.P+"/raw_unit_cost/natural_unit_raw_runner.py"])
        run_path = prefix+"/NATURAL_RAW_UNIT_RUN_"+arm+"01.json"
        R.new_json(R.safe(root, run_path), run)
        # Actual static/source/prior-U/model-stat replay, no namespace creation.
        M.verify_configuration(root, R.safe(root, run_path))
        created.append(dict(condition=arm, configuration_ref=R.ref(root,run_path), plan_ref=run["plan_ref"]))
    print(json.dumps(dict(status="PASS_ACTUAL_CPU_RAW_AB_SOURCE_INPUT_PRIOR_U_PREPARATION", created=created,
        source_lock_ref=lock_ref, source_proof_ref=proof_ref, actual_GPU_operations=0,
        table_issued=False, ordinary_I_authorized=False), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
