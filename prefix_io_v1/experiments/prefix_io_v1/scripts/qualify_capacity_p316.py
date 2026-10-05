"""Issue a capacity-domain permit only from completed real guarded GPU evidence."""
import argparse
import json
from pathlib import Path
from concurrent_capacity_contract_p316 import (
    DELTA, DOMAIN_IDS, CANDIDATE_SHA256, validate_declaration,
    connector_delta, validate as validate_manifest, digest)
from validate_heldout_costs_capacity_p316 import costs, _domain_plan

ROOT = Path(__file__).resolve().parents[3]
STATUS = "PASSED_CAPACITY_UNLOADED_POINT_GATE"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def _local(root, value):
    path = (Path(root) / value).resolve()
    require(path.is_relative_to(Path(root).resolve()), "qualification path outside project")
    return path

def _manifest_index(root, entries, domain, candidate, gpu_uuid):
    require(type(entries) is dict and set(entries) == {"mixed_readwrite"},
            "only the frozen mixed capacity manifest is permitted")
    item = entries["mixed_readwrite"]
    require(type(item) is dict and set(item) == {"path", "sha256"},
            "capacity manifest digest declaration required")
    path = _local(root, item["path"])
    require(digest(path) == item["sha256"], "capacity manifest changed")
    manifest = json.loads(path.read_text())
    require(manifest.get("capacity_domain_id") == domain["capacity_domain_id"],
            "manifest cannot cross capacity domains")
    validate_manifest(manifest, candidate, gpu_uuid, root=root)
    return path

def verify_gate(path, *, capacity_domain_id, manifest_path=None, root=None):
    project = ROOT if root is None else Path(root).resolve()
    permit = json.loads(Path(path).read_text())
    require(permit.get("status") == STATUS and permit.get("schema_version") == 2 and
            type(permit.get("schema_version")) is int,
            "old or unqualified permit cannot authorize a capacity domain")
    domain = validate_declaration(permit, capacity_domain_id)
    require(permit.get("engine_delta") == DELTA and
            type(permit["engine_delta"].get("max_num_seqs")) is int and
            permit.get("connector_delta") == connector_delta(capacity_domain_id) and
            type(permit.get("maximum_absolute_relative_median_error")) is float and
            permit["maximum_absolute_relative_median_error"] == .25 and
            permit.get("ordinary_quotas_installed") is False and
            permit.get("formal_goodput") is False,
            "capacity gate standard/scope changed")
    cp = _local(project, permit["cost_plan"])
    require(digest(cp) == permit["cost_plan_sha256"], "capacity cost plan changed")
    plan = json.loads(cp.read_text())
    require(_domain_plan(plan, capacity_domain_id, cost=True) == domain,
            "capacity plan/permit mismatch")
    result = costs(project, cp, capacity_domain_id=capacity_domain_id)
    recorded = _local(project, permit["cost_result"])
    require(result["status"] == "PASSED_HELDOUT_POINT_PREDICTION" and
            digest(recorded) == permit["cost_result_sha256"] and
            result == json.loads(recorded.read_text()), "real capacity point gate failed/changed")
    require(len(result["guarded_gpu_receipts"]) == 6 and
            len({r["label"] for r in result["guarded_gpu_receipts"]}) == 6 and
            permit.get("guarded_gpu_receipts") == result["guarded_gpu_receipts"],
            "six completed distinct guarded GPU phases required")
    require(permit["candidate_sha256"] == plan["candidate_sha256"] == CANDIDATE_SHA256 and
            digest(_local(project, plan["candidate"])) == CANDIDATE_SHA256,
            "frozen candidate changed")
    require(permit.get("capacity_source_hashes") == result["capacity_source_hashes"],
            "capacity source evidence changed")
    reference_plan = _local(project, plan["reference_plan"])
    reference_result = _local(project, plan["reference_result"])
    require(digest(reference_plan) == permit["reference_plan_sha256"] and
            digest(reference_result) == permit["reference_result_sha256"],
            "exact cached reference evidence changed")
    candidate = json.loads(_local(project, plan["candidate"]).read_text())
    rp = json.loads(reference_plan.read_text())
    approved_manifest = _manifest_index(project, permit["manifests"], domain, candidate, rp["gpu_uuid"])
    if manifest_path is not None:
        requested = (project / Path(manifest_path)).resolve()
        require(requested == approved_manifest and
                digest(requested) == permit["manifests"]["mixed_readwrite"]["sha256"],
                "requested replay manifest is not the qualified capacity manifest")
    return permit

def issue_permit(root, plan_path, result_path, manifest_index_path, output_path, *, capacity_domain_id):
    project = Path(root).resolve()
    cp = _local(project, str(Path(plan_path)))
    plan = json.loads(cp.read_text())
    domain = _domain_plan(plan, capacity_domain_id, cost=True)
    result = costs(project, cp, capacity_domain_id=capacity_domain_id)
    require(result["status"] == "PASSED_HELDOUT_POINT_PREDICTION",
            "capacity point failed; PASS permit will not be issued")
    recorded = _local(project, str(Path(result_path)))
    require(recorded.is_file() and json.loads(recorded.read_text()) == result,
            "completed unchanged cost result required before permit issuance")
    candidate = json.loads(_local(project, plan["candidate"]).read_text())
    reference_plan = _local(project, plan["reference_plan"])
    rp = json.loads(reference_plan.read_text())
    entries = json.loads(Path(manifest_index_path).read_text())
    _manifest_index(project, entries, domain, candidate, rp["gpu_uuid"])
    permit = dict(schema_version=2, status=STATUS,
        capacity_domain_id=capacity_domain_id, capacity_domain=domain,
        engine_delta=dict(DELTA), connector_delta=connector_delta(capacity_domain_id),
        runtime_scope="Exact finite capacity C2/d8 unloaded point gate; bounded mixed development screen only. Loaded prediction, unseen content, formal goodput and global-optimum claims remain unqualified.",
        maximum_absolute_relative_median_error=.25,
        cost_plan=str(cp.relative_to(project)), cost_plan_sha256=digest(cp),
        cost_result=str(recorded.relative_to(project)), cost_result_sha256=digest(recorded),
        candidate_sha256=CANDIDATE_SHA256, manifests=entries,
        reference_plan_sha256=digest(reference_plan),
        reference_result_sha256=digest(_local(project, plan["reference_result"])),
        capacity_source_hashes=result["capacity_source_hashes"],
        guarded_gpu_receipts=result["guarded_gpu_receipts"],
        ordinary_quotas_installed=False, formal_goodput=False)
    output = _local(project, str(Path(output_path)))
    with output.open("x") as stream:
        json.dump(permit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(verify_gate(output, capacity_domain_id=capacity_domain_id, root=project) == permit,
            "post-write capacity verification failed")
    return permit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-domain", choices=DOMAIN_IDS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    permit = issue_permit(ROOT, args.plan, args.result, args.manifests, args.output,
                          capacity_domain_id=args.capacity_domain)
    print(json.dumps(dict(status=permit["status"], capacity_domain=permit["capacity_domain"])))

if __name__ == "__main__":
    main()
