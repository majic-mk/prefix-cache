"""Offline strict raw-pair ingestion into the existing paired cost verifier.

No GPU is imported or launched. This script can assemble complete observations,
but it cannot attest that their origin was GPU execution or activate production.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from prepare_p4_gpu_next_day import require,safe_path,inspect_authorization,OUT

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--project",type=Path,default=Path("."))
    p.add_argument("--bundle",required=True)
    p.add_argument("--bundle-ref",required=True)
    p.add_argument("--plan",required=True)
    p.add_argument("--plan-ref",required=True)
    p.add_argument("--verifier-ref",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--check-launch",action="store_true")
    a=p.parse_args(argv);root=a.project.resolve()
    inspect_authorization(root)
    if a.check_launch:
        print(json.dumps(dict(status="BLOCKED_CPU_ONLY_OFFLINE_ADAPTER",new_gpu_runs=0,
                              gpu_initialized=False,budget_reserved_seconds=0)))
        return 78
    # Pure CPU modules are loaded only after authorization and launch denial.
    from prefix_io_control.p4_production_table_contract import TableContext,EvidenceRef
    from prefix_io_control.p4_paired_measurement_verifier import load_verification_plan
    from prefix_io_control.p4_raw_pair_recorder import prepare_raw_pair,assemble_raw_pair
    def pinned(path):
        ref=EvidenceRef.from_mapping(json.loads(safe_path(root,path).read_text()))
        ref.verify(root)
        return ref
    bref=pinned(a.bundle_ref);pref=pinned(a.plan_ref);vref=pinned(a.verifier_ref)
    require(bref.path==a.bundle and pref.path==a.plan,"independently pinned input paths")
    bundle=json.loads(bref.verify(root).read_text())
    required={"schema_version","scope","context","cell_id","load","stage","units","action_operations",
              "existing_io","baseline_runs","action_runs","baseline_windows","action_windows","origin"}
    require(type(bundle) is dict and set(bundle)==required and
            type(bundle["schema_version"]) is int and bundle["schema_version"]==1 and
            bundle["scope"]=="p4_explicit_pair_observations","complete explicit raw bundle")
    require(a.output_dir.startswith(OUT+"/"),"bounded append-new scalar preparation")
    safe_path(root,a.output_dir)
    prepared=prepare_raw_pair(TableContext.from_mapping(bundle["context"]),bundle["cell_id"],
        **{k:bundle[k] for k in required-{"schema_version","scope","context","cell_id"}})
    plan=load_verification_plan(root,a.plan,expected_plan_ref=pref)
    geometry=dict(cell_id=prepared.cell_id,load=bundle["load"],existing_io=bundle["existing_io"],
                  stage=bundle["stage"],physical_bytes=bundle["units"]*prepared.context.transfer_quantum_bytes)
    table,receipt=assemble_raw_pair(root,prepared,a.output_dir,cell_geometry=geometry,
        expected_plan=plan,expected_verifier_ref=vref)
    bref.verify(root);pref.verify(root);vref.verify(root)
    receipt.update(schema_version=1,bundle_ref=bref.__dict__,plan_ref=pref.__dict__,
        verifier_ref=vref.__dict__,cpu_mock_cells=table.cell_count,P5_SLO=None,P5_allowed=False,
        effect_verified=False,GPU_collector_implemented=False)
    dest=safe_path(root,a.output_dir+"/receipt.json")
    with dest.open("x") as h:json.dump(receipt,h,indent=2);h.write("\n")
    print(json.dumps({k:receipt[k] for k in ("status","origin","production_qualified","GPU_runs_performed")}))
    return 0

if __name__=="__main__":raise SystemExit(main())
