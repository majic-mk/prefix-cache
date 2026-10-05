"""Offline strict raw-pair ingestion into the existing paired cost verifier.

No GPU is imported or launched. This script can assemble complete observations,
but it cannot attest that their origin was GPU execution or activate production.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from prepare_p4_gpu_next_day import require,safe_path,inspect_authorization,OUT


def _prepare_v2_bundle(root, bundle, plan):
    """Bind four complete V2 roles; never infer missing GPU capture fields."""
    from hashlib import sha256
    from prefix_io_control.p4_production_table_contract import TableContext, EvidenceRef
    from prefix_io_control.p4_raw_pair_recorder import PreparedRawPair, ROLES, MAX_JSON_BYTES
    required={"schema_version","scope","context","cell_id","origin","cell_geometry","raw_refs"}
    require(type(bundle) is dict and set(bundle)==required and
            type(bundle["schema_version"]) is int and bundle["schema_version"]==2 and
            bundle["scope"]=="p4_explicit_pair_observations", "exact V2 byte-ref bundle")
    context=TableContext.from_mapping(bundle["context"])
    require(plan.schema_version==2 and context==plan.context and
            type(bundle["cell_id"]) is str and
            {name for name,_ in plan.action_operations}=={bundle["cell_id"]},
            "V2 bundle must match exact one-cell frozen V2 plan")
    require(type(bundle["origin"]) is str and bundle["origin"] in ("cpu_fixture","native_gpu_recording"),
            "explicit V2 observation origin required")
    geometry=bundle["cell_geometry"]
    require(type(geometry) is dict and
            set(geometry)=={"cell_id","load","existing_io","stage","physical_bytes"} and
            geometry["cell_id"]==bundle["cell_id"], "exact V2 cell geometry")
    refs=bundle["raw_refs"]
    require(type(refs) is dict and set(refs)==set(ROLES), "exact four V2 raw byte refs")
    role_json=[];source_refs=[];counts={}
    for role in ROLES:
        ref=EvidenceRef.from_mapping(refs[role]);path=ref.verify(root)
        raw=path.read_bytes()
        require(0<len(raw)<=MAX_JSON_BYTES and len(raw)==ref.bytes and sha256(raw).hexdigest()==ref.sha256,
                "V2 raw bytes changed during ingestion")
        obj=json.loads(raw)
        arm="baseline" if role.startswith("baseline") else "action"
        payload="runs" if role.endswith("wrapper") else "windows"
        scope="paired_measurement_wrapper" if payload=="runs" else "paired_window_observations"
        require(type(obj) is dict and
                set(obj)=={"schema_version","scope","origin","context","arm","cell_id",payload} and
                type(obj["schema_version"]) is int and obj["schema_version"]==2 and
                obj["scope"]==scope and obj["origin"]==bundle["origin"] and
                obj["cell_id"]==bundle["cell_id"] and obj["arm"]==arm and
                TableContext.from_mapping(obj["context"])==context,
                "all raw role V2 versions/context/cell/origin must match bundle")
        require(type(obj[payload]) is list and obj[payload], "complete nonempty V2 raw roles")
        if payload=="runs":
            counts[arm]=len(obj["runs"])
            for run in obj["runs"]:
                require(type(run) is dict and "complete_trace_ref" in run and
                        type(run.get("pair_id")) is str, "actual complete V2 step/output trace required")
                tref=EvidenceRef.from_mapping(run["complete_trace_ref"])
                traw=tref.verify(root).read_bytes()
                require(0<len(traw)<=MAX_JSON_BYTES and len(traw)==tref.bytes and
                        sha256(traw).hexdigest()==tref.sha256, "V2 complete trace bytes changed")
                trace=json.loads(traw)
                require(type(trace) is dict and type(trace.get("schema_version")) is int and
                        trace["schema_version"]==2 and trace.get("scope")=="p4_complete_step_output_trace" and
                        trace.get("origin")==bundle["origin"] and trace.get("arm")==arm and
                        trace.get("cell_id")==bundle["cell_id"] and trace.get("pair_id")==run["pair_id"] and
                        TableContext.from_mapping(trace.get("context"))==context,
                        "complete trace version/context/cell/origin/arm must match")
                source_refs.append(tref)
        else:
            counts[arm+"_measured"]=sum(row.get("phase")=="measured" for row in obj["windows"]
                                        if type(row) is dict)
        ref.verify(root);source_refs.append(ref);role_json.append((role,raw))
    timing_ref=dict(plan.timing_contract)["reference_source_ref"]
    timing_ref.verify(root);source_refs.append(timing_ref)
    require(counts["baseline"]==counts["action"] and
            counts["baseline_measured"]==counts["action_measured"],
            "V2 arms must have complete paired counts")
    prepared=PreparedRawPair(context,bundle["cell_id"],bundle["origin"],tuple(role_json),
                             counts["baseline"],counts["baseline_measured"])
    return prepared,geometry,tuple(source_refs)


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
    require(type(bundle) is dict and type(bundle.get("schema_version")) is int and
            bundle["schema_version"] in (1,2), "explicit bounded raw bundle version")
    version=bundle["schema_version"]
    require(a.output_dir.startswith(OUT+"/"),"bounded append-new scalar preparation")
    safe_path(root,a.output_dir)
    plan=load_verification_plan(root,a.plan,expected_plan_ref=pref)
    require(plan.schema_version==version,"bundle/plan measurement version mismatch")
    source_refs=()
    if version==1:
        required={"schema_version","scope","context","cell_id","load","stage","units","action_operations",
                  "existing_io","baseline_runs","action_runs","baseline_windows","action_windows","origin"}
        require(set(bundle)==required and bundle["scope"]=="p4_explicit_pair_observations",
                "complete explicit raw bundle")
        prepared=prepare_raw_pair(TableContext.from_mapping(bundle["context"]),bundle["cell_id"],
            **{k:bundle[k] for k in required-{"schema_version","scope","context","cell_id"}})
        geometry=dict(cell_id=prepared.cell_id,load=bundle["load"],existing_io=bundle["existing_io"],
                      stage=bundle["stage"],physical_bytes=bundle["units"]*prepared.context.transfer_quantum_bytes)
    else:
        prepared,geometry,source_refs=_prepare_v2_bundle(root,bundle,plan)
    table,receipt=assemble_raw_pair(root,prepared,a.output_dir,cell_geometry=geometry,
        expected_plan=plan,expected_verifier_ref=vref)
    bref.verify(root);pref.verify(root);vref.verify(root)
    for ref in source_refs:ref.verify(root)
    receipt.update(schema_version=version,bundle_ref=bref.__dict__,plan_ref=pref.__dict__,
        verifier_ref=vref.__dict__,cpu_mock_cells=table.cell_count,P5_SLO=None,P5_allowed=False,
        effect_verified=False,GPU_collector_implemented=False,
        measurement_schema_version=version,
        ingested_source_refs=[ref.__dict__ for ref in source_refs])
    dest=safe_path(root,a.output_dir+"/receipt.json")
    with dest.open("x") as h:json.dump(receipt,h,indent=2);h.write("\n")
    print(json.dumps({k:receipt[k] for k in ("status","origin","production_qualified","GPU_runs_performed")}))
    return 0

if __name__=="__main__":raise SystemExit(main())

