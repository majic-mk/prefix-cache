"""Replay old diagnostic values; not a live P4 shadow or a performance counterfactual."""
import argparse, hashlib, json
from pathlib import Path

def ref(p, root):
    raw=p.read_bytes()
    return {"path": str(p.relative_to(root)), "bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project",type=Path,default=Path("."))
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    root=a.project.resolve()
    paths=[root/"experiments/prefix_io_v1/runs"/label/"details/result.json" for label in
        ("server08-p3-16-mixed-01-off","server08-p3-16-mixed-06-off")]
    rows=[]
    for p in paths:
        j=json.loads(p.read_text());d=j["probe"]["flush_diagnostic"]
        retired=[r for r in d["records"] if r["kind"]=="native_parent_retired"]
        observations=[s for r in retired for s in r["source_blocks"]]
        rows.append(dict(input=ref(p,root),parent_retire_records=len(retired),
            sampled_source_observations=len(observations),
            positive_active_refs=sum(type(s["active_refs"]) is int and s["active_refs"]>0 for s in observations),
            same_generation=sum(s["same_generation"] is True for s in observations),
            truncated_parent_records=sum(r["source_truncated"] is True for r in retired),
            known_immediate_reusable_observations=sum(s["immediately_reusable_bytes"] is not None for s in observations),
            diagnostic_immediately_reusable_bytes=d["immediately_reusable_bytes"],
            production_GPU_release_credit=None))
    result=dict(schema_version=1,status="OFFLINE_REPLAY_OF_P3_OBSERVATIONS_NOT_P4_LIVE_SHADOW",
        scope="Old bounded owner diagnostics; parent completion alone does not establish GPU reuse.",
        input_lock=ref(root/"artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json",root),
        lock_validation="Full locked bytes separately checked at P4 entry and closeout",
        rows=rows, GPU_initialized=False,new_GPU_runs=0,
        new_native_snapshots=0,production_GPU_release_credit=None,
        counterfactual_runtime_gain=None,P4_efficacy_verified=False)
    out=a.output.resolve();out.relative_to(root/"artifacts/prefix_io_v1/server08-p4-01-cpu")
    assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","new_GPU_runs","new_native_snapshots","P4_efficacy_verified")}))
if __name__=="__main__":main()
