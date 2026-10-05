"""P314 exact full-output comparison; empty ambiguity-list schema is explicit."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[3]
O=R/"artifacts/prefix_io_v1/server08-p3-14"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 model=R/"experiments/prefix_io_v1/runs/server08-p3-14-model-off-01/details/result.json"
 reference=R/"experiments/prefix_io_v1/runs/server08-p3-13-native-fullref-01/details/result.json"
 m=json.loads(model.read_text());ref=json.loads(reference.read_text())
 indexed={(r["family"],r["kind"]):r for r in ref["rows"]}
 manifest=json.loads((O/"d8/mixed_readwrite-manifest.json").read_text())
 families={f["name"]:f["tokens"] for f in manifest["families"]};rows=[]
 for row in m["rows"]:
  kind="cold" if row["num_cached_tokens"]==0 else "gpu_hot";rr=indexed[(row["family"],kind)]
  assert rr["prompt_token_ids"]==families[row["family"]]
  assert type(row["ambiguous_events"]) is list
  rows.append(dict(request_id=row["request_id"],family=row["family"],reference_kind=kind,
   num_cached_tokens=row["num_cached_tokens"],output_tokens=len(row["output_tokens"]),
   output_exact=row["output_tokens"]==rr["output_tokens"],per_token_complete=row["per_token_complete"],
   token_timestamps=len(row["engine_token_timestamps"]),ambiguous_event_count=len(row["ambiguous_events"])))
 good=(m["status"]=="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY" and len(rows)==10
  and {r["request_id"] for r in rows}=={"c2-"+str(i) for i in range(10)}
  and all(r["output_exact"] and r["per_token_complete"] and r["output_tokens"]==r["token_timestamps"]==128
   and r["ambiguous_event_count"]==0 for r in rows))
 proof=dict(status="PASS_FULL_MODEL_MIGRATION_REFERENCE" if good else "FAILED_FULL_MODEL_MIGRATION_REFERENCE",
  comparison_scope="Previously seen five families; single native/off replay; no method efficacy claim",
  source_result=str(model),source_result_sha256=sha(model),reference=str(reference),reference_sha256=sha(reference),
  requests=len(rows),output_tokens=sum(r["output_tokens"] for r in rows),comparisons=rows,
  ordinary_quota_installed=m["ordinary_quota_installed"],cohort_read_bytes=m["cohort_read_bytes"],
  cohort_write_bytes=m["cohort_write_bytes"],source_preservation=m["source_preservation"],
  native_engine_shutdown=m["engine_shutdown"],formal_performance_claim=False,
  retained_initial_analysis=str(O/"model-off-comparison.json"),
  initial_analysis_error="Compared the empty ambiguous_events list with integer 0. v2 checks its list schema/count; raw GPU result unchanged.",
  gpu_rerun_performed=False)
 with (O/"model-off-comparison-v2.json").open("x") as f:json.dump(proof,f,indent=2)
 print(json.dumps(proof,indent=2));assert good
if __name__=="__main__":main()
