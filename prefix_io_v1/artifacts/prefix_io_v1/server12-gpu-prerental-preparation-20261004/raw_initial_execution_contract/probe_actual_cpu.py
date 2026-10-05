"""Read existing actual child metadata and invoke original validation only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PREP=Path(__file__).resolve().parent.parent
source=PREP/"activation/native_conditional_cost.py"
assert hashlib.sha256(source.read_bytes()).hexdigest()=="675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
spec=importlib.util.spec_from_file_location("_CPU_read_actual_capture_only",source)
V=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=V
spec.loader.exec_module(V)
actual=Path(sys.argv[1]).resolve()
report=json.loads(actual.read_bytes())
row=report["windows"][0]
output=row["frontend"]["output"]
out=dict(origin="CPU_read_actual_metadata_no_new_GPU",actual_child_ref=dict(bytes=actual.stat().st_size,
    sha256=hashlib.sha256(actual.read_bytes()).hexdigest()),actual_child_status=report["status"],
    frame_count=len(row["capture"]["frames"]),witness_count=len(row["capture"]["event_witnesses"]),
    initial_context=row["capture"]["frames"][0]["prepared"]["context_length"],
    initial_prefill=row["capture"]["frames"][0]["prepared"]["prefill_tokens"],actual_GPU_runs=0,
    new_GPU_qualification=False,elapsed_values_exported=False)
for cached in (511,496):
    try:
        frames=V.validate_capture(row["capture"],run_id=row["request_id"],request_id=output["native_request_id"],
            output_ids=output["output_token_ids"],prompt_tokens=512,measured_offset=16,warmup_offsets=[1],cached_tokens=cached)
        out["capture_cached_"+str(cached)]=dict(status="PASS_EXISTING_ACTUAL_METADATA",frames=len(frames))
    except Exception as exc:
        out["capture_cached_"+str(cached)]=dict(status="REJECTED",error_type=type(exc).__name__,reason=str(exc))
try:
    source_sha=report["native_journal"]["source_sha256"]
    run_id=report["label"]
    drain=V.original_post_shutdown_drain(report["native_post_shutdown"],report["native_tail_assertions"],
        run_id=run_id,native_source_sha256=source_sha)
    V.validate_io(report["native_journal"],drain,capture=row["capture"],frames=frames,run_id=run_id,
        native_source_sha256=source_sha,arm="baseline" if row["condition"]=="A" else "action",measured_offset=16,physical_bytes=917504,
        operations=1,independent_payload=row["independent_payload"])
    out["actual_IO"]=dict(status="PASS_EXISTING_ACTUAL_METADATA")
except Exception as exc:
    out["actual_IO"]=dict(status="REJECTED",error_type=type(exc).__name__,reason=str(exc))
print(json.dumps(out,ensure_ascii=False,indent=2))
