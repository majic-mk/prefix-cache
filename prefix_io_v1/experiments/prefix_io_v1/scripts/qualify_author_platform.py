"""Validate the author's platform metadata mapping under the approved UUID."""
import argparse
import json
import os
import traceback
p=argparse.ArgumentParser()
p.add_argument("--gpu-uuid",required=True)
a=p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES")==a.gpu_uuid
report={"status":"FAILED","expected_uuid":a.gpu_uuid,
        "scope":"platform device metadata only","model_loaded":False,
        "cache_qualified":False}
try:
    from vllm.platforms import current_platform
    assert current_platform.device_type=="cuda"
    actual=current_platform.get_device_uuid(0)
    if isinstance(actual,bytes):
        actual=actual.decode("ascii")
    assert actual==a.gpu_uuid,(actual,a.gpu_uuid)
    report.update(uuid=actual,platform_class=type(current_platform).__name__,
                  name=current_platform.get_device_name(0),
                  capability=list(current_platform.get_device_capability(0)),
                  memory_bytes=current_platform.get_device_total_memory(0),
                  status="PASSED")
except Exception as exc:
    report.update(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
print(json.dumps(report,indent=2))
raise SystemExit(0 if report["status"]=="PASSED" else 1)
