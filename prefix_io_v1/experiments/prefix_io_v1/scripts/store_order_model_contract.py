"""Frozen P3 mandatory-order screen; no general research-policy authorization."""
import hashlib,json
from pathlib import Path
from experiment_storage import ROOT

SCOPE="P3_MANDATORY_STORE_ORDER_SIMPLE_BASELINE"
NATIVE="third_party/work/py-kvcache-p3-order-cpu/py_kvcache"
SOURCES=("src/prefix_io_control/store_order.py","src/prefix_io_control/order_options.py",
         NATIVE+"/reactor.py",NATIVE+"/vllm.py")
EVIDENCE="artifacts/prefix_io_v1/server07-p3-12/primitive-result.json"

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_config(path,root=ROOT):
    data=json.loads(Path(path).read_text())
    require(set(data)=={"schema_version","mode","candidate_parents","scope","qualification",
                        "qualification_sha256","frozen_sources"},"unexpected config fields")
    require(type(data["schema_version"]) is int and data["schema_version"]==1,"schema mismatch")
    require(data["mode"] in ("off","pressure"),"unsupported order mode")
    require(type(data["candidate_parents"]) is int and data["candidate_parents"]==32,"candidate window changed")
    require(data["scope"]==SCOPE,"scope changed")
    q=root/EVIDENCE
    require(Path(data["qualification"]).resolve()==q.resolve(),"unexpected qualification path")
    require(sha(q)==data["qualification_sha256"],"qualification digest changed")
    report=json.loads(q.read_text())
    require(report["status"]=="PASS_REAL_GPU_STORE_ORDER" and report["gpu_byte_exact"] is True,
            "real GPU byte qualification missing")
    cases=report["cases"]
    require(len(cases)==6 and {(c["mode"],c["phase"]) for c in cases}==
            {(m,p) for m in ("off","pressure") for p in ("store","restore","shared")},"incomplete primitive cases")
    for c in cases:
        a=c["aio"];account=c["accounting"]
        require(a["closed"] and a["drained"] and a["fatal"] is None and
                a["accepted"]==a["completed"]==a["reaped"] and
                all(a[k]==0 for k in ("outstanding","pending","ready","unreaped")),"primitive AIO not drained")
        require(account["valid"] and account["outstanding_records"]==0 and
                all(s["failed_ops"]==0 and s["inflight_bytes"]==0 and
                    s["accepted_bytes"]==s["transferred_bytes"] for s in account["stages"].values()),
                "primitive accounting failed")
        require(c["actual_staging_bytes"]<=c["staging_budget_bytes"],"primitive staging exceeded")
        if c["mode"]=="pressure":require(c["order"] and not c["order"]["faulted"],"primitive order fault")
    stores={c["mode"]:c for c in cases if c["phase"]=="store"}
    require(stores["off"]["parent_completion_order"]==[1,2] and
            stores["pressure"]["parent_completion_order"]==[2,1] and
            stores["pressure"]["order"]["reordered_passes"]>0,"precedence not exercised")
    require(set(data["frozen_sources"])==set(SOURCES) and
            data["frozen_sources"]==report["source_sha256"],"qualification/source mismatch")
    for rel,digest in data["frozen_sources"].items():
        require(sha(root/rel)==digest,"qualified source changed: "+rel)
    require(Path(report["native_module"]).resolve()==(root/NATIVE/"reactor.py").resolve(),"wrong qualified runtime")
    return data
