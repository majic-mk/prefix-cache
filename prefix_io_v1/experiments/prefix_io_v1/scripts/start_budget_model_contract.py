"""Frozen small P3 model qualification candidates, not full research policy modes."""
import hashlib,json
from pathlib import Path
from experiment_storage import ROOT
from prefix_io_control.start_options import parse_start_options,KEY

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_config(path,root=ROOT):
    data=json.loads(Path(path).read_text())
    require(set(data)=={"schema_version","mode","options","qualification","frozen_sources","scope"},"unknown model config fields")
    require(type(data["schema_version"]) is int and data["schema_version"]==1,"schema mismatch")
    mode=data["mode"];require(mode in ("off","fixed","pressure"),"unsupported candidate")
    expected={"mode":"off"}
    if mode!="off":
        expected=dict(schema_version=1,mode=mode,run_id="model-qualification",epoch_ns=10_000_000,
            starts_per_epoch=4,max_wait_ns=100_000_000,reserve_free_slots=1 if mode=="pressure" else 0)
    require(data["options"]==expected,"candidate parameters changed")
    parse_start_options({KEY:data["options"]})
    require(data["scope"]=="P3_MODEL_START_QUALIFICATION_NOT_RESEARCH_EFFECT","scope changed")
    native=root/"third_party/work/py-kvcache-p3-quota-cpu/py_kvcache"
    sources=["src/prefix_io_control/start_budget.py","src/prefix_io_control/stage_accounting.py",
        "src/prefix_io_control/start_options.py",
        "third_party/work/py-kvcache-p3-quota-cpu/py_kvcache/reactor.py",
        "third_party/work/py-kvcache-p3-quota-cpu/py_kvcache/vllm.py"]
    require(set(data["frozen_sources"])==set(sources),"frozen source coverage incomplete")
    for rel,digest in data["frozen_sources"].items():
        require(hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,"qualified source changed")
    q=Path(data["qualification"])
    require(q.resolve().is_relative_to(root/"artifacts/prefix_io_v1/server07-p3-09"),"qualification outside project evidence")
    report=json.loads(q.read_text())
    require(report["status"]=="PASS_REAL_GPU_START_BUDGET_AND_STAGE_ACCOUNTING" and
        report["gpu_byte_exact"] and len(report["cases"])==13,"GPU primitive qualification missing")
    return data
