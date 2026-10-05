import hashlib,json
from pathlib import Path
import pytest
from start_budget_model_contract import validate_config

def fixture(tmp_path,mode):
    files=["src/prefix_io_control/start_budget.py","src/prefix_io_control/stage_accounting.py",
        "src/prefix_io_control/start_options.py","third_party/work/py-kvcache-p3-quota-cpu/py_kvcache/reactor.py",
        "third_party/work/py-kvcache-p3-quota-cpu/py_kvcache/vllm.py"]
    lock={}
    for rel in files:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("CPU test fixture")
        lock[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    q=tmp_path/"artifacts/prefix_io_v1/server07-p3-09/gpu.json";q.parent.mkdir(parents=True)
    q.write_text(json.dumps(dict(status="PASS_REAL_GPU_START_BUDGET_AND_STAGE_ACCOUNTING",gpu_byte_exact=True,cases=[{}]*13)))
    options={"mode":"off"}
    if mode!="off":options=dict(schema_version=1,mode=mode,run_id="model-qualification",epoch_ns=10000000,
        starts_per_epoch=4,max_wait_ns=100000000,reserve_free_slots=1 if mode=="pressure" else 0)
    data=dict(schema_version=1,mode=mode,options=options,qualification=str(q),frozen_sources=lock,
        scope="P3_MODEL_START_QUALIFICATION_NOT_RESEARCH_EFFECT")
    p=tmp_path/"config.json";p.write_text(json.dumps(data));return p,data

@pytest.mark.parametrize("mode",["off","fixed","pressure"])
def test_frozen_model_config(mode,tmp_path):
    p,d=fixture(tmp_path,mode);assert validate_config(p,tmp_path)==d

@pytest.mark.parametrize("case",["parameter","source","qualification","coverage"])
def test_model_gate_rejects_unqualified_changes(case,tmp_path):
    p,d=fixture(tmp_path,"fixed")
    if case=="parameter":d["options"]["starts_per_epoch"]=8
    if case=="source":(tmp_path/next(iter(d["frozen_sources"]))).write_text("changed")
    if case=="qualification":Path(d["qualification"]).write_text('{"status":"FAILED","cases":[]}')
    if case=="coverage":d["frozen_sources"].pop(next(iter(d["frozen_sources"])))
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):validate_config(p,tmp_path)
