import copy,json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import store_order_model_contract as contract
import store_order_worker_probe as probe
from run_store_order_pilot import validate_result
from tests.prefix_io_v1_start_budget._fixture import rig,eventually

@pytest.fixture
def config_tree(tmp_path):
    for rel in contract.SOURCES:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((contract.ROOT/rel).read_bytes())
    report=json.loads((contract.ROOT/contract.EVIDENCE).read_text())
    report["native_module"]=str(tmp_path/contract.NATIVE/"reactor.py")
    q=tmp_path/contract.EVIDENCE;q.parent.mkdir(parents=True);q.write_text(json.dumps(report))
    data=dict(schema_version=1,mode="pressure",candidate_parents=32,scope=contract.SCOPE,
              qualification=str(q),qualification_sha256=contract.sha(q),frozen_sources=report["source_sha256"])
    return tmp_path,q,data

def checked(tree):
    root,q,data=tree;p=root/"config.json";p.write_text(json.dumps(data))
    return contract.validate_config(p,root=root)

@pytest.mark.parametrize("mode",["off","pressure"])
def test_qualified_contract(config_tree,mode):
    config_tree[2]["mode"]=mode;assert checked(config_tree)["mode"]==mode

@pytest.mark.parametrize("field,value",[
    ("schema_version",True),("schema_version",2),("mode","joint"),
    ("candidate_parents",True),("candidate_parents",64),("scope","production"),
    ("qualification","/tmp/not-authorized"),("qualification_sha256","bad"),("new_field",1),
    ("frozen_sources",{})])
def test_contract_rejects_changes(config_tree,field,value):
    config_tree[2][field]=value
    with pytest.raises(ValueError):checked(config_tree)

@pytest.mark.parametrize("fault",["source","status","exact","cases","completion","fault","drain","account","module"])
def test_bad_primitive_receipt(config_tree,fault):
    root,q,data=config_tree;report=json.loads(q.read_text())
    if fault=="source":(root/contract.SOURCES[0]).write_text("changed")
    elif fault=="status":report["status"]="CPU_PASS"
    elif fault=="exact":report["gpu_byte_exact"]=False
    elif fault=="cases":report["cases"].pop()
    elif fault=="completion":report["cases"][3]["parent_completion_order"]=[1,2]
    elif fault=="fault":report["cases"][3]["order"]["faulted"]=True
    elif fault=="drain":report["cases"][0]["aio"]["outstanding"]=1
    elif fault=="account":report["cases"][0]["accounting"]["outstanding_records"]=1
    elif fault=="module":report["native_module"]="/wrong/runtime.py"
    q.write_text(json.dumps(report));data["qualification_sha256"]=contract.sha(q)
    with pytest.raises(ValueError):checked(config_tree)

@pytest.mark.parametrize("mode",["off","pressure"])
def test_worker_model_boundary_uses_actual_native_order(monkeypatch,mode):
    with rig(monkeypatch,enabled=False,depth=1) as x:
        worker=NS();config=dict(mode=mode,candidate_parents=32)
        monkeypatch.setenv("PREFIX_IO_ORDER_MODEL_CONFIG","fixture")
        monkeypatch.setattr(probe,"validate_config",lambda p:config)
        monkeypatch.setattr(probe,"handlers",lambda:[NS(coordinator=x.coordinator)])
        def original(worker,action,limits=None):
            if action=="finish":x.stop()
            return {}
        monkeypatch.setattr(probe,"original",original)
        assert probe.worker_probe(worker,"install")["store_order_mode"]==mode
        assert probe.worker_probe(worker,"start")["store_order_mode"]==mode
        with pytest.raises(ValueError):probe.worker_probe(worker,"start")
        issued=[];native=x.r._schedule_store_copy
        def record(j,i,s):issued.append(j.job_id);return native(j,i,s)
        x.r._schedule_store_copy=record
        large=x.submit(files=4,job_id=7);small=x.submit(files=1,job_id=8)
        x.coordinator.request_mandatory([small]);x.start()
        assert small.result(timeout=2)==4096 and large.result(timeout=2)==16384
        result=probe.worker_probe(worker,"finish")["store_order_final"][0]
        assert result["drained"] and result["mode"]==mode
        assert issued[0]==(8 if mode=="pressure" else 7)
        assert (result["order"] is None)==(mode=="off")

@pytest.mark.parametrize("bad",["active","copies","io","quota","order","progress","runtime"])
def test_worker_refuses_unsafe_activation(monkeypatch,bad):
    with rig(monkeypatch,enabled=False) as x:
        r=x.r
        if bad=="active":r._active=[object()]
        elif bad=="copies":r._pending_copies=[object()]
        elif bad=="io":r._inflight={1:object()}
        elif bad=="quota":r._prefix_start_budget=object()
        elif bad=="order":r._prefix_store_order=object()
        elif bad=="progress":r._prefix_progress=None
        elif bad=="runtime":monkeypatch.setattr(probe.inspect,"getfile",lambda t:"/wrong/runtime.py")
        try:
            with pytest.raises(ValueError):probe.prepare([r],dict(mode="pressure",candidate_parents=32))
        finally:
            r._active=[];r._pending_copies=[];r._inflight={}
            r._prefix_start_budget=None;r._prefix_store_order=None

def test_worker_will_not_treat_native_fallback_as_success(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        entries=probe.prepare([x.r],dict(mode="pressure",candidate_parents=32))
        order=entries[0][1];x.r._prefix_store_order=order;x.start();x.stop()
        order.fail("injected")
        with pytest.raises(ValueError):probe.finish_orders(entries,"pressure")

def test_report_checks_actual_mode_shutdown_and_state():
    raw=dict(status="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY",engine_shutdown="completed",
        cohort_probe_start=dict(store_order_mode="off"),
        final_probe=dict(store_order_final=[dict(mode="off",drained=True,order=None)]))
    assert validate_result(raw,"off")
    for mutate in [
        lambda r:r.update(status="FAILED"),lambda r:r.update(engine_shutdown="failed"),
        lambda r:r["cohort_probe_start"].update(store_order_mode="pressure"),
        lambda r:r["final_probe"].update(store_order_final=[]),
        lambda r:r["final_probe"]["store_order_final"][0].update(order=dict(faulted=False)),
        lambda r:r["final_probe"]["store_order_final"][0].update(drained=False)]:
        other=copy.deepcopy(raw);mutate(other)
        with pytest.raises(ValueError):validate_result(other,"off")
