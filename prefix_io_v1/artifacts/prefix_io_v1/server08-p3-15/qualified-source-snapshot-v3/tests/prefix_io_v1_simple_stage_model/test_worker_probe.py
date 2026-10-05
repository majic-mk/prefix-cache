import sys
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import Mock
import pytest
import simple_stage_worker_probe as module

def rig(monkeypatch):
    stats={s:dict(inflight_ops=0,inflight_bytes=0) for s in ("ssd_read","ssd_write","h2d","d2h")}
    accounting=dict(valid=True,outstanding_records=0,stages=stats)
    aio=dict(closed=True,drained=True,outstanding=0,accepted=4,reaped=4)
    r=NS(_worker=NS(is_alive=lambda:False),_prefix_dispatch_controller=None,
        _prefix_stage_accounting=NS(snapshot=lambda:accounting),ring=NS(snapshot=lambda:aio),
        parent_admission_snapshot=lambda:dict(accepted_parents=0,peak_accepted_parents=2),
        _active=[],_inflight={},_pending_copies=[],_copy_ready=[],
        _ready_fds_load=[],_ready_fds_preload=[],actual_staging_bytes=4096,file_store=NS(io_size=4096),
        inspect_snapshot=Mock(return_value={"owner":"native"}))
    h=NS(coordinator=NS(reactor=r))
    state=ModuleType("vllm.distributed.kv_transfer.kv_transfer_state")
    state.get_kv_transfer_group=lambda:NS(connector_worker=NS(worker=NS(handlers=[h])))
    monkeypatch.setitem(sys.modules,state.__name__,state)
    monkeypatch.setattr(module,"original_probe",lambda worker,action,limits=None:{"original_action":action})
    return r,accounting,aio

def test_owner_metadata_request_and_real_native_quantum(monkeypatch):
    r,_,_=rig(monkeypatch)
    result=module.worker_probe(NS(),"start")
    assert result["original_action"]=="start"
    assert result["simple_native_control"]==[dict(owner="native",native_io_size=4096)]
    r.inspect_snapshot.assert_called_once_with(timeout=5)

def test_off_finish_requires_complete_native_drain_not_future_done(monkeypatch):
    r,_,_=rig(monkeypatch)
    result=module.worker_probe(NS(),"finish")
    assert result["simple_native_control"][0]["physical_drained"] is True
    assert result["simple_native_control"][0]["controller"] is None

@pytest.mark.parametrize("kind",["ready_load","ready_preload","copy","aio","stages","parents","worker","quantum"])
def test_incomplete_or_mismatched_physical_state_rejects(monkeypatch,kind):
    r,a,io=rig(monkeypatch)
    if kind=="ready_load":r._ready_fds_load.append(object())
    elif kind=="ready_preload":r._ready_fds_preload.append(object())
    elif kind=="copy":r._copy_ready.append(object())
    elif kind=="aio":io["outstanding"]=1
    elif kind=="stages":a["stages"]["h2d"]["inflight_bytes"]=1
    elif kind=="parents":r.parent_admission_snapshot=lambda:dict(accepted_parents=1)
    elif kind=="worker":r._worker.is_alive=lambda:True
    elif kind=="quantum":
        r._prefix_dispatch_controller=NS(config=NS(byte_quantum=8192),
            snapshot=lambda **kwargs:dict(pending_attempt=False))
    with pytest.raises(RuntimeError):module.worker_probe(NS(),"finish")
