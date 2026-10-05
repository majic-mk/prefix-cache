"""Enable the qualified P3 ordering only at a fully drained cohort boundary."""
import inspect,os
from pathlib import Path
from store_readiness_worker_probe import worker_probe as original
from store_order_model_contract import ROOT,NATIVE,validate_config,require
from prefix_io_control.store_order import MandatoryStoreOrder

def handlers():
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    return get_kv_transfer_group().connector_worker.worker.handlers

def drained(r):
    return not r._active and not r._pending_copies and not r._inflight

def prepare(reactors,config):
    require(bool(reactors),"no native reactors")
    for r in reactors:
        require(drained(r),"cohort ordering requires drained native work")
        require(r.progress_bridge_enabled,"mandatory identity bridge missing")
        require(getattr(r,"_prefix_start_budget",None) is None,"ordinary quota not qualified with order")
        require(getattr(r,"_prefix_store_order",None) is None,"order already installed")
        require(Path(inspect.getfile(type(r))).resolve()==(ROOT/NATIVE/"reactor.py").resolve(),"wrong native runtime")
    orders=[]
    for r in reactors:
        order=None
        if config["mode"]=="pressure":
            order=MandatoryStoreOrder(candidate_parents=config["candidate_parents"]);order.bind()
        orders.append((r,order))
    return orders

def finish_orders(entries,mode):
    result=[]
    for r,order in entries:
        require(drained(r) and not r._worker.is_alive(),"order snapshot requires native shutdown")
        require(r._prefix_store_order is order,"unexpected order replacement")
        snapshot=order.snapshot() if order else None
        require(snapshot is None or not snapshot["faulted"],"order fell back after fault")
        require((mode=="off")== (snapshot is None),"mode/state mismatch")
        result.append(dict(mode=mode,order=snapshot,drained=True,
                           reactor_source=str(Path(inspect.getfile(type(r))).resolve())))
    return result

def worker_probe(worker,action,limits=None):
    if action=="install":
        config=validate_config(os.environ["PREFIX_IO_ORDER_MODEL_CONFIG"])
        result=original(worker,action,limits)
        worker._store_order_config=config
        worker._store_order_entries=None
        result["store_order_mode"]=config["mode"]
        return result
    if action=="start":
        require(worker._store_order_entries is None,"cohort cannot start twice")
        entries=prepare([h.coordinator.reactor for h in handlers()],worker._store_order_config)
        result=original(worker,action,limits)
        # No cohort request exists yet. Assign once while drained; native owner
        # binds its thread identity on its first actual ordering call.
        for r,order in entries:r._prefix_store_order=order
        worker._store_order_entries=entries
        result["store_order_mode"]=worker._store_order_config["mode"]
        return result
    result=original(worker,action,limits)
    if action=="finish":
        require(worker._store_order_entries is not None,"cohort ordering never started")
        result["store_order_final"]=finish_orders(worker._store_order_entries,worker._store_order_config["mode"])
    return result
