"""Matched ordering screen around the unchanged P3 concurrent model driver."""
import argparse,json,os,sys,traceback
from pathlib import Path
from store_order_model_contract import validate_config,sha,require
import store_readiness_worker_probe as passive
from store_order_worker_probe import worker_probe as ordered_probe
import run_concurrent_pilot as native

def validate_result(raw,mode):
    require(raw["status"]=="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY","native replay failed")
    require(raw["engine_shutdown"]=="completed","engine shutdown failed")
    require(raw["cohort_probe_start"]["store_order_mode"]==mode,"actual mode mismatch")
    states=raw["final_probe"]["store_order_final"]
    require(bool(states),"missing native order states")
    for s in states:
        require(s["mode"]==mode and s["drained"],"undrained or wrong order state")
        if mode=="off":require(s["order"] is None,"off installed an order")
        else:require(s["order"] is not None and not s["order"]["faulted"],"order fault")
    return states

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--order-config",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    a,rest=ap.parse_known_args();config_path=a.order_config.resolve();out=a.output.resolve()
    require("--flush-cause-probe" in rest and "--store-readiness-probe" in rest,"matched passive observations required")
    require(not any(x.startswith("--start-budget-config") for x in rest),"quota/order combination not qualified")
    require(not out.exists(),"append-only result required")
    config=validate_config(config_path)
    old_env=os.environ.get("PREFIX_IO_ORDER_MODEL_CONFIG");old_probe=passive.worker_probe
    old_native=native.worker_probe;old_argv=sys.argv
    report=dict(status="FAILED",actual_order_mode=config["mode"],scope=config["scope"],
                config_path=str(config_path),config_sha256=sha(config_path),
                qualification_sha256=config["qualification_sha256"],
                ordinary_start_quota=False,full_stage_caps=False,research_joint_policy=False,
                formal_performance_claim=False,underlying_policy_label="shadow refers to passive observation only")
    code=1
    try:
        os.environ["PREFIX_IO_ORDER_MODEL_CONFIG"]=str(config_path)
        passive.worker_probe=ordered_probe
        sys.argv=[str(Path(native.__file__).resolve()),"--output",str(out),*rest]
        code=native.main()
        require(code==0,"native driver returned failure")
        raw=json.loads((out/"result.json").read_text())
        report["native_states"]=validate_result(raw,config["mode"])
        report["raw_result_sha256"]=sha(out/"result.json")
        report["status"]="PASS_REAL_MODEL_STORE_ORDER"
    except BaseException as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());code=1
    finally:
        passive.worker_probe=old_probe;native.worker_probe=old_native;sys.argv=old_argv
        if old_env is None:os.environ.pop("PREFIX_IO_ORDER_MODEL_CONFIG",None)
        else:os.environ["PREFIX_IO_ORDER_MODEL_CONFIG"]=old_env
        if out.exists():
            with (out/"order-config.json").open("x") as f:json.dump(config,f,indent=2)
            with (out/"order-result.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));return code
if __name__=="__main__":raise SystemExit(main())
