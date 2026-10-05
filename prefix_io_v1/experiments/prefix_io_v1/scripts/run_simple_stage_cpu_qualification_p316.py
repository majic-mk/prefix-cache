"""CPU qualification reuses the existing guarded two-process native test runner."""
import argparse, json, os, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
from experiment_storage import ROOT, preflight
from run_gpu_stage import cleanup_session
from run_shadow_cpu_qualification import plan as original_plan

HISTORICAL_FAULT_TESTS = [
 "tests/prefix_io_v1_progress/test_mandatory_bridge.py::test_failed_future_keeps_signal_until_other_io_drains",
 "tests/prefix_io_v1_start_budget/test_stage_accounting.py::test_rejected_write_is_not_counted_as_accepted",
] + [
 "tests/prefix_io_v1_dispatch_shadow/test_native_bridge.py::"+name+"["+value+"]"
 for name, values in (
  ("test_backend_exception_is_uncertain_and_native_error_unchanged", ("ssd_read","ssd_write")),
  ("test_cuda_backend_exception_is_uncertain_not_zero_execution", ("False","True")),
  ("test_api_acceptance_precedes_failed_end_event_and_is_never_refunded", ("False","True")))
 for value in values
]

def plan(label):
    result = original_plan(label)
    old = str(ROOT/"third_party/work/py-kvcache-p3-shadow-cpu")
    new = str(ROOT/"third_party/work/py-kvcache-p3-16-cpu")
    row = result["processes"][0]
    row["worktree"] = new
    row["environment_delta"]["PYTHONPATH"] = row["environment_delta"]["PYTHONPATH"].replace(old,new)
    row["command"] = [s.replace(old,new) for s in row["command"]]
    row["command"] += ["tests/prefix_io_v1_simple_stage_policy", "tests/prefix_io_v1_simple_stage_model",
                      "tests/prefix_io_v1_parent_admission", "tests/prefix_io_v1_copy_failure_drain",
                      "tests/prefix_io_v1_interference_calibration", "tests/prefix_io_v1_compact_staging_capacity",
                      "tests/prefix_io_v1_pinned_shutdown", "tests/prefix_io_v1_p3_evidence_audit"]
    row["command"] += ["--deselect="+name for name in HISTORICAL_FAULT_TESTS]
    import copy
    history = copy.deepcopy(result["processes"][1])
    history["name"] = "legacy-fault-history"
    history["worktree"] = old
    history["environment_delta"]["PYTHONPATH"] = history["environment_delta"]["PYTHONPATH"].replace(
        str(ROOT/"third_party/work/py-kvcache-p3-order-cpu"), old)
    history_out=Path(result["output"])
    history["environment_delta"]["AIO_CPU_GPU_GUARD_PATH"]=str(history_out/"legacy-fault-history-cuda-guard.json")
    history["command"]=[str(ROOT/".venv/bin/python"),str(ROOT/"experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py"),
        "-q","--import-mode=importlib"]+HISTORICAL_FAULT_TESTS+[
        "--basetemp",str(history_out/"legacy-fault-history-pytest-tmp"),
        "--junitxml",str(history_out/"legacy-fault-history-cpu.xml")]
    result["processes"].append(history)
    # Frozen P315 bare fixture lacks fresh scalar initialization. Its old
    # override-count assertion remains historical, with new known/unknown
    # capacity progress qualified separately in P316 tests.
    capacity_name = "tests/prefix_io_v1_parent_admission/test_stage_bridge.py::test_zero_stage_quota_preserves_mandatory_and_continuations[pressure]"
    row["command"].append("--deselect="+capacity_name)
    capacity_history = copy.deepcopy(history)
    capacity_history["name"] = "legacy-p315-capacity-fixture"
    capacity_history["worktree"] = str(ROOT/"third_party/work/py-kvcache-p3-15-cpu")
    capacity_history["environment_delta"]["PYTHONPATH"] = capacity_history["environment_delta"]["PYTHONPATH"].replace(
        old, capacity_history["worktree"])
    capacity_history["environment_delta"]["AIO_CPU_GPU_GUARD_PATH"] = str(history_out/"legacy-p315-capacity-fixture-cuda-guard.json")
    capacity_history["command"] = [str(ROOT/".venv/bin/python"),str(ROOT/"experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py"),
        "-q","--import-mode=importlib",capacity_name,"--basetemp",str(history_out/"legacy-p315-capacity-fixture-pytest-tmp"),
        "--junitxml",str(history_out/"legacy-p315-capacity-fixture-cpu.xml")]
    result["processes"].append(capacity_history)
    result["historical_capacity_fixture"] = capacity_name

    result["historical_fault_tests"]=HISTORICAL_FAULT_TESTS
    result["historical_fault_scope"]="Old immediate-error/no-stream-proof behavior reproduced only in its frozen P313 worktree; NOT new lifecycle qualification. New tests cover full-parent drain and unknown preservation."
    result["scope"] = "P3 common intake and four-stage simple policy CPU integration; no GPU initialization"
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--label",required=True); ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args(); p=plan(a.label)
    if a.dry_run: print(json.dumps(p,indent=2)); return 0
    budget=json.loads((ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())
    if budget["active_reservation"] is not None: raise RuntimeError("CPU tests cannot overlap reserved GPU measurements")
    out=Path(p["output"]); out.mkdir(parents=True,exist_ok=False)
    with (out/"plan.json").open("x") as f: json.dump(p,f,indent=2)
    results=[]
    for row in p["processes"]:
        proc=None; code=1; error=None; start=time.monotonic(); cleanup=None
        env=os.environ.copy(); env.update(row["environment_delta"])
        try:
            with (out/(row["name"]+"-process.log")).open("xb") as log:
                proc=subprocess.Popen(row["command"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=proc.wait(timeout=row["maximum_seconds"])
        except BaseException as exc: error=repr(exc)
        finally:
            if proc is not None:
                cleanup=cleanup_session(proc)
                if not cleanup["session_drained"] or cleanup.get("session_members_before_cleanup"): code=1
            results.append(dict(name=row["name"],exit=code,error=error,cleanup=cleanup,seconds=time.monotonic()-start))
    cases={}
    for path in out.glob("*-cpu.xml"):
        tree=ET.parse(path)
        for c in tree.findall(".//testcase"):
            identity=(c.attrib.get("classname"),c.attrib.get("name"))
            if identity in cases: raise RuntimeError("duplicate CPU test identity")
            outcome="failed" if c.find("failure") is not None or c.find("error") is not None else "skipped" if c.find("skipped") is not None else "passed"
            cases[identity]=outcome
    counts={kind:sum(v==kind for v in cases.values()) for kind in ("passed","skipped","failed")}
    guard_proofs=[]
    for row in p["processes"]:
        path=Path(row["environment_delta"]["AIO_CPU_GPU_GUARD_PATH"])
        guard=json.loads(path.read_text()) if path.exists() else None
        guard_proofs.append(dict(process=row["name"],evidence=str(path),proof=guard,
            verified=guard is not None and guard.get("cuda_initialized") is False and guard.get("gpu_workloads_run")==0))
    all_guards=all(x["verified"] for x in guard_proofs)
    complete_matrix=counts["passed"]>=964
    result=dict(status="PASS_CPU_SIMPLE_STAGE" if all(x["exit"]==0 for x in results) and counts["failed"]==0 and all_guards and complete_matrix else "FAIL_CPU_SIMPLE_STAGE",
                processes=results,unique_counts=counts,cuda_guard_proofs=guard_proofs,complete_historical_matrix=complete_matrix,
                cuda_initialized=False if all_guards else None,gpu_runs=0,storage_after=preflight(out,0))
    with (out/"result.json").open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2)); return int(result["status"]!="PASS_CPU_SIMPLE_STAGE")
if __name__=="__main__":raise SystemExit(main())
