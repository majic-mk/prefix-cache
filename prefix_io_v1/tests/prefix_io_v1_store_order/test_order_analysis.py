import copy,json,sys
from pathlib import Path
import pytest
import analyze_store_order_model as analysis
import run_store_order_pilot as driver

def test_abba_effect_uses_all_repeats_and_correct_sign():
    metrics=("cohort_seconds","response_seconds","tail_drain_seconds","pending_flush_wait_seconds",
             "scheduled_ttft_mean_seconds","ttft_mean_seconds","itl_p95_seconds")
    rows=[dict(mode=m,**{k:v for k in metrics}) for m,v in
          [("off",10),("pressure",9),("pressure",20),("off",12)]]
    result=analysis.comparisons(rows)
    assert result["cohort_seconds"]["off_mean"]==11
    assert result["cohort_seconds"]["pressure_mean"]==14.5
    assert result["cohort_seconds"]["reduction_percent"]<0
    assert result["cohort_seconds"]["pair_reduction_percent"][0]>0
    assert result["cohort_seconds"]["pair_reduction_percent"][1]<0
    with pytest.raises(ValueError):analysis.comparisons(rows[:3])

def test_only_run_specific_config_fields_are_removed():
    raw=dict(engine=dict(kv_transfer_config=dict(kv_connector_extra_config=dict(
        shared_storage_path="/a",prefix_io_observation_run_id="run-a",iodepth=8))),model="m")
    other=copy.deepcopy(raw);extra=other["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    extra.update(shared_storage_path="/b",prefix_io_observation_run_id="run-b")
    assert analysis.normalized_config(raw)==analysis.normalized_config(other)
    extra["iodepth"]=4
    assert analysis.normalized_config(raw)!=analysis.normalized_config(other)
    assert raw["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["shared_storage_path"]=="/a"

@pytest.mark.parametrize("bad",["missing_passive","quota"])
def test_driver_rejects_unqualified_combinations_before_native(monkeypatch,tmp_path,bad):
    args=["pilot","--order-config",str(tmp_path/"config"),"--output",str(tmp_path/"out")]
    if bad=="quota":args+=["--flush-cause-probe","--store-readiness-probe","--start-budget-config=bad"]
    monkeypatch.setattr(sys,"argv",args)
    monkeypatch.setattr(driver.native,"main",lambda:pytest.fail("GPU driver must not launch"))
    with pytest.raises(ValueError):driver.main()

@pytest.mark.parametrize("native_failure",[False,True])
def test_driver_delegates_unchanged_and_records_actual_mode(monkeypatch,tmp_path,native_failure):
    config=tmp_path/"cfg.json";config.write_text("{}");out=tmp_path/"run"/"details"
    parsed=dict(mode="off",scope="test",qualification_sha256="receipt")
    monkeypatch.setattr(driver,"validate_config",lambda p:parsed)
    old=driver.passive.worker_probe;old_native=driver.native.worker_probe
    monkeypatch.setenv("PREFIX_IO_ORDER_MODEL_CONFIG","previous")
    args=["pilot","--order-config",str(config),"--output",str(out),
          "--flush-cause-probe","--store-readiness-probe","--model-dir","frozen-model"]
    monkeypatch.setattr(sys,"argv",args)
    def native():
        import os
        assert driver.passive.worker_probe is driver.ordered_probe
        assert os.environ["PREFIX_IO_ORDER_MODEL_CONFIG"]==str(config)
        assert sys.argv[1:]==["--output",str(out),*args[5:]]
        out.mkdir(parents=True)
        raw=dict(status="FAILED" if native_failure else "PASSED_NATIVE_C2_DEVELOPMENT_REPLAY",
             engine_shutdown="completed",cohort_probe_start=dict(store_order_mode="off"),
             final_probe=dict(store_order_final=[dict(mode="off",order=None,drained=True)]))
        (out/"result.json").write_text(json.dumps(raw))
        return int(native_failure)
    monkeypatch.setattr(driver.native,"main",native)
    assert driver.main()==int(native_failure)
    receipt=json.loads((out/"order-result.json").read_text())
    assert receipt["status"]==("FAILED" if native_failure else "PASS_REAL_MODEL_STORE_ORDER")
    assert receipt["actual_order_mode"]=="off"
    assert driver.passive.worker_probe is old and driver.native.worker_probe is old_native
    assert sys.argv is args
    import os
    assert os.environ["PREFIX_IO_ORDER_MODEL_CONFIG"]=="previous"
