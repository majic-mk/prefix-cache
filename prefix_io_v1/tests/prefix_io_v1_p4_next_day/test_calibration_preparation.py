import importlib.util,json,sys
from pathlib import Path
import pytest
from tests.prefix_io_v1_p4_next_day.test_next_day_plan import project,m as prep
SCRIPTS=Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("p4_startup_test",SCRIPTS/"prepare_p4_calibration_startup.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def calibration(project):
    path=project/m.ENGINE_SOURCE
    path.write_text("ENGINE="+repr(dict(dtype="bfloat16",quantization=None,kv_transfer_config=None,
        enforce_eager=True,enable_prefix_caching=True,max_model_len=4096,max_num_seqs=1,
        max_num_batched_tokens=4096,kv_cache_memory_bytes=0,disable_log_stats=True))+
        "\nSAMPLING="+repr(dict(temperature=0.0,max_tokens=1,min_tokens=1))+"\n")
    (project/m.CALIBRATION_SOURCE).write_text("pass")
    dest=project/m.MODEL_PLAN;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(model_id="Qwen/Qwen2.5-7B-Instruct",
       revision="16c174980d8a1492910551634b4969e69cdc2444")))
    return project

@pytest.mark.parametrize("stage",["ssd_read","ssd_write","h2d","d2h"])
@pytest.mark.parametrize("units",[1,2,4,8])
def test_real_startup_preserves_original_executor_and_single_action(calibration,stage,units):
    data=m.prepare(calibration,"prepared",stage,units)
    assert data["engine"]["kv_transfer_config"] is None and data["engine"]["dtype"]=="bfloat16"
    assert data["engine"]["enforce_eager"] and data["sampling"]["max_tokens"]==128
    assert data["engine"]["kv_cache_memory_bytes"]+data["action"]["source_owned_bytes"]==2*1024**3
    assert data["action"]["physical_bytes"]==units*917504
    assert data["action"]["physical_operations"] is None
    assert not data["GPU_collector_implemented"] and not data["GPU_collector_verified"]
    assert not data["collection_domain"]["scheduled_work_v1_is_active_decode"]
    assert data["collection_domain"]["step_timing_scope"] is None
    assert data["source_lock_pending"] and not data["production_qualified"]

def test_context_and_source_fields_cannot_be_invented(calibration):
    (calibration/m.MODEL_PLAN).write_text("{}")
    with pytest.raises((ValueError,KeyError)):m.prepare(calibration,"prepared","h2d",2)

def test_launch_check_is_cpu_only_and_ledger_unchanged(calibration):
    before=(calibration/prep.LEDGER).read_bytes()
    p=calibration/m.OUT/"startup-denied.json"
    assert m.main(["--project",str(calibration),"--name","prepared","--stage","h2d",
                   "--units","2","--output",str(p),"--check-launch"])==78
    data=json.loads(p.read_text());assert not data["gpu_initialized"] and data["new_gpu_runs"]==0
    assert (calibration/prep.LEDGER).read_bytes()==before
    with pytest.raises(ValueError):m.main(["--project",str(calibration),"--name","prepared","--stage","h2d",
                   "--units","2","--output",str(p)])
