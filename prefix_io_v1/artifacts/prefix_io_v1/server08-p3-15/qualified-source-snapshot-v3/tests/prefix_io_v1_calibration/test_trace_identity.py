import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"))
from acquire_native_aio_costs import trace_summary

def trace(path,ids):
    events=[]
    for req in ids:
        events.append(dict(name="py_kvcache.transfer",args=dict(req_id=req,
            direction="storage_to_gpu",success=True,num_bytes=917504,
            src_cache=1,src_file=0,src_preload=0)))
        events.append(dict(name="py_kvcache.cuda_staging",args=dict(req_id=req,direction="storage_to_gpu")))
    path.write_text(json.dumps(dict(traceEvents=events)))
    return path

def test_internal_suffix_maps_exact_external_id(tmp_path):
    r=trace_summary(trace(tmp_path/"t.json",["2-0123abcd","20-0123abcd"]),"2")
    assert r["internal_request_id"]=="2-0123abcd"
    assert r["src_cache"]==r["h2d_events"]==1

def test_ambiguous_internal_ids_rejected(tmp_path):
    with pytest.raises(RuntimeError,match="ambiguous"):
        trace_summary(trace(tmp_path/"t.json",["2-0123abcd","2-1234abcd"]),"2")

def test_unrelated_request_never_counts(tmp_path):
    r=trace_summary(trace(tmp_path/"t.json",["20-0123abcd"]),"2")
    assert r["src_cache"]==r["h2d_events"]==0
    assert r["load_transfers"]==[]

def test_plain_external_id_supported(tmp_path):
    r=trace_summary(trace(tmp_path/"t.json",["2"]),"2")
    assert r["src_cache"]==r["h2d_events"]==1
