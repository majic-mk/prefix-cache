"""CPU exact-numeric checks, separate from actual GPU evidence."""
import copy,importlib.util,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/prefix_io_v1/scripts"))
spec=importlib.util.spec_from_file_location("cached_checks",ROOT/"experiments/prefix_io_v1/scripts/analyze_cached_references.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row():
    return dict(prompt_token_ids=list(range(17)),output_token_ids=[0],num_cached_tokens=16,
                metrics=dict(is_corrupted=False),
                output_logprobs=[{str(k):dict(logprob=-float(k),rank=k+1) for k in range(5)}])
def test_exact_cached_match():
    r=m.compare_rows(row(),row(),16)
    assert r["output_equal"] and r["top5_exact_equal"]
def test_any_finite_difference_is_retained():
    a=row();b=row();b["output_logprobs"][0]["0"]["logprob"]=-1e-8
    r=m.compare_rows(a,b,16)
    assert r["output_equal"] and not r["top5_exact_equal"] and r["max_common_logprob_delta"]==1e-8
def test_tied_rank_is_not_float_difference():
    a=row();b=row()
    for r in (a,b):r["output_logprobs"][0]["1"]["logprob"]=0.
    b["output_logprobs"][0]["0"]["rank"]=2;b["output_logprobs"][0]["1"]["rank"]=1
    assert m.compare_rows(a,b,16)["top5_exact_equal"]
def test_equal_top5_does_not_hide_selected_token_mismatch():
    a=row();b=row();b["output_token_ids"]=[1]
    r=m.compare_rows(a,b,16)
    assert r["top5_exact_equal"] and not r["output_equal"]
@pytest.mark.parametrize("case",["nan","absent","shape","prompt","corrupted"])
def test_invalid_reference_rejected(case):
    a=row();b=row()
    if case=="nan":b["output_logprobs"][0]["0"]["logprob"]=float("nan")
    elif case=="absent":b.pop("output_logprobs")
    elif case=="shape":b["num_cached_tokens"]=0
    elif case=="prompt":b["prompt_token_ids"][0]=99
    elif case=="corrupted":b["metrics"]["is_corrupted"]=True
    with pytest.raises(ValueError):m.compare_rows(a,b,16)
