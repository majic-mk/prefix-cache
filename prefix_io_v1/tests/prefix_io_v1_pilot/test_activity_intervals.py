import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"))
from analyze_long_replay import merge,overlap
def test_intersection_does_not_double_count_concurrent_kernels():
    assert overlap([(1,6),(3,8)],[(2,4),(3,5),(7,9)])==4
def test_adjacent_intervals_merge():
    assert merge([(5,7),(1,3),(3,5),(1,1)])==[(1,7)]
def test_disjoint_and_empty():
    assert overlap([],[(1,4)])==0
    assert overlap([(0,1)],[(1,2)])==0
@pytest.mark.parametrize("item",[[(2,1)],[(0,float("nan"))]])
def test_invalid_interval_rejected(item):
    with pytest.raises(ValueError):merge(item)

from analyze_long_replay import is_cache_copy
@pytest.mark.parametrize("size,direction,expected",[(917504,"HtoD",True),(3*917504,"DtoH",True),(8,"HtoD",False),(32768,"HtoD",False),(917504,"DtoD",False)])
def test_fused_copy_filter_excludes_metadata(size,direction,expected):
    assert is_cache_copy(dict(cat="gpu_memcpy",name="Memcpy "+direction,args=dict(bytes=size))) is expected

from analyze_long_replay import count_token_events
def test_actual_event_count_and_missing_evidence_rejection():
    row=dict(per_token_complete=True,output_tokens=[1,2],engine_token_timestamps=[1.,2.],ambiguous_events=[])
    assert count_token_events([row])==2
    row.pop("engine_token_timestamps")
    with pytest.raises(ValueError):count_token_events([row])
