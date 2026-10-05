import pytest
from prefix_io_control.token_timeline import TokenTimeline,percentile

def test_actual_events_keep_intervals():
    t=TokenTimeline();t.append([7],10.);t.append([7,8],10.2);t.append([7,8],10.2)
    t.append([7,8,9],10.7)
    assert t.export()["per_token_complete"]
    assert t.export()["itl_seconds"]==pytest.approx([.2,.5])

def test_chunk_does_not_fabricate_itl():
    t=TokenTimeline();t.append([7,8],10.)
    assert not t.export()["per_token_complete"]
    assert t.export()["itl_seconds"] is None
    assert len(t.times)==1

@pytest.mark.parametrize("ts",[0,float("nan"),float("inf"),None])
def test_invalid_time_rejected(ts):
    with pytest.raises(ValueError):TokenTimeline().append([1],ts)

def test_changed_output_rejected():
    t=TokenTimeline();t.append([1],10.)
    with pytest.raises(ValueError):t.append([2,3],11.)

def test_clock_reversal_rejected():
    t=TokenTimeline();t.append([1],10.)
    with pytest.raises(ValueError):t.append([1,2],9.)

def test_percentile_small_sample():
    assert percentile([], .95) is None
    assert percentile([1], .95)==1
    assert percentile([0,10],.95)==pytest.approx(9.5)
