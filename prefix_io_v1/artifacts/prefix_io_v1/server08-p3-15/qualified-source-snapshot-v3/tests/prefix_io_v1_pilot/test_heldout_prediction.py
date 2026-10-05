import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"))
from validate_heldout_costs import prediction_check

def test_prediction_uses_frozen_value_as_denominator():
    r=prediction_check(2.,[2.4]*4,.25)
    assert r["relative_error"]==pytest.approx(.2) and r["passed"]
def test_slow_samples_are_retained_and_can_fail():
    r=prediction_check(1.,[.1,2.,2.,10.],.25)
    assert not r["passed"] and r["samples_s"]==[.1,2.,2.,10.]
def test_faster_observation_can_also_show_miscalibration():
    assert not prediction_check(1.,[.5]*4,.25)["passed"]
@pytest.mark.parametrize("prediction,samples,tolerance",[
    (0,[1.]*4,.25),(float("nan"),[1.]*4,.25),(1,[1.]*3,.25),
    (1,[1.,float("inf"),1.,1.],.25),(1,[1.]*4,1.)])
def test_bad_prediction_inputs_fail_closed(prediction,samples,tolerance):
    with pytest.raises(ValueError):prediction_check(prediction,samples,tolerance)
