from analyze_flush_diagnostic import match_batch
import pytest

def sample():
    return dict(ids=[2,3],ids_truncated=False,start=4.0),dict(parent_ids=[3,2],parents_truncated=False,monotonic=3.0)

def test_exact_parent_set_and_temporal_order_required():
    wait,batch=sample()
    assert match_batch(wait,[batch])["unique_match"]

@pytest.mark.parametrize("bad",["missing_parent","extra_parent","late_batch","truncated_batch","truncated_wait","ambiguous"])
def test_uncertain_join_stays_unknown(bad):
    wait,batch=sample();batches=[batch]
    if bad=="missing_parent":batch["parent_ids"]=[2]
    if bad=="extra_parent":batch["parent_ids"]=[1,2,3]
    if bad=="late_batch":batch["monotonic"]=5.
    if bad=="truncated_batch":batch["parents_truncated"]=True
    if bad=="truncated_wait":wait["ids_truncated"]=True
    if bad=="ambiguous":batches.append(dict(batch))
    result=match_batch(wait,batches)
    assert not result["unique_match"] and result["batch"] is None
