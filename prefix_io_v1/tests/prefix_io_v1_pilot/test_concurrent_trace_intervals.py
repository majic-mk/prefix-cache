from analyze_concurrent_pilot import merged,overlaps,distribution

def test_union_prevents_duplicate_overlap():
    assert merged([(1,3),(2,4),(4,5),(9,9),(7,8)])==[[1,5],[7,8]]

def test_touching_boundary_is_not_overlap():
    rows=merged([(1,3),(5,7)])
    assert not overlaps(rows,3,5)
    assert overlaps(rows,2,4)
    assert overlaps(rows,4,6)

def test_empty_and_nested_intervals():
    assert not overlaps([],1,2)
    assert merged([(0,10),(2,3),(4,5)])==[[0,10]]
    assert not overlaps([[0,10]],2,2)

def test_empty_distribution_is_unknown_not_zero():
    assert distribution([])==dict(n=0,median=None,p95=None,max=None)
