from evaluation.protocol import make_student_split

def test_deterministic_split_and_query_not_candidate():
    a,_=make_student_split('stu', list(range(30)), [0,1]*15, seed=42, valid_count=30)
    b,_=make_student_split('stu', list(range(30)), [0,1]*15, seed=42, valid_count=30)
    assert a == b
    assert set(a.query_item_ids).isdisjoint(set(a.support_item_ids))
    assert a.warm_start_item in a.support_item_ids

def test_item_bounds_filter():
    sp,_=make_student_split('stu', [-1,0,1,2,99,3,4,5,6,7], [1]*10, seed=1, valid_count=8, min_query_items=2)
    assert sp is not None
    assert all(0 <= x < 8 for x in sp.support_item_ids + sp.query_item_ids)
