from evaluation.protocol import make_student_split

def test_policies_share_same_split_object():
    sp,_=make_student_split('s', list(range(20)), [0,1]*10, seed=42, valid_count=20)
    random_query=list(sp.query_item_ids)
    mirt_query=list(sp.query_item_ids)
    assert random_query == mirt_query
