from evaluation.protocol import clean_interactions

def test_invalid_ids_filtered_uniformly():
    items,resps=clean_interactions([0,5,10,-1,2], [1,0,1,1,0], 6)
    assert items == [0,5,2]
    assert resps == [1,0,0]
