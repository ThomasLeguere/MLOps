import avg_list

def test_simple():
    assert avg_list.avg_list([0,0,0,0]) == 0
    assert avg_list.avg_list([1,2,3,4]) == 2.5
