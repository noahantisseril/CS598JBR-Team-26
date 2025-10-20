from HumanEval_5 import intersperse

def test_case_empty():
    assert intersperse([], 4) == []

def test_case_single_element():
    assert intersperse([5], 4) == [5]

def test_case_typical():
    assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]

def test_case_negative():
    assert intersperse([-1, -2, -3], -4) == [-1, -4, -2, -4, -3]

def test_case_large():
    assert intersperse([100, 200, 300], 400) == [100, 400, 200, 400, 300]

def test_case_zero():
    assert intersperse([0, 0, 0], 4) == [0, 4, 0, 4, 0]