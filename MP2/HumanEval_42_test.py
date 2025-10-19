import pytest
from HumanEval_42 import incr_list

def test_case_1():
    assert incr_list([1, 2, 3]) == [2, 3, 4]

def test_case_2():
    assert incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [6, 4, 6, 3, 4, 4, 10, 1, 124]

def test_case_3():
    assert incr_list([0]) == [1]

def test_case_4():
    assert incr_list([-1, -2, -3]) == [0, -1, -2]

def test_case_5():
    assert incr_list([]) == []

def test_case_6():
    assert incr_list([-1]) == [0]

def test_case_7():
    assert incr_list([100, 200, 300]) == [101, 201, 301]

def test_case_8():
    assert incr_list([-100, -200, -300]) == [-99, -199, -299]