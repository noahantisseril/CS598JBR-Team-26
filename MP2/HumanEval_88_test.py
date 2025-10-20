import pytest
from HumanEval_88 import sort_array

def test_case_1():
    assert sort_array([]) == []

def test_case_2():
    assert sort_array([5]) == [5]

def test_case_3():
    assert sort_array([2, 4, 3, 0, 1, 5]) == [0, 1, 2, 3, 4, 5]

def test_case_4():
    assert sort_array([2, 4, 3, 0, 1, 5, 6]) == [6, 5, 4, 3, 2, 1, 0]

def test_case_5():
    assert sort_array([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

def test_case_6():
    assert sort_array([5, 4, 3, 2, 1]) == [5, 4, 3, 2, 1]

def test_case_7():
    assert sort_array([1, 2, 3, 4, 5, 6]) == [6, 5, 4, 3, 2, 1]

def test_case_8():
    assert sort_array([6, 5, 4, 3, 2, 1]) == [6, 5, 4, 3, 2, 1]

def test_case_9():
    assert sort_array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

def test_case_10():
    assert sort_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1