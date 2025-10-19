import pytest
from HumanEval_126 import is_sorted

def test_case_1():
    assert is_sorted([5]) == True

def test_case_2():
    assert is_sorted([1, 2, 3, 4, 5]) == True

def test_case_3():
    assert is_sorted([1, 3, 2, 4, 5]) == False

def test_case_4():
    assert is_sorted([1, 2, 3, 4, 5, 6]) == True

def test_case_5():
    assert is_sorted([1, 2, 3, 4, 5, 6, 7]) == True

def test_case_6():
    assert is_sorted([1, 3, 2, 4, 5, 6, 7]) == False

def test_case_7():
    assert is_sorted([1, 2, 2, 3, 3, 4]) == True

def test_case_8():
    assert is_sorted([1, 2, 2, 2, 3, 4]) == False

def test_case_9():
    assert is_sorted([1, 1, 2, 2, 3, 4]) == True

def test_case_10():
    assert is_sorted([1, 2, 1, 3, 4, 5]) == False

def test_case_11():
    assert is_sorted([1, 2, 3, 3, 4, 5]) == False

def test_case_12():
    assert is_sorted([1, 2, 3, 4, 4, 5]) == True

def test_case_13():
    assert is_sorted([1, 1, 2, 3, 4, 5]) == False

def test_case_14():
    assert is_sorted([1, 1, 2, 2, 3, 3]) == True