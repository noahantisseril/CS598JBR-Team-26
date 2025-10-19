import pytest
from HumanEval_57 import monotonic

def test_case_1():
    assert monotonic([1, 2, 4, 20]) == True

def test_case_2():
    assert monotonic([1, 20, 4, 10]) == False

def test_case_3():
    assert monotonic([4, 1, 0, -10]) == True

def test_case_4():
    assert monotonic([]) == True  # Edge case: empty list

def test_case_5():
    assert monotonic([1]) == True  # Edge case: single element

def test_case_6():
    assert monotonic([10, 9, 8, 7]) == True  # Descending order

def test_case_7():
    assert monotonic([7, 8, 9, 10]) == True  # Ascending order

def test_case_8():
    assert monotonic([1, 1, 1, 1]) == True  # All elements are same

def test_case_9():
    assert monotonic([10, 20, 30, 40]) == True  # Monotonic increasing order

def test_case_10():
    assert monotonic([40, 30, 20, 10]) == True  # Monotonic decreasing order

def test_case_11():
    assert monotonic([1, 1, 2, 2]) == True  # Two same elements, then two same elements

def test_case_12():
    assert monotonic([2, 2, 1, 1]) == True  # Two same elements, then two same elements

def test_case_13():
    assert monotonic([1, 2, 3, 4, 5, 6]) == True  # Increasing sequence

def test_case_14():
    assert monotonic([6, 5, 4, 3, 2, 1]) == True  # Decreasing sequence

def test_case_15():
    assert monotonic([-