import pytest
from HumanEval_13 import greatest_common_divisor

def test_case_1():
    assert greatest_common_divisor(3, 5) == 1

def test_case_2():
    assert greatest_common_divisor(25, 15) == 5

def test_case_3():
    assert greatest_common_divisor(0, 5) == 5

def test_case_4():
    assert greatest_common_divisor(5, 0) == 5

def test_case_5():
    assert greatest_common_divisor(0, 0) == 0

def test_case_6():
    assert greatest_common_divisor(1, 1) == 1

def test_case_7():
    assert greatest_common_divisor(-5, 15) == 5

def test_case_8():
    assert greatest_common_divisor(15, -5) == 5

def test_case_9():
    assert greatest_common_divisor(-5, -15) == 5

def test_case_10():
    assert greatest_common_divisor(-15, 5) == 5