import pytest
from HumanEval_106 import f

def test_case_1():
    assert f(0) == []

def test_case_2():
    assert f(1) == [1]

def test_case_3():
    assert f(5) == [1, 2, 6, 24, 15]

def test_case_4():
    assert f(10) == [1, 2, 6, 24, 15, 14, 42, 362, 726, 2002]

def test_case_5():
    assert f(15) == [1, 2, 6, 24, 15, 14, 42, 362, 726, 2002, 7480, 50050, 399168, 3628800, 39916800]

def test_case_6():
    assert f(1) != [2]

def test_case_7():
    assert f(5) != [1, 3, 6, 24, 15]

def test_case_8():
    assert f(10) != [1, 3, 6, 24, 15, 14, 42, 362, 726, 2002]

def test_case_9():
    assert f(15) != [1, 3, 6, 24, 15, 14, 42, 362, 726, 2002, 7480, 50050, 399168, 3628800, 39916801]

def test_case_10():
    assert f(1) == [1]