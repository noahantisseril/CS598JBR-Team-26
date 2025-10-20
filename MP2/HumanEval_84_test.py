import pytest
from HumanEval_84 import solve

def test_case_1():
    assert solve(1000) == "1"

def test_case_2():
    assert solve(150) == "110"

def test_case_3():
    assert solve(147) == "1100"

def test_case_4():
    assert solve(0) == "0"

def test_case_5():
    assert solve(1) == "1"

def test_case_6():
    assert solve(10) == "1010"

def test_case_7():
    assert solve(9999) == "11111111111111111111111111111111"

def test_case_8():
    assert solve(5555) == "11111"

def test_case_9():
    assert solve(9876) == "111111110"

def test_case_10():
    assert solve(1234) == "11110"