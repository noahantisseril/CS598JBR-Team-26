import pytest
from HumanEval_102 import choose_num

def test_case_1():
    assert choose_num(12, 15) == 14

def test_case_2():
    assert choose_num(13, 12) == -1

def test_case_3():
    assert choose_num(0, 2) == 0

def test_case_4():
    assert choose_num(1, 1) == -1

def test_case_5():
    assert choose_num(2, 4) == 4

def test_case_6():
    assert choose_num(10, 10) == -1

def test_case_7():
    assert choose_num(5, 7) == 6

def test_case_8():
    assert choose_num(11, 13) == 12

def test_case_9():
    assert choose_num(9, 9) == -1

def test_case_10():
    assert choose_num(2, 2) == -1