import pytest
from HumanEval_75 import is_multiply_prime

def test_case_1():
    assert is_multiply_prime(30) == True

def test_case_2():
    assert is_multiply_prime(105) == True

def test_case_3():
    assert is_multiply_prime(200) == False

def test_case_4():
    assert is_multiply_prime(0) == False

def test_case_5():
    assert is_multiply_prime(-10) == False

def test_case_6():
    assert is_multiply_prime(13) == False

def test_case_7():
    assert is_multiply_prime(3) == False

def test_case_8():
    assert is_multiply_prime(21) == True

def test_case_9():
    assert is_multiply_prime(14) == False

def test_case_10():
    assert is_multiply_prime(646) == True

def test_case_11():
    assert is_multiply_prime(12) == False

def test_case_12():
    assert is_multiply_prime(97) == False