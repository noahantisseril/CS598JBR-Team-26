import pytest
from HumanEval_39 import prime_fib

def test_case_1():
    assert prime_fib(1) == 2

def test_case_2():
    assert prime_fib(2) == 3

def test_case_3():
    assert prime_fib(3) == 5

def test_case_4():
    assert prime_fib(4) == 13

def test_case_5():
    assert prime_fib(5) == 89

def test_case_6():
    assert prime_fib(10) == 44945

def test_case_7():
    assert prime_fib(15) == 3702484

def test_case_8():
    assert prime_fib(20) == 28657

def test_case_9():
    assert prime_fib(25) == 28657

def test_case_10():
    assert prime_fib(30) == 139583