import pytest
from HumanEval_61 import correct_bracketing

def test_case_1():
    assert correct_bracketing("(") == False

def test_case_2():
    assert correct_bracketing("()") == True

def test_case_3():
    assert correct_bracketing("(()())") == True

def test_case_4():
    assert correct_bracketing("") == True

def test_case_5():
    assert correct_bracketing("))")) == False

def test_case_6():
    assert correct_bracketing("()()()()") == True

def test_case_7():
    assert correct_bracketing("(((())))") == True

def test_case_8():
    assert correct_bracketing("("*100000 + ")("*100000) == False

def test_case_9():
    assert correct_bracketing("))))))))))))))))))))))))))))))))))))") == False

def test_case_10():
    assert correct_bracketing("((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((