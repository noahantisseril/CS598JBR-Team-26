import pytest
from HumanEval_125 import split_words

def test_case_1():
    assert split_words("Hello world!") == ["Hello", "world!"]

def test_case_2():
    assert split_words("Hello,world!") == ["Hello", "world!"]

def test_case_3():
    assert split_words("abcdef") == 3

def test_case_4():
    assert split_words("") == 0

def test_case_5():
    assert split_words("ABCDEF") == 0

def test_case_6():
    assert split_words("aBcDeF") == 3

def test_case_7():
    assert split_words("123456") == 0

def test_case_8():
    assert split_words(",") == ['']

def test_case_9():
    assert split_words(" ") == ['']

def test_case_10():
    assert split_words("Hello,World!") == ["Hello", "World!"]