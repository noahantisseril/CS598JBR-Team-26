import pytest
from HumanEval_161 import solve

def test_case_1():
    assert solve("1234") == "4321"

def test_case_2():
    assert solve("ab") == "AB"

def test_case_3():
    assert solve("#a@C") == "#A@c"

def test_case_4():
    assert solve("aBcD") == "AbCd"

def test_case_5():
    assert solve("") == ""

def test_case_6():
    assert solve("123aBcD") == "DcBa321"

def test_case_7():
    assert solve("!@#") == "#@!"

def test_case_8():
    assert solve("@1B$") == "@1b$"

def test_case_9():
    assert solve("a1b2c3") == "A1B2C3"

def test_case_10():
    assert solve("!@1B$") == "!@1b$"