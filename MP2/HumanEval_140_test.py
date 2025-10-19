import pytest
from HumanEval_140 import fix_spaces

def test_case_1():
    assert fix_spaces("Example") == "Example"

def test_case_2():
    assert fix_spaces("Example 1") == "Example_1"

def test_case_3():
    assert fix_spaces(" Example 2") == "_Example_2"

def test_case_4():
    assert fix_spaces(" Example   3") == "_Example-3"

def test_case_5():
    assert fix_spaces("  Example  4") == "--Example-4"

def test_case_6():
    assert fix_spaces("") == ""

def test_case_7():
    assert fix_spaces("  ") == "--"

def test_case_8():
    assert fix_spaces("   Example  5") == "---Example-5"

def test_case_9():
    assert fix_spaces("    Example    6") == "----Example--6"

def test_case_10():
    assert fix_spaces("     Example     7") == "-----Example---7"