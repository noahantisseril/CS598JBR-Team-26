import pytest
from HumanEval_21 import rescale_to_unit

def test_case_1():
    assert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_case_2():
    assert rescale_to_unit([-1.0, 1.0]) == [0.0, 1.0]

def test_case_3():
    assert rescale_to_unit([0.0]) == [0.0]

def test_case_4():
    assert rescale_to_unit([-1.0, -0.5, 0.0, 0.5, 1.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_case_5():
    assert rescale_to_unit([100.0, 200.0, 300.0]) == [0.0, 0.5, 1.0]