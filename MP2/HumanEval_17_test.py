import pytest
from HumanEval_17 import parse_music

def test_case_1():
    assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]

def test_case_2():
    assert parse_music('o o| o| o| o| o| o| o| o| o|') == [4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4]

def test_case_3():
    assert parse_music('') == []

def test_case_4():
    assert parse_music('o o| .| o| .| .| .| .| .| .| .|') == [4, 2, 1, 2, 1, 1, 1, 1, 1, 1]

def test_case_5():
    assert parse_music('o| o| o| o| o| o| o| o| o| o|') == [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]