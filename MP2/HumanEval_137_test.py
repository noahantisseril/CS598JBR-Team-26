from HumanEval_137 import compare_one

def test_case_1():
    assert compare_one(1, 2.5) == 2.5

def test_case_2():
    assert compare_one(1, "2.3") == "2.3"

def test_case_3():
    assert compare_one("5.1", "6") == "6"

def test_case_4():
    assert compare_one("1", 1) == None

def test_case_5():
    assert compare_one(2, 2) == None

def test_case_6():
    assert compare_one("0.1", "0,1") == "0,1"

def test_case_7():
    assert compare_one(-1, 1) == 1

def test_case_8():
    assert compare_one("-1", "-5.6") == "-5.6"

def test_case_9():
    assert compare_one("1", "-1") == "1"

def test_case_10():
    assert compare_one(-1, -5.6) == -1