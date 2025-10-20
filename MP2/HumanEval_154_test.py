from HumanEval_154 import cycpattern_check

def test_case_1():
    assert cycpattern_check("abcd","abd") == False

def test_case_2():
    assert cycpattern_check("hello","ell") == True

def test_case_3():
    assert cycpattern_check("whassup","psus") == False

def test_case_4():
    assert cycpattern_check("abab","baa") == True

def test_case_5():
    assert cycpattern_check("efef","eeff") == False

def test_case_6():
    assert cycpattern_check("himenss","simen") == True

def test_case_7():
    assert cycpattern_check("","") == False

def test_case_8():
    assert cycpattern_check("a","") == False

def test_case_9():
    assert cycpattern_check("","b") == False

def test_case_10():
    assert cycpattern_check("a","b") == False