import numpy as np


def _print_failure(name, actual, expected, details=None):
    print(f"[FAIL] {name}")
    print(f"  expected: {expected!r}")
    print(f"  actual:   {actual!r}")
    if details:
        print("  details:")
        for line in str(details).splitlines():
            print(f"    {line}")


def check_equal(name, actual, expected):
    try:
        assert actual == expected
    except AssertionError:
        _print_failure(name, actual, expected)
        raise
    print(f"[PASS] {name}")


def check_is_none(name, actual):
    try:
        assert actual is None
    except AssertionError:
        _print_failure(name, actual, None)
        raise
    print(f"[PASS] {name}")


def check_true(name, condition, actual=None, expected=True):
    try:
        assert condition
    except AssertionError:
        shown_actual = condition if actual is None else actual
        _print_failure(name, shown_actual, expected)
        raise
    print(f"[PASS] {name}")


def check_array_equal(name, actual, expected):
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        _print_failure(name, actual, expected, error)
        raise
    print(f"[PASS] {name}")


def check_allclose(name, actual, expected, rtol=1e-7, atol=0):
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        _print_failure(name, actual, expected, error)
        raise
    print(f"[PASS] {name}")
