"""
Unit and regression test for the feflow package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import feflow


def test_feflow_imported():
    """Sample test, will always pass so long as import statement worked."""
    print("importing ", feflow.__name__)
    assert "feflow" in sys.modules


# Assert that a certain exception is raised
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
