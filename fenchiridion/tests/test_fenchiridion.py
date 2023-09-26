"""
Unit and regression test for the fenchiridion package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import fenchiridion


def test_fenchiridion_imported():
    """Sample test, will always pass so long as import statement worked."""
    print("importing ", fenchiridion.__name__)
    assert "fenchiridion" in sys.modules


# Assert that a certain exception is raised
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
