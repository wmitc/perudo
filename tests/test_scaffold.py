"""Smoke test so the suite is green from the first commit."""

import perudo


def test_package_imports_and_has_version():
    assert isinstance(perudo.__version__, str)
    assert perudo.__version__
