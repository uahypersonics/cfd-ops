"""Smoke test: package imports and exposes a version."""

import cfd_ops


def test_version():
    assert cfd_ops.__version__
