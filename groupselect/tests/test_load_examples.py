import pytest


def test_load_examples():
    try:
        import groupselect.examples
    except Exception as ex:
        raise ex
