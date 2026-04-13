# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import warnings

from feupy.utils.tables._legacy import (
    column_to_string,
    append_nones,
)


def test_column_to_string():
    col = [1, 2, 3]
    out = column_to_string(col)

    assert out == "[1,2,3]"


def test_append_nones():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        out = append_nones(5, [1, 2])

        assert len(out) == 5
        assert out[-1] is None

        # check deprecation warning
        assert any("deprecated" in str(wi.message).lower() for wi in w)


def test_append_nones_error():
    with pytest.raises(ValueError):
        append_nones(2, [1, 2, 3])