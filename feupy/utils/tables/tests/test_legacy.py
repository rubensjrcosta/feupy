# Licensed under a 3-clause BSD style license - see LICENSE

import pytest

from feupy.utils.tables._legacy import append_nones, column_to_string


def test_column_to_string():
    with pytest.warns(DeprecationWarning):
        result = column_to_string([1, 2, 3])

    assert result == "[1,2,3]"


def test_append_nones():
    with pytest.warns(DeprecationWarning):
        result = append_nones(5, [1, 2])

    assert result == [1, 2, None, None, None]


def test_append_nones_error():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="longer"):
            append_nones(2, [1, 2, 3])
