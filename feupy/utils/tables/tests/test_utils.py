# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
import pytest
from astropy.table import Table

from feupy.utils.tables.utils import pad_list_to_length, remove_nan_rows


@pytest.mark.parametrize(
    ("length", "input_list", "expected_length"),
    [(5, [1, 2, 3], 5), (3, [1, 2, 3], 3), (2, [], 2)],
)
def test_pad_list_to_length(length, input_list, expected_length):
    result = pad_list_to_length(length, input_list)

    assert len(result) == expected_length
    assert result[: len(input_list)] == input_list


def test_pad_list_to_length_padding():
    result = pad_list_to_length(4, [1, 2])

    assert np.isnan(result[-1])
    assert np.isnan(result[-2])


def test_pad_list_to_length_error():
    with pytest.raises(ValueError, match="longer"):
        pad_list_to_length(2, [1, 2, 3])


def test_remove_nan_rows():
    table = Table(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )

    result = remove_nan_rows(table)

    assert len(result) == 2
    assert not np.isnan(result["a"]).any()
    assert list(result["label"]) == ["a", "c"]


def test_remove_nan_rows_without_nan():
    table = Table({"a": [1.0, 2.0], "b": [1, 2]})

    result = remove_nan_rows(table)

    assert len(result) == 2
