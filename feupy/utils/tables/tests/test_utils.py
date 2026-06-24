# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.table import Table

from feupy.utils.tables.utils import (
    pad_list_to_length,
    remove_nan_rows,
)


def test_pad_list_to_length_ok():
    lst = [1, 2, 3]
    out = pad_list_to_length(5, lst)

    assert len(out) == 5
    assert out[-1] is np.nan


def test_pad_list_to_length_error():
    with pytest.raises(ValueError):
        pad_list_to_length(2, [1, 2, 3])


def test_remove_nan_rows():
    table = Table(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )

    filtered = remove_nan_rows(table)

    assert len(filtered) == 2
    assert not np.isnan(filtered["a"]).any()