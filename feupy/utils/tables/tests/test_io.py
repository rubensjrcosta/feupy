# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
import pytest
from astropy.table import Table

from feupy.utils.tables.io import _get_format, read_table, write_table


@pytest.fixture
def sample_table():
    return Table({"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [("test.fits", "fits"), ("test.csv", "ascii.ecsv")],
)
def test_get_format(file_name, expected):
    assert _get_format(file_name) == expected


@pytest.mark.parametrize("file_name", ["test.fits", "test.csv"])
def test_write_read_table(tmp_path, sample_table, file_name):
    write_table(sample_table, tmp_path, file_name, overwrite=True)
    loaded = read_table(tmp_path, file_name)

    assert len(loaded) == len(sample_table)
    assert np.allclose(loaded["x"], sample_table["x"])
    assert np.allclose(loaded["y"], sample_table["y"])


def test_invalid_extension(tmp_path, sample_table):
    with pytest.raises(ValueError, match="must end with"):
        write_table(sample_table, tmp_path, "test.txt")
