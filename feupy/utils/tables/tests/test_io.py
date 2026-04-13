# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import numpy as np
import pytest
from astropy.table import Table

from feupy.utils.tables.io import read_table, write_table


@pytest.fixture
def sample_table():
    return Table(
        {
            "x": [1, 2, 3],
            "y": [0.1, 0.2, 0.3],
        }
    )


def test_write_read_fits(tmp_path, sample_table):
    file_name = "test.fits"

    write_table(sample_table, tmp_path, file_name, overwrite=True)
    loaded = read_table(tmp_path, file_name)

    assert len(loaded) == 3
    assert np.allclose(loaded["x"], sample_table["x"])


def test_write_read_csv(tmp_path, sample_table):
    file_name = "test.csv"

    write_table(sample_table, tmp_path, file_name, overwrite=True)
    loaded = read_table(tmp_path, file_name)

    assert len(loaded) == 3
    assert np.allclose(loaded["y"], sample_table["y"])


def test_invalid_extension(tmp_path, sample_table):
    with pytest.raises(ValueError):
        write_table(sample_table, tmp_path, "test.txt")