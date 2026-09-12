# Licensed under a 3-clause BSD style license - see LICENSE.rst

import matplotlib.pyplot as plt

from feupy.visualization.styles.linestyles import LINESTYLES_DEFAULT


def test_linestyles_count():
    assert len(LINESTYLES_DEFAULT) == 14


def test_linestyles_include_standard_styles():
    assert "solid" in LINESTYLES_DEFAULT
    assert "dotted" in LINESTYLES_DEFAULT
    assert "dashed" in LINESTYLES_DEFAULT
    assert "dashdot" in LINESTYLES_DEFAULT


def test_linestyles_are_valid():
    fig, ax = plt.subplots()

    try:
        for linestyle in LINESTYLES_DEFAULT:
            line = ax.plot([0, 1], [0, 1], linestyle=linestyle)[0]
            assert line.get_linestyle() is not None
    finally:
        plt.close(fig)
