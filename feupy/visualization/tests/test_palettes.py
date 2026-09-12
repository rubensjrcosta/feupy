# Licensed under a 3-clause BSD style license - see LICENSE.rst

from matplotlib.colors import is_color_like

from feupy.visualization.styles.palettes import (
    PALETTE_DEFAULT,
    PALETTE_IBM,
    PALETTE_TABLEAU,
    PALETTE_WONG,
)

PALETTES = [
    PALETTE_DEFAULT,
    PALETTE_TABLEAU,
    PALETTE_IBM,
    PALETTE_WONG,
]


def test_palette_entries_have_color_and_label():
    for palette in PALETTES:
        assert all(len(entry) == 2 for entry in palette)


def test_palette_colors_are_valid():
    for palette in PALETTES:
        for color, _ in palette:
            assert is_color_like(color)


def test_palette_labels_are_nonempty_strings():
    for palette in PALETTES:
        for _, label in palette:
            assert isinstance(label, str)
            assert label


def test_palette_colors_are_unique():
    for palette in PALETTES:
        colors = [color for color, _ in palette]

        assert len(colors) == len(set(colors))


def test_expected_palette_sizes():
    assert len(PALETTE_DEFAULT) == 36
    assert len(PALETTE_TABLEAU) == 9
    assert len(PALETTE_IBM) == 12
    assert len(PALETTE_WONG) == 12
