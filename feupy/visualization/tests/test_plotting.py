# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import patch

from feupy.visualization.styles.markers.plotting import (
    build_fp_kwargs,
    build_point_kwargs,
)


def test_build_point_kwargs():
    palette = [["red", "red"], ["blue", "blue"]]

    with (
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker",
            side_effect=["o", "*"],
        ),
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker_size",
            side_effect=[6.0, 9.6],
        ),
    ):
        result = build_point_kwargs(
            ["source-a", "source-b"],
            marker_size=6,
            palette=palette,
        )

    assert result == {
        "source-a": {
            "label": "source-a",
            "marker": "o",
            "markersize": 6.0,
            "color": "red",
        },
        "source-b": {
            "label": "source-b",
            "marker": "*",
            "markersize": 9.6,
            "color": "blue",
        },
    }


def test_build_point_kwargs_cycles_palette():
    palette = [["red", "red"]]

    with (
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker",
            return_value="o",
        ),
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker_size",
            return_value=6.0,
        ),
    ):
        result = build_point_kwargs(
            ["source-a", "source-b"],
            palette=palette,
        )

    assert result["source-a"]["color"] == "red"
    assert result["source-b"]["color"] == "red"


def test_build_point_kwargs_passes_uniform_size():
    with (
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker",
            return_value="*",
        ),
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker_size",
            return_value=6.0,
        ) as mock_size,
    ):
        build_point_kwargs(
            ["source-a"],
            marker_size=6,
            uniform_size=False,
        )

    mock_size.assert_called_once_with("*", 6, False)


def test_build_fp_kwargs_without_sources():
    palette = [["green", "green"]]

    with (
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker",
            return_value="s",
        ),
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker_size",
            return_value=6.0,
        ),
    ):
        result = build_fp_kwargs(
            ["dataset-a"],
            palette=palette,
        )

    assert result == {
        "dataset-a": {
            "label": "dataset-a",
            "marker": "s",
            "markersize": 6.0,
            "color": "green",
            "ls": "solid",
        }
    }


def test_build_fp_kwargs_with_sources():
    palette = [["red", "red"], ["blue", "blue"]]

    with (
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker",
            side_effect=["o", "^"],
        ),
        patch(
            "feupy.visualization.styles.markers.plotting.resolve_marker_size",
            side_effect=[6.0, 7.2],
        ),
    ):
        result = build_fp_kwargs(
            labels=["dataset-a", "dataset-b"],
            sources=["source-a", "source-b"],
            palette=palette,
        )

    assert result["source-a"]["label"] == "dataset-a"
    assert result["source-a"]["color"] == "red"
    assert result["source-b"]["label"] == "dataset-b"
    assert result["source-b"]["color"] == "blue"
