# Licensed under a 3-clause BSD style license - see LICENSE.rst

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from feupy.visualization.roimap import ROIMapPlotter


class DummySources(list):
    """Minimal source collection exposing source labels."""

    @property
    def labels(self):
        return [source.label for source in self]


def make_plotter():
    center = SkyCoord(ra=10 * u.deg, dec=20 * u.deg, frame="icrs")
    radius = 0.5 * u.deg

    return ROIMapPlotter(center=center, radius=radius)


def test_init():
    plotter = make_plotter()

    assert plotter.center.ra.deg == 10
    assert plotter.center.dec.deg == 20
    assert plotter.radius == 0.5 * u.deg
    assert plotter.ax is None


def test_customize_legend_default():
    plotter = make_plotter()
    plotter.ax = MagicMock()

    plotter.customize_legend()

    plotter.ax.legend.assert_called_once_with(
        bbox_to_anchor=(0, -0.45),
        ncol=3,
        loc="lower left",
        markerscale=0.75,
        fontsize=5,
        labelcolor="black",
    )


def test_customize_legend_custom():
    plotter = make_plotter()
    plotter.ax = MagicMock()

    kwargs = {"loc": "upper right", "fontsize": 8}

    plotter.customize_legend(kwargs)

    plotter.ax.legend.assert_called_once_with(**kwargs)


def test_plot_roi():
    plotter = make_plotter()
    ax = MagicMock()

    with patch("feupy.visualization.roimap.RegionGeom") as region_geom:
        region_geom.return_value.plot_region.return_value = ax

        result = plotter.plot_roi(
            color="red",
            linestyle=":",
        )

    assert result is ax
    assert plotter.ax is ax

    region_geom.return_value.plot_region.assert_called_once_with(
        color="red",
        linestyle=":",
    )


def test_plot_sources_builds_markers():
    plotter = make_plotter()
    plotter.ax = MagicMock(name="initial_ax")

    sources = DummySources(
        [
            SimpleNamespace(
                label="source-a",
                position=SkyCoord(ra=1 * u.deg, dec=2 * u.deg),
            ),
            SimpleNamespace(
                label="source-b",
                position=SkyCoord(ra=3 * u.deg, dec=4 * u.deg),
            ),
        ]
    )

    ref_markers = {
        "source-a": {
            "label": "source-a",
            "marker": "o",
            "markersize": 6,
            "color": "red",
        },
        "source-b": {
            "label": "source-b",
            "marker": "s",
            "markersize": 6,
            "color": "blue",
        },
    }

    first_ax = MagicMock(name="first_ax")
    second_ax = MagicMock(name="second_ax")

    with (
        patch(
            "feupy.visualization.roimap.build_point_kwargs",
            return_value=ref_markers,
        ) as build_markers,
        patch("feupy.visualization.roimap.RegionGeom") as region_geom,
    ):
        region_geom.return_value.plot_region.side_effect = [
            first_ax,
            second_ax,
        ]

        plotter.plot_sources(sources)

    build_markers.assert_called_once_with(
        sources,
        marker_size=6,
        palette=None,
    )

    assert region_geom.call_count == 2
    assert region_geom.return_value.plot_region.call_count == 2
    assert plotter.ax is second_ax

    first_call = region_geom.return_value.plot_region.call_args_list[0]
    first_kwargs = first_call.kwargs

    assert first_kwargs["facecolor"] == "red"
    assert first_kwargs["edgecolor"] == "black"
    assert first_kwargs["kwargs_point"]["fillstyle"] == "full"
    assert first_kwargs["kwargs_point"]["lw"] == 0


def test_plot_sources_uses_custom_markers():
    plotter = make_plotter()
    plotter.ax = MagicMock()

    sources = DummySources(
        [
            SimpleNamespace(
                label="source-a",
                position=SkyCoord(ra=1 * u.deg, dec=2 * u.deg),
            )
        ]
    )

    ref_markers = {
        "source-a": {
            "label": "source-a",
            "marker": "D",
            "markersize": 8,
            "color": "green",
        }
    }

    with (
        patch("feupy.visualization.roimap.build_point_kwargs") as build_markers,
        patch("feupy.visualization.roimap.RegionGeom") as region_geom,
    ):
        region_geom.return_value.plot_region.return_value = MagicMock()

        plotter.plot_sources(
            sources,
            ref_markers=ref_markers,
        )

    build_markers.assert_not_called()


def test_set_axes():
    plotter = make_plotter()
    plotter.ax = MagicMock()

    plotter.set_axes(
        xlabel="RA",
        ylabel="Dec",
        size=10,
    )

    plotter.ax.set_xlabel.assert_called_once_with("RA", size=10)
    plotter.ax.set_ylabel.assert_called_once_with("Dec", size=10)
    plotter.ax.grid.assert_called_once_with(True)


def test_add_roi_text():
    plotter = make_plotter()
    plotter.ax = MagicMock()
    plotter.ax.transAxes = MagicMock()

    plotter.add_roi_text()

    plotter.ax.text.assert_called_once_with(
        0.1,
        0.93,
        "ROI (0.5 deg)",
        transform=plotter.ax.transAxes,
    )


def test_save_plot():
    plotter = make_plotter()

    with patch("feupy.visualization.roimap.plt.savefig") as savefig:
        plotter.save_plot("roi.png")

    savefig.assert_called_once_with(
        "roi.png",
        dpi=300,
        bbox_inches="tight",
    )


def test_save_plot_none():
    plotter = make_plotter()

    with patch("feupy.visualization.roimap.plt.savefig") as savefig:
        plotter.save_plot(None)

    savefig.assert_not_called()


def test_plot():
    plotter = make_plotter()

    fig, ax = plt.subplots()

    try:
        plotter.ax = ax

        sources = DummySources(
            [
                SimpleNamespace(
                    label="source-a",
                    position=SkyCoord(
                        ra=1 * u.deg,
                        dec=2 * u.deg,
                    ),
                )
            ]
        )

        markers = {"source-a": {"color": "red"}}
        legend_kwargs = {"loc": "upper right"}

        with (
            patch.object(plotter, "plot_roi") as plot_roi,
            patch.object(plotter, "plot_sources") as plot_sources,
            patch.object(plotter, "customize_legend") as customize_legend,
            patch.object(plotter, "set_axes") as set_axes,
            patch.object(plotter, "add_roi_text") as add_roi_text,
            patch.object(plotter, "save_plot") as save_plot,
        ):
            result = plotter.plot(
                sources=sources,
                file_path="roi.png",
                ref_markers=markers,
                kwargs_legend=legend_kwargs,
            )

        assert result is ax

        plot_roi.assert_called_once_with()
        plot_sources.assert_called_once_with(
            sources,
            markers,
        )
        customize_legend.assert_called_once_with(
            legend_kwargs,
        )
        set_axes.assert_called_once_with()
        add_roi_text.assert_called_once_with()
        save_plot.assert_called_once_with("roi.png")

    finally:
        plt.close(fig)
