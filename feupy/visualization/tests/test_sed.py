# Licensed under a 3-clause BSD style license - see LICENSE.rst

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

from feupy.visualization.sed import SEDPlotter


class DummyCollection(list):
    """Minimal list-like collection exposing a ``names`` property."""

    @property
    def names(self):
        return [item.name for item in self]


def test_default_configuration():
    plotter = SEDPlotter(DummyCollection(), sed_type="e2dnde")

    axis = plotter._default_axis()
    limits = plotter._default_limits()
    legend = plotter._default_legend()

    assert axis["units"] == ("TeV", "TeV cm-2 s-1")
    assert "TeV" in axis["label"][0]
    assert "E^{2}" in axis["label"][1]

    assert u.allclose(limits["energy_bounds"], [1e-5, 2e3] * u.TeV)
    assert limits["ylim"] == [1e-23, 1e-7]

    assert legend["loc"] == "lower left"
    assert legend["frameon"] is False


def test_save_plot():
    plotter = SEDPlotter(DummyCollection())

    with patch("feupy.visualization.sed.plt.savefig") as savefig:
        plotter._save_plot("sed.png")

    savefig.assert_called_once_with(
        "sed.png",
        dpi=300,
        bbox_inches="tight",
    )


def test_save_plot_none():
    plotter = SEDPlotter(DummyCollection())

    with patch("feupy.visualization.sed.plt.savefig") as savefig:
        plotter._save_plot(None)

    savefig.assert_not_called()


def test_plot_datasets_with_model_error():
    spectral_model = MagicMock()
    dataset_model = SimpleNamespace(spectral_model=spectral_model)

    models = MagicMock()
    models.names = ["dataset-a"]
    models.__getitem__.return_value = dataset_model

    dataset = SimpleNamespace(
        name="dataset-a",
        data=MagicMock(),
        models=models,
    )

    plotter = SEDPlotter(DummyCollection([dataset]))
    energy_bounds = [1, 10] * u.TeV

    with patch(
        "feupy.visualization.sed.get_energy_bounds_from_datasets",
        return_value=energy_bounds,
    ):
        plotter._plot_datasets(
            ax=MagicMock(),
            plot_kwargs={
                "ax": MagicMock(),
                "sed_type": "e2dnde",
            },
            ref_markers={
                "dataset-a": {
                    "marker": "o",
                    "markersize": 4,
                    "color": "red",
                }
            },
        )

    dataset.data.plot.assert_called_once()
    spectral_model.plot_error.assert_called_once()

    kwargs = spectral_model.plot_error.call_args.kwargs

    assert u.allclose(kwargs["energy_bounds"], energy_bounds)
    assert kwargs["edgecolor"] == "red"
    assert kwargs["facecolor"] == "red"
    assert kwargs["alpha"] == 0.2


def test_plot_models_without_models():
    plotter = SEDPlotter(DummyCollection(), models=None)

    plotter._plot_models(
        ax=MagicMock(),
        plot_kwargs={},
        energy_bounds=[1, 10] * u.TeV,
        show_error=True,
        ref_markers={},
    )


def test_plot_models_with_error_band():
    spectral_model = MagicMock()

    model = SimpleNamespace(
        name="model-a",
        spectral_model=spectral_model,
    )

    plotter = SEDPlotter(
        DummyCollection(),
        models=DummyCollection([model]),
    )

    energy_bounds = [1, 10] * u.TeV

    plotter._plot_models(
        ax=MagicMock(),
        plot_kwargs={"sed_type": "e2dnde"},
        energy_bounds=energy_bounds,
        show_error=True,
        ref_markers={"model-a": {"color": "blue"}},
    )

    spectral_model.plot.assert_called_once()
    spectral_model.plot_error.assert_called_once()

    plot_kwargs = spectral_model.plot.call_args.kwargs
    error_kwargs = spectral_model.plot_error.call_args.kwargs

    assert plot_kwargs["label"] == "model-a"
    assert plot_kwargs["color"] == "blue"
    assert plot_kwargs["marker"] == ","
    assert u.allclose(plot_kwargs["energy_bounds"], energy_bounds)

    assert error_kwargs["facecolor"] == "blue"
    assert error_kwargs["edgecolor"] == "blue"
    assert error_kwargs["alpha"] == 0.05


def test_plot_builds_markers_and_returns_axes():
    dataset = SimpleNamespace(
        name="dataset-a",
        data=MagicMock(),
        models=None,
    )
    datasets = DummyCollection([dataset])

    spectral_model = MagicMock()
    model = SimpleNamespace(
        name="model-a",
        spectral_model=spectral_model,
    )
    models = DummyCollection([model])

    plotter = SEDPlotter(datasets, models=models)

    markers = {
        "dataset-a": {
            "label": "dataset-a",
            "marker": "o",
            "markersize": 4,
            "color": "red",
        },
        "model-a": {
            "label": "model-a",
            "marker": "s",
            "markersize": 4,
            "color": "blue",
        },
    }

    fig, ax = plt.subplots()

    try:
        with (
            patch(
                "feupy.visualization.sed.build_fp_kwargs",
                return_value=markers,
            ) as build_markers,
            patch.object(plotter, "_plot_datasets") as plot_datasets,
            patch.object(plotter, "_plot_models") as plot_models,
            patch.object(plotter, "_save_plot") as save_plot,
            patch.object(ax, "legend") as legend,
        ):
            result = plotter.plot(
                ax=ax,
                file_path="sed.png",
                box_name="test",
                error_band=True,
            )

        assert result is ax

        build_markers.assert_called_once_with(
            labels=["dataset-a", "model-a"],
            marker_size=4,
        )

        plot_datasets.assert_called_once()
        plot_models.assert_called_once()
        save_plot.assert_called_once_with("sed.png")
        legend.assert_called_once()

        assert ax.get_xlabel()
        assert ax.get_ylabel()
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "test"

    finally:
        plt.close(fig)


def test_plot_uses_custom_configuration():
    plotter = SEDPlotter(DummyCollection())

    axis = {
        "label": ("Energy custom", "Flux custom"),
        "units": ("GeV", "GeV cm-2 s-1"),
    }
    limits = {
        "energy_bounds": [1, 100] * u.GeV,
        "ylim": [1e-15, 1e-10],
    }
    legend_kwargs = {"loc": "upper right"}
    model_bounds = [10, 50] * u.GeV

    fig, ax = plt.subplots()

    try:
        with (
            patch.object(plotter, "_plot_datasets"),
            patch.object(plotter, "_plot_models") as plot_models,
            patch.object(plotter, "_save_plot"),
            patch.object(ax, "legend") as legend,
        ):
            plotter.plot(
                ax=ax,
                ref_markers={},
                axis=axis,
                limits=limits,
                kwargs_legend=legend_kwargs,
                kwargs_models={"energy_bounds": model_bounds},
            )

        assert ax.get_xlabel() == "Energy custom"
        assert ax.get_ylabel() == "Flux custom"
        assert tuple(ax.get_xlim()) == pytest.approx((1, 100))

        legend.assert_called_once_with(**legend_kwargs)

        passed_bounds = plot_models.call_args.args[2]
        assert u.allclose(passed_bounds, model_bounds)

    finally:
        plt.close(fig)
