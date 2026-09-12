# Licensed under a 3-clause BSD style license - see LICENSE.rst

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from astropy.table import Table

from feupy.visualization.sensitivity import (
    plot_irfs,
    plot_irfs_superpose,
    plot_sensitivity_from_table,
    plot_tables_sensitivity,
)


def make_sensitivity_table(name="IRF A", int_sens=None):
    table = Table()
    table["e_ref"] = [0.1, 1.0, 10.0] * u.TeV
    table["e_min"] = [0.05, 0.5, 5.0] * u.TeV
    table["e_max"] = [0.15, 1.5, 15.0] * u.TeV
    table["e2dnde"] = [1e-11, 5e-12, 1e-12] * u.Unit("TeV cm-2 s-1")
    table["excess"] = [10, 20, 30] * u.dimensionless_unscaled
    table["background"] = [100, 80, 60] * u.dimensionless_unscaled
    table["on_radii"] = [0.2, 0.15, 0.1] * u.deg

    table.meta["IRF_NAME"] = name

    if int_sens is not None:
        table.meta["INT_SENS"] = int_sens

    return table


@pytest.mark.parametrize(
    ("which", "ylabel"),
    [
        ("excess", "Excess counts"),
        ("background", "Background counts"),
    ],
)
def test_plot_sensitivity_from_table_counts(which, ylabel):
    table = make_sensitivity_table()

    fig, ax = plt.subplots()

    try:
        result = plot_sensitivity_from_table(
            table,
            which=which,
            ax=ax,
        )

        assert result is ax
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert "Energy" in ax.get_xlabel()
        assert ax.get_ylabel() == ylabel
    finally:
        plt.close(fig)


def test_plot_sensitivity_from_table_with_xerr():
    table = make_sensitivity_table()
    ax = MagicMock()

    with patch("feupy.visualization.sensitivity.quantity_support") as quantity_support:
        quantity_support.return_value.__enter__.return_value = None
        quantity_support.return_value.__exit__.return_value = None

        plot_sensitivity_from_table(
            table,
            ax=ax,
            plot_xerr=True,
            label="test",
        )

    kwargs = ax.errorbar.call_args.kwargs

    assert kwargs["label"] == "test"
    assert u.allclose(
        kwargs["xerr"],
        (table["e_max"] - table["e_min"]) / 2,
    )


def test_plot_irfs_requires_list():
    with pytest.raises(
        TypeError,
        match="`tables` must be a list of tables",
    ):
        plot_irfs("not-a-list")


def test_plot_irfs():
    table = make_sensitivity_table(
        name="South",
        int_sens=1e-12 * u.Unit("TeV cm-2 s-1"),
    )

    fig, ax = plt.subplots()

    try:
        with (
            patch(
                "feupy.visualization.sensitivity.plot_sensitivity_from_table"
            ) as plot_table,
            patch.object(ax, "legend") as legend,
        ):
            result = plot_irfs(
                [table],
                ax=ax,
                which="e2dnde",
            )

        assert result is ax
        plot_table.assert_called_once()

        kwargs = plot_table.call_args.kwargs
        assert kwargs["which"] == "e2dnde"
        assert kwargs["ax"] is ax
        assert kwargs["label"].startswith("South (")
        assert "1.00e-12" in kwargs["label"]

        legend.assert_called_once_with(
            loc="best",
            fontsize=7,
        )
    finally:
        plt.close(fig)


def test_plot_irfs_without_integrated_sensitivity_label():
    table = make_sensitivity_table(
        name="South",
        int_sens=1e-12 * u.Unit("TeV cm-2 s-1"),
    )

    fig, ax = plt.subplots()

    try:
        with (
            patch(
                "feupy.visualization.sensitivity.plot_sensitivity_from_table"
            ) as plot_table,
            patch.object(ax, "legend"),
        ):
            plot_irfs(
                [table],
                ax=ax,
                int_sens_label=False,
            )

        assert plot_table.call_args.kwargs["label"] == "South"
    finally:
        plt.close(fig)


def test_plot_irfs_with_model():
    table = make_sensitivity_table()
    spectral_model = MagicMock()
    model = SimpleNamespace(
        name="model-a",
        spectral_model=spectral_model,
    )
    energy_bounds = [0.1, 10] * u.TeV

    fig, ax = plt.subplots()

    try:
        with (
            patch("feupy.visualization.sensitivity.plot_sensitivity_from_table"),
            patch.object(ax, "legend"),
        ):
            plot_irfs(
                [table],
                model=model,
                energy_bounds=energy_bounds,
                ax=ax,
            )

        spectral_model.plot.assert_called_once()
        spectral_model.plot_error.assert_called_once()

        kwargs = spectral_model.plot.call_args.kwargs
        assert kwargs["label"] == "model-a"
        assert kwargs["color"] == "k"
        assert kwargs["sed_type"] == "e2dnde"
        assert u.allclose(kwargs["energy_bounds"], energy_bounds)
    finally:
        plt.close(fig)


def test_plot_irfs_superpose_requires_lists():
    with pytest.raises(
        TypeError,
        match="Both inputs must be lists",
    ):
        plot_irfs_superpose([], "not-a-list")


def test_plot_irfs_superpose():
    south = make_sensitivity_table("South")
    north = make_sensitivity_table("North")

    fig, ax = plt.subplots()

    try:
        with (
            patch(
                "feupy.visualization.sensitivity.plot_sensitivity_from_table"
            ) as plot_table,
            patch.object(ax, "legend") as legend,
        ):
            result = plot_irfs_superpose(
                [south],
                [north],
                ax=ax,
            )

        assert result is ax
        assert plot_table.call_count == 2

        labels = [call.kwargs["label"] for call in plot_table.call_args_list]
        assert labels == ["South", "North"]

        legend.assert_called_once_with(
            loc="best",
            fontsize=7,
        )
    finally:
        plt.close(fig)


def test_plot_tables_sensitivity_requires_lists():
    with pytest.raises(
        TypeError,
        match="`tables_south` must be a list",
    ):
        plot_tables_sensitivity("not-a-list")

    with pytest.raises(
        TypeError,
        match="`tables_north` must be a list",
    ):
        plot_tables_sensitivity(
            [],
            tables_north="not-a-list",
        )


def test_plot_tables_sensitivity_south_only():
    table = make_sensitivity_table()

    with patch("feupy.visualization.sensitivity.plot_irfs") as plot_irfs_mock:
        fig, ax = plot_tables_sensitivity(
            [table],
            box_name="CTAO South",
            sens_info="50 h",
        )

    try:
        plot_irfs_mock.assert_called_once()
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        assert tuple(ax.get_xlim()) == pytest.approx((3e-2, 1e2))
        assert len(ax.texts) == 2
        assert ax.texts[0].get_text() == "CTAO South"
        assert ax.texts[1].get_text() == "50 h"
    finally:
        plt.close(fig)


def test_plot_tables_sensitivity_superpose():
    south = make_sensitivity_table("South")
    north = make_sensitivity_table("North")

    with (
        patch("feupy.visualization.sensitivity.plot_irfs_superpose") as superpose,
        patch("feupy.visualization.sensitivity.plot_irfs") as plot_irfs_mock,
    ):
        fig, ax = plot_tables_sensitivity(
            [south],
            tables_north=[north],
        )

    try:
        superpose.assert_called_once()
        plot_irfs_mock.assert_not_called()
    finally:
        plt.close(fig)


def test_plot_tables_sensitivity_with_models_and_save():
    table = make_sensitivity_table()

    spectral_model = MagicMock()
    model = SimpleNamespace(
        name="intrinsic",
        spectral_model=spectral_model,
    )

    abs_spectral_model = MagicMock()
    abs_model = SimpleNamespace(
        name="absorbed",
        spectral_model=abs_spectral_model,
    )

    energy_bounds = [0.1, 20] * u.TeV

    with (
        patch("feupy.visualization.sensitivity.plot_irfs"),
        patch("feupy.visualization.sensitivity.plt.savefig") as savefig,
    ):
        fig, ax = plot_tables_sensitivity(
            [table],
            model=model,
            abs_model=abs_model,
            file_path="sensitivity.png",
            energy_bounds=energy_bounds,
            ylim=[1e-13, 1e-9],
            xaxis_label="Energy custom",
            yaxis_label="Sensitivity custom",
        )

    try:
        spectral_model.plot.assert_called_once()
        spectral_model.plot_error.assert_called_once()
        abs_spectral_model.plot.assert_called_once()
        abs_spectral_model.plot_error.assert_called_once()

        abs_kwargs = abs_spectral_model.plot.call_args.kwargs
        assert abs_kwargs["linestyle"] == "--"
        assert abs_kwargs["color"] == "black"

        assert ax.get_xlabel() == "Energy custom"
        assert ax.get_ylabel() == "Sensitivity custom"
        assert tuple(ax.get_xlim()) == pytest.approx((0.1, 20))

        savefig.assert_called_once_with("sensitivity.png")
    finally:
        plt.close(fig)
