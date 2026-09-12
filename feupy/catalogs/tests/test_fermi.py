# Licensed under a 3-clause BSD style license - see LICENSE.rst

from unittest.mock import MagicMock, patch

from feupy.catalogs.fermi import (
    get_flux_points_2PC,
    get_flux_points_3PC,
)


def test_get_flux_points_2pc():
    source = MagicMock()
    table = source.flux_points_table
    spectral_model = source.spectral_model.return_value
    expected = MagicMock()

    with patch(
        "feupy.catalogs.fermi.FluxPoints.from_table",
        return_value=expected,
    ) as from_table:
        result = get_flux_points_2PC(source)

    source.spectral_model.assert_called_once_with()
    from_table.assert_called_once_with(
        table=table,
        reference_model=spectral_model,
    )
    assert result is expected


def test_get_flux_points_3pc_default_fit():
    source = MagicMock()
    table = source.flux_points_table
    spectral_model = source.spectral_model.return_value
    expected = MagicMock()

    with patch(
        "feupy.catalogs.fermi.FluxPoints.from_table",
        return_value=expected,
    ) as from_table:
        result = get_flux_points_3PC(source)

    source.spectral_model.assert_called_once_with("auto")
    from_table.assert_called_once_with(
        table=table,
        reference_model=spectral_model,
    )
    assert result is expected


def test_get_flux_points_3pc_selected_fit():
    source = MagicMock()
    table = source.flux_points_table
    spectral_model = source.spectral_model.return_value

    with patch(
        "feupy.catalogs.fermi.FluxPoints.from_table",
    ) as from_table:
        get_flux_points_3PC(source, fit="b 23")

    source.spectral_model.assert_called_once_with("b 23")
    from_table.assert_called_once_with(
        table=table,
        reference_model=spectral_model,
    )
