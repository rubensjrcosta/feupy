# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import PowerLawSpectralModel

from feupy.utils.datasets import (
    cut_energy_flux_points_datasets,
    datasets_select_by_name,
    flux_points_dataset_from_table,
    get_energy_bounds_from_datasets,
    get_feupy_data_path,
)


def make_flux_points_table():
    """Create a small flux-points table for tests."""
    table = Table()
    table["e_ref"] = [1.0, 3.0, 10.0] * u.TeV
    table["e_min"] = [0.8, 2.5, 8.0] * u.TeV
    table["e_max"] = [1.2, 3.5, 12.0] * u.TeV
    table["dnde"] = [1.0e-12, 5.0e-13, 1.0e-13] / (u.cm**2 * u.s * u.TeV)
    table["dnde_err"] = [1.0e-13, 5.0e-14, 1.0e-14] / (u.cm**2 * u.s * u.TeV)
    return table


def make_flux_points_dataset(name="test"):
    """Create a small FluxPointsDataset for tests."""
    flux_points = FluxPoints.from_table(
        make_flux_points_table(),
        sed_type="dnde",
        format="gadf-sed",
    )
    return FluxPointsDataset(data=flux_points, name=name)


# -------------------------
# FEUPY_DATA path
# -------------------------
def test_get_feupy_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv("FEUPY_DATA", str(tmp_path))

    path = get_feupy_data_path()

    assert path == tmp_path.resolve()


def test_get_feupy_data_path_missing(monkeypatch):
    monkeypatch.delenv("FEUPY_DATA", raising=False)

    with pytest.raises(RuntimeError, match="FEUPY_DATA"):
        get_feupy_data_path()


def test_get_feupy_data_path_not_found(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    monkeypatch.setenv("FEUPY_DATA", str(missing))

    with pytest.raises(RuntimeError, match="does not exist"):
        get_feupy_data_path()


# -------------------------
# Dataset selection
# -------------------------
def test_datasets_select_by_name():
    datasets = Datasets(
        [
            make_flux_points_dataset("dataset-a"),
            make_flux_points_dataset("dataset-b"),
        ]
    )

    selected = datasets_select_by_name(datasets, ["dataset-b"])

    assert isinstance(selected, Datasets)
    assert len(selected) == 1
    assert selected[0].name == "dataset-b"


def test_datasets_select_by_name_multiple():
    datasets = Datasets(
        [
            make_flux_points_dataset("dataset-a"),
            make_flux_points_dataset("dataset-b"),
            make_flux_points_dataset("dataset-c"),
        ]
    )

    selected = datasets_select_by_name(
        datasets,
        ["dataset-a", "dataset-c"],
    )

    assert [dataset.name for dataset in selected] == [
        "dataset-a",
        "dataset-c",
    ]


# -------------------------
# Energy bounds
# -------------------------
def test_get_energy_bounds_single_dataset():
    dataset = make_flux_points_dataset()

    energy_bounds = get_energy_bounds_from_datasets(dataset)

    assert energy_bounds.unit == u.TeV
    assert energy_bounds[0].value == pytest.approx(0.8)
    assert energy_bounds[1].value == pytest.approx(12.0)


def test_get_energy_bounds_multiple_datasets():
    dataset_a = make_flux_points_dataset("dataset-a")

    table = make_flux_points_table()
    table["e_min"] = [0.5, 2.5, 8.0] * u.TeV
    table["e_max"] = [1.2, 3.5, 20.0] * u.TeV

    flux_points = FluxPoints.from_table(
        table,
        sed_type="dnde",
        format="gadf-sed",
    )
    dataset_b = FluxPointsDataset(
        data=flux_points,
        name="dataset-b",
    )

    energy_bounds = get_energy_bounds_from_datasets(Datasets([dataset_a, dataset_b]))

    assert energy_bounds[0].to_value(u.TeV) == pytest.approx(0.5)
    assert energy_bounds[1].to_value(u.TeV) == pytest.approx(20.0)


# -------------------------
# Energy selection
# -------------------------
def test_cut_energy_flux_points_single_dataset():
    dataset = make_flux_points_dataset()

    result = cut_energy_flux_points_datasets(
        dataset,
        e_ref_min=2.0 * u.TeV,
        e_ref_max=5.0 * u.TeV,
    )

    assert isinstance(result, FluxPointsDataset)
    assert len(result.data.energy_ref) == 1

    assert np.all(result.data.energy_ref >= 2.0 * u.TeV)
    assert np.all(result.data.energy_ref <= 5.0 * u.TeV)


# -------------------------
# Table -> FluxPointsDataset
# -------------------------
def test_flux_points_dataset_from_table():
    table = make_flux_points_table()

    dataset = flux_points_dataset_from_table(
        table,
        sed_type="dnde",
        name="test-dataset",
    )

    assert isinstance(dataset, FluxPointsDataset)
    assert dataset.name == "test-dataset"
    assert len(dataset.data.energy_ref) == 3


def test_flux_points_dataset_from_table_with_model():
    table = make_flux_points_table()
    model = PowerLawSpectralModel()

    dataset = flux_points_dataset_from_table(
        table,
        reference_model=model,
        sed_type="dnde",
        name="test-dataset",
        model_name="test-model",
    )

    assert len(dataset.models) == 1
    assert dataset.models[0].name == "test-model"
