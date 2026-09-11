# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
import pytest

from feupy.utils.aic import calculate_aic, calculate_relative_aic


class FakeFreeParameters:
    def __init__(self, n_parameters):
        self.names = [f"par-{index}" for index in range(n_parameters)]


class FakeParameters:
    def __init__(self, n_parameters):
        self.free_parameters = FakeFreeParameters(n_parameters)


class FakeModels:
    def __init__(self, n_parameters):
        self.parameters = FakeParameters(n_parameters)


class FakeFitResult:
    def __init__(
        self,
        success=True,
        total_stat=10.0,
        n_parameters=2,
        models=True,
    ):
        self.success = success
        self.total_stat = total_stat

        if models:
            self.models = FakeModels(n_parameters)


class FakeDataset:
    def __init__(self, n_points=10, n_upper_limits=0):
        is_ul = np.zeros(n_points, dtype=bool)
        is_ul[:n_upper_limits] = True

        self.data = type(
            "FakeData",
            (),
            {
                "is_ul": type(
                    "FakeUpperLimitMask",
                    (),
                    {"data": is_ul},
                )()
            },
        )()


def test_calculate_aic_basic():
    datasets = [FakeDataset(n_points=10)]
    fit_result = FakeFitResult(
        success=True,
        total_stat=10.0,
        n_parameters=2,
    )

    aicc = calculate_aic(datasets, fit_result)

    assert aicc == pytest.approx(15.714285714285714)


def test_calculate_aic_multiple_datasets():
    datasets = [
        FakeDataset(n_points=5),
        FakeDataset(n_points=7),
    ]
    fit_result = FakeFitResult(
        total_stat=20.0,
        n_parameters=2,
    )

    aicc = calculate_aic(datasets, fit_result)

    expected = 24.0 + 12.0 / 9.0
    assert aicc == pytest.approx(expected)


def test_calculate_aic_excludes_upper_limits():
    datasets = [FakeDataset(n_points=10, n_upper_limits=2)]
    fit_result = FakeFitResult(
        total_stat=10.0,
        n_parameters=2,
    )

    aicc = calculate_aic(datasets, fit_result)

    expected = 14.0 + 12.0 / 5.0
    assert aicc == pytest.approx(expected)


def test_calculate_aic_unsuccessful_fit():
    datasets = [FakeDataset()]
    fit_result = FakeFitResult(success=False)

    with pytest.raises(ValueError, match="Fit was not successful"):
        calculate_aic(datasets, fit_result)


def test_calculate_aic_missing_models():
    datasets = [FakeDataset()]
    fit_result = FakeFitResult(models=False)

    with pytest.raises(AttributeError, match="models is missing"):
        calculate_aic(datasets, fit_result)


def test_calculate_aic_invalid_models():
    datasets = [FakeDataset()]
    fit_result = FakeFitResult()
    fit_result.models = []

    with pytest.raises(TypeError, match="free_parameters"):
        calculate_aic(datasets, fit_result)


def test_calculate_aic_insufficient_points():
    datasets = [FakeDataset(n_points=3)]
    fit_result = FakeFitResult(n_parameters=2)

    with pytest.raises(ValueError, match=r"N > k \+ 1"):
        calculate_aic(datasets, fit_result)


def test_calculate_relative_aic():
    datasets = [FakeDataset(n_points=10)]

    fit_h0 = FakeFitResult(
        total_stat=10.0,
        n_parameters=2,
    )
    fit_h1 = FakeFitResult(
        total_stat=8.0,
        n_parameters=2,
    )

    relative_aic = calculate_relative_aic(
        datasets,
        fit_h0,
        fit_h1,
    )

    aic_h0 = 14.0 + 12.0 / 7.0
    aic_h1 = 12.0 + 12.0 / 7.0
    expected = (1 - aic_h1 / aic_h0) * 100

    assert relative_aic == pytest.approx(expected)
    assert relative_aic > 0
