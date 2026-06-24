# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from feupy.utils.aic import calculate_aic, calculate_relative_aic


class FakeFreeParameters:
    def __init__(self, n):
        self.names = ["a"] * n


class FakeParameters:
    def __init__(self, n):
        self.free_parameters = FakeFreeParameters(n)


class FakeModels:
    def __init__(self, n_free):
        self.parameters = FakeParameters(n_free)


class FakeFitResult:
    def __init__(self, success=True, stat=10.0, k=2):
        self.success = success
        self.total_stat = stat
        self.models = FakeModels(k)


class FakeDataset:
    def __init__(self):
        self.data = type(
            "d",
            (),
            {"is_ul": type("u", (), {"data": np.array([False, False])})},
        )()


def test_calculate_aic_basic():
    datasets = [FakeDataset()]
    fit = FakeFitResult(success=True, stat=10.0, k=2)

    aic = calculate_aic(datasets, fit)

    assert isinstance(aic, float)
    assert aic > 0


def test_calculate_aic_fail():
    datasets = [FakeDataset()]
    fit = FakeFitResult(success=False)

    with pytest.raises(ValueError):
        calculate_aic(datasets, fit)


def test_relative_aic():
    datasets = [FakeDataset()]

    fit0 = FakeFitResult(stat=10.0, k=2)
    fit1 = FakeFitResult(stat=8.0, k=2)

    delta = calculate_relative_aic(datasets, fit0, fit1)

    assert isinstance(delta, float)