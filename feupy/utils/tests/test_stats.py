# Licensed under a 3-clause BSD style license - see LICENSE

from unittest.mock import MagicMock

import numpy as np
import pytest
from gammapy.stats import WStatCountsStatistic

from feupy.utils import stats
from feupy.utils.stats import (
    compute_significance,
    compute_wstat,
    sigma_to_ts,
    ts_to_sigma,
)


def test_all():
    expected = {
        "sigma_to_ts",
        "ts_to_sigma",
        "compute_wstat",
        "compute_significance",
    }

    assert set(stats.__all__) == expected

    for name in stats.__all__:
        assert hasattr(stats, name)


@pytest.mark.parametrize("sigma", [1, 2, 3, 5])
def test_sigma_ts_roundtrip(sigma):
    ts = sigma_to_ts(sigma)
    result = ts_to_sigma(ts)

    assert result == pytest.approx(sigma)


def test_sigma_to_ts_monotonic():
    assert sigma_to_ts(2) > sigma_to_ts(1)


def test_sigma_to_ts_different_df():
    assert sigma_to_ts(3, df=2) != pytest.approx(sigma_to_ts(3, df=1))


def test_compute_wstat():
    statistic = compute_wstat(n_on=10, n_off=5, alpha=0.2)

    assert isinstance(statistic, WStatCountsStatistic)
    assert statistic.n_on == pytest.approx(10)
    assert statistic.n_off == pytest.approx(5)
    assert statistic.alpha == pytest.approx(0.2)
    assert statistic.sqrt_ts >= 0


def test_compute_significance():
    dataset = MagicMock()
    dataset.counts.data = np.array([5, 7, 8])
    dataset.counts_off.data = np.array([10, 12, 13])

    result = compute_significance(dataset, alpha=0.2)
    expected = WStatCountsStatistic(
        n_on=20,
        n_off=35,
        alpha=0.2,
    ).sqrt_ts

    assert result == pytest.approx(expected)
