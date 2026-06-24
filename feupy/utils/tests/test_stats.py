# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from feupy.utils.stats import (
    sigma_to_ts,
    ts_to_sigma,
    compute_wstat,
)


def test_sigma_to_ts_monotonic():
    ts1 = sigma_to_ts(1)
    ts2 = sigma_to_ts(2)

    assert ts2 > ts1


def test_ts_to_sigma_roundtrip():
    ts = sigma_to_ts(3)
    sigma = ts_to_sigma(ts)

    assert np.isclose(sigma, 3, atol=0.1)


def test_compute_wstat():
    stat = compute_wstat(n_on=10, n_off=5, alpha=0.2)

    assert stat.sqrt_ts >= 0