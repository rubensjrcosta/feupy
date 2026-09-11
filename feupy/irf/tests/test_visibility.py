# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.coordinates import SkyCoord

from feupy.irf.visibility import (
    CTAOVisibilityEstimator,
    make_ctao_visibility_table,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def target():
    """Simple test target (Crab-like position)."""
    return SkyCoord(83.63, 22.01, unit="deg")


@pytest.fixture
def estimator(target):
    """Default estimator (fast setup)."""
    return CTAOVisibilityEstimator(
        target=target,
        year=2025,
        time_step_min=60,  # faster test
    )


# -----------------------------------------------------------------------------
# Basic functionality
# -----------------------------------------------------------------------------


def test_estimator_init(estimator):
    assert estimator.year == 2025
    assert estimator.dt == 60
    assert estimator.target is not None


def test_compute_visibility_keys(estimator):
    vis = estimator.compute_visibility("cta_south", show_progress=False)

    assert isinstance(vis, dict)
    assert set(vis.keys()) == {"20", "40", "60"}


def test_compute_visibility_positive(estimator):
    vis = estimator.compute_visibility("cta_south", show_progress=False)

    for val in vis.values():
        assert val >= 0


# -----------------------------------------------------------------------------
# Physical sanity checks
# -----------------------------------------------------------------------------


def test_visibility_not_all_zero(estimator):
    vis = estimator.compute_visibility("cta_south", show_progress=False)

    assert any(v > 0 for v in vis.values())


def test_visibility_diff_between_sites(estimator):
    vis_south = estimator.compute_visibility("cta_south", show_progress=False)
    vis_north = estimator.compute_visibility("cta_north", show_progress=False)

    # They should not be identical
    assert vis_south != vis_north


# -----------------------------------------------------------------------------
# Airmass weighting
# -----------------------------------------------------------------------------


def test_airmass_weight_changes_result(target):
    est1 = CTAOVisibilityEstimator(target, time_step_min=60, use_airmass_weight=False)
    est2 = CTAOVisibilityEstimator(target, time_step_min=60, use_airmass_weight=True)

    vis1 = est1.compute_visibility("cta_south", show_progress=False)
    vis2 = est2.compute_visibility("cta_south", show_progress=False)

    assert vis1 != vis2


# -----------------------------------------------------------------------------
# Table generation
# -----------------------------------------------------------------------------


def test_make_visibility_table(estimator):
    df = make_ctao_visibility_table(estimator, show_progress=False)

    assert not df.empty
    assert "Observatory" in df.columns
    assert "Zenith (deg)" in df.columns
    assert "Visibility (hours)" in df.columns


def test_table_has_both_sites(estimator):
    df = make_ctao_visibility_table(estimator, show_progress=False)

    assert "CTAO South" in df["Observatory"].values
    assert "CTAO North" in df["Observatory"].values


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_invalid_observatory(estimator):
    with pytest.raises(KeyError):
        estimator.compute_visibility("invalid_site", show_progress=False)


def test_zero_time_step(target):
    with pytest.raises(Exception):
        CTAOVisibilityEstimator(target, time_step_min=0)
