# Licensed under a 3-clause BSD style license - see LICENSE

import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import FixedPointingInfo, PointingMode
from gammapy.maps import MapAxis, RegionGeom
from regions import CircleSkyRegion

from feupy.utils import geometry
from feupy.utils.geometry import (
    create_energy_axis,
    create_pointing,
    create_pointing_position,
    create_region_geometry,
    define_on_region,
)


def test_all():
    expected = {
        "create_energy_axis",
        "create_pointing",
        "create_pointing_position",
        "define_on_region",
        "create_region_geometry",
    }

    assert set(geometry.__all__) == expected

    for name in geometry.__all__:
        assert hasattr(geometry, name)


def test_create_energy_axis():
    axis = create_energy_axis(
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
        nbin=4,
        per_decade=False,
    )

    assert isinstance(axis, MapAxis)
    assert axis.name == "energy"
    assert axis.nbin == 4


def test_create_pointing_position():
    position = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs")

    pointing_position = create_pointing_position(
        position=position,
        position_angle=0 * u.deg,
        separation=1 * u.deg,
    )

    assert isinstance(pointing_position, SkyCoord)
    assert pointing_position.separation(position).to_value(u.deg) == 1.0


def test_create_pointing():
    pointing_position = SkyCoord(
        ra=83.63 * u.deg,
        dec=22.01 * u.deg,
        frame="icrs",
    )

    pointing = create_pointing(pointing_position)

    assert isinstance(pointing, FixedPointingInfo)
    assert pointing.mode == PointingMode.POINTING
    assert pointing.fixed_icrs.separation(pointing_position).to_value(u.deg) == 0.0


def test_define_on_region():
    center = SkyCoord(
        ra=83.63 * u.deg,
        dec=22.01 * u.deg,
        frame="icrs",
    )

    region = define_on_region(center=center, radius=0.2 * u.deg)

    assert isinstance(region, CircleSkyRegion)
    assert region.center.separation(center).to_value(u.deg) == 0.0
    assert region.radius == 0.2 * u.deg


def test_create_region_geometry():
    center = SkyCoord(
        ra=83.63 * u.deg,
        dec=22.01 * u.deg,
        frame="icrs",
    )
    region = define_on_region(center=center, radius=0.2 * u.deg)
    axis = create_energy_axis(
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
        nbin=4,
        per_decade=False,
    )

    geom = create_region_geometry(region, axes=[axis])

    assert isinstance(geom, RegionGeom)
    assert geom.axes["energy"].nbin == 4
