# Licensed under a 3-clause BSD style license - see LICENSE
"""Geometry utilities."""

from gammapy.data import FixedPointingInfo
from gammapy.maps import MapAxis, RegionGeom
from regions import CircleSkyRegion

__all__ = [
    "create_energy_axis",
    "create_pointing",
    "create_pointing_position",
    "define_on_region",
    "create_region_geometry",
]


def create_energy_axis(
    energy_min,
    energy_max,
    nbin=5,
    per_decade=True,
    name="energy",
):
    """Create an energy axis.

    Parameters
    ----------
    energy_min : `~astropy.units.Quantity`
        Minimum energy.
    energy_max : `~astropy.units.Quantity`
        Maximum energy.
    nbin : int, optional
        Number of bins. Default is 5.
    per_decade : bool, optional
        Whether ``nbin`` is interpreted as the number of bins per decade.
        Default is True.
    name : str, optional
        Axis name. Default is ``"energy"``.

    Returns
    -------
    axis : `~gammapy.maps.MapAxis`
        Energy axis.
    """
    return MapAxis.from_energy_bounds(
        energy_min=energy_min,
        energy_max=energy_max,
        nbin=nbin,
        per_decade=per_decade,
        name=name,
    )


def create_pointing_position(position, position_angle, separation):
    """Create a pointing position offset from a sky position.

    Parameters
    ----------
    position : `~astropy.coordinates.SkyCoord`
        Reference sky position.
    position_angle : `~astropy.units.Quantity`
        Position angle of the offset.
    separation : `~astropy.units.Quantity`
        Angular separation from the reference position.

    Returns
    -------
    pointing_position : `~astropy.coordinates.SkyCoord`
        Offset pointing position.
    """
    return position.directional_offset_by(position_angle, separation)


def create_pointing(pointing_position):
    """Create fixed pointing information.

    Parameters
    ----------
    pointing_position : `~astropy.coordinates.SkyCoord`
        Pointing sky position.

    Returns
    -------
    pointing : `~gammapy.data.FixedPointingInfo`
        Fixed pointing information.
    """
    return FixedPointingInfo(fixed_icrs=pointing_position.icrs)


def define_on_region(center, radius):
    """Create a circular on-region.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Region center.
    radius : `~astropy.units.Quantity`
        Region radius.

    Returns
    -------
    region : `~regions.CircleSkyRegion`
        Circular sky region.
    """
    return CircleSkyRegion(center=center, radius=radius)


def create_region_geometry(on_region, axes):
    """Create a region geometry.

    Parameters
    ----------
    on_region : `~regions.SkyRegion`
        Sky region used to define the geometry.
    axes : list of `~gammapy.maps.MapAxis`
        Non-spatial axes.

    Returns
    -------
    geom : `~gammapy.maps.RegionGeom`
        Region geometry.
    """
    return RegionGeom.create(region=on_region, axes=axes)
