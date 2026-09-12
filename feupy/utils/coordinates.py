# Licensed under a 3-clause BSD style license - see LICENSE
"""Utilities for coordinate conversions."""

import astropy.units as u
from astropy.coordinates import SkyCoord

__all__ = [
    "convert_skycoord_to_dict",
    "convert_pos_config_to_skycoord",
    "convert_dict_to_skycoord",
    "convert_table_to_skycoord",
]


def convert_skycoord_to_dict(position):
    """Convert a sky coordinate to a dictionary.

    Parameters
    ----------
    position : `~astropy.coordinates.SkyCoord`
        Input sky coordinate.

    Returns
    -------
    position_dict : dict
        Dictionary containing ``lon``, ``lat``, and ``frame``.
    """
    return {
        "lon": position.spherical.lon,
        "lat": position.spherical.lat,
        "frame": position.frame.name,
    }


def convert_pos_config_to_skycoord(pos_config):
    """Convert a position configuration to a sky coordinate.

    Parameters
    ----------
    pos_config : object
        Configuration-like object providing ``lon``, ``lat``, and ``frame``
        attributes.

    Returns
    -------
    position : `~astropy.coordinates.SkyCoord`
        Sky coordinate.

    Raises
    ------
    AttributeError
        If one of the required attributes is missing.
    """
    required_attributes = ("lon", "lat", "frame")

    for attribute in required_attributes:
        if not hasattr(pos_config, attribute):
            raise AttributeError(f"pos_config missing '{attribute}'")

    return SkyCoord(
        pos_config.lon,
        pos_config.lat,
        frame=pos_config.frame,
    )


def convert_dict_to_skycoord(pos_dict):
    """Convert a position dictionary to a sky coordinate.

    The dictionary can contain either ``lon`` and ``lat`` or ``ra`` and
    ``dec``. Values without units are interpreted as degrees.

    Parameters
    ----------
    pos_dict : dict
        Position dictionary. The coordinate frame defaults to ``"icrs"``.

    Returns
    -------
    position : `~astropy.coordinates.SkyCoord`
        Sky coordinate.

    Raises
    ------
    KeyError
        If no supported coordinate keys are found.
    """
    if {"lon", "lat"}.issubset(pos_dict):
        lon_key, lat_key = "lon", "lat"
    elif {"ra", "dec"}.issubset(pos_dict):
        lon_key, lat_key = "ra", "dec"
    else:
        raise KeyError("Dictionary must contain ('lon', 'lat') or ('ra', 'dec').")

    frame = pos_dict.get("frame", "icrs")
    lon = pos_dict[lon_key]
    lat = pos_dict[lat_key]

    if not hasattr(lon, "unit"):
        lon = lon * u.deg

    if not hasattr(lat, "unit"):
        lat = lat * u.deg

    return SkyCoord(lon, lat, frame=frame)


def convert_table_to_skycoord(table):
    """Convert coordinate columns from a table to a sky coordinate.

    Supported column pairs are ``RAJ2000/DEJ2000``,
    ``RAJ2000/DECJ2000``, ``RA/DEC``, and ``ra/dec``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table containing coordinate columns.

    Returns
    -------
    position : `~astropy.coordinates.SkyCoord`
        Sky coordinates corresponding to the table rows.

    Raises
    ------
    KeyError
        If no supported coordinate columns are found.
    """
    column_names = table.colnames

    if {"RAJ2000", "DEJ2000"}.issubset(column_names):
        lon_name, lat_name, frame = "RAJ2000", "DEJ2000", "icrs"
    elif {"RAJ2000", "DECJ2000"}.issubset(column_names):
        lon_name, lat_name, frame = "RAJ2000", "DECJ2000", "fk5"
    elif {"RA", "DEC"}.issubset(column_names):
        lon_name, lat_name, frame = "RA", "DEC", "icrs"
    elif {"ra", "dec"}.issubset(column_names):
        lon_name, lat_name, frame = "ra", "dec", "icrs"
    else:
        raise KeyError(
            "No valid coordinate columns found "
            "(RA/DEC, ra/dec, RAJ2000/DEJ2000, RAJ2000/DECJ2000)."
        )

    unit = table[lon_name].unit.to_string() if table[lon_name].unit else "deg"

    return SkyCoord(
        table[lon_name],
        table[lat_name],
        unit=unit,
        frame=frame,
    )
