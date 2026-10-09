# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Porter (2017) index, FITS reader and interpolator.

Synthetic fixtures avoid requiring the 23,760 original GALPROP FITS files.
"""

from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from feupy.isrf.porter2017 import (
    Porter2017ISRF,
    read_flux_fits,
    read_galprop_index,
    skycoord_to_galprop,
)


@pytest.fixture
def synthetic_model(tmp_path):
    """Make a 2 x 2 x 2 grid with four wavelengths and known SEDs."""
    model_dir = tmp_path / "R12"
    model_dir.mkdir()
    radius = [0.1, 10.0]
    azimuth = [0.0, 180.0]
    height = [-2.0, 2.0]
    index_lines = [
        "!!3D", "8 4 12 2 2 2",
        " ".join(map(str, radius)),
        " ".join(map(str, azimuth)),
        " ".join(map(str, height)),
        "1.0 1.0", "0", "0", "0",
    ]
    for cell_id in range(8):
        iz, ir, iphi = np.unravel_index(cell_id, (2, 2, 2))
        z, r, phi = height[iz], radius[ir], azimuth[iphi]
        name = f"cell_{cell_id}_Flux.fits.gz"
        index_lines.extend([
            str(cell_id),
            f"{r} 0 {z}",
            "1 1 1",
            *([f"unused_{cell_id}_{k}.dat" for k in range(10)]),
            name,
        ])
        # Constant across photon energy, affine in z and r; independent of phi.
        value = 5 + 0.1 * r + 0.2 * z
        components = {
            "Direct": 0.5 * value,
            "Scattered": 0.2 * value,
            "Transient": 0.2 * value,
            "Thermal": 0.1 * value,
            "Total": value,
        }
        columns = [
            fits.Column(
                name="Wavelength",
                format="D",
                array=np.array([1, 10, 100, 1000], dtype=float),
            )
        ]
        for name_component, component_value in components.items():
            columns.append(
                fits.Column(
                    name=name_component,
                    format="D",
                    array=np.full(4, component_value),
                )
            )
        table = fits.BinTableHDU.from_columns(columns, name="Energy Density")
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(model_dir / name)
    (model_dir / "robitaille_DL07_PAHISMMix.dat").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_index(synthetic_model):
    index = read_galprop_index(
        synthetic_model / "R12" / "robitaille_DL07_PAHISMMix.dat"
    )
    assert index["n_cells"] == 8
    assert index["n_wavelength"] == 4
    assert len(index["flux_files"]) == 8
    np.testing.assert_array_equal(index["r"], [0.1, 10])


def test_fits_reader(synthetic_model):
    filename = synthetic_model / "R12" / "cell_0_Flux.fits.gz"
    energy, sed = read_flux_fits(filename)
    assert energy.size == 4
    assert np.all(np.diff(energy.to_value(u.eV)) > 0)
    assert sed.unit.is_equivalent(u.eV / u.cm**3)
    with pytest.raises(ValueError):
        read_flux_fits(filename, component="NotAComponent")


def test_spectrum_grid_and_component_sum(synthetic_model):
    model = Porter2017ISRF(data_dir=synthetic_model, model="R12")
    # l=0, b=0, d=0.5 => R=8.0 kpc, z=0, phi=0.
    position = SkyCoord(l=0 * u.deg, b=0 * u.deg,
                        distance=0.5 * u.kpc, frame="galactic")
    energy, total = model.spectrum(position)
    expected = 5 + 0.1 * 8
    np.testing.assert_allclose(
        total.to_value(u.eV / u.cm**3),
        expected,
        rtol=1e-7,
    )

    parts = sum(model.spectrum(position, component=name)[1] for name in
                ("Direct", "Scattered", "Transient", "Thermal"))
    assert u.allclose(parts, total, rtol=1e-12)
    assert model.energy.size == 4
    assert model.energy_density(position).unit.is_equivalent(u.eV / u.cm**3)
    e, n = model.photon_density(position)
    assert u.allclose(e**2 * n, total, rtol=1e-12)


def test_vectorized_positions(synthetic_model):
    model = Porter2017ISRF(data_dir=synthetic_model)
    position = SkyCoord(l=[0, 0] * u.deg, b=[0, 0] * u.deg,
                        distance=[0.5, 1.5] * u.kpc, frame="galactic")
    _, sed = model.spectrum(position)
    assert sed.shape == (2, 4)
    np.testing.assert_allclose(
        sed[0].to_value(u.eV / u.cm**3),
        5.8,
        rtol=1e-7,
    )
    
    np.testing.assert_allclose(
        sed[1].to_value(u.eV / u.cm**3),
        5.7,
        rtol=1e-7,
    )


def test_bounds_error(synthetic_model):
    model = Porter2017ISRF(data_dir=synthetic_model)
    # Galactic centre: R~0, below the minimum radius 0.1 kpc.
    position = SkyCoord(l=0 * u.deg, b=0 * u.deg,
                        distance=8.5 * u.kpc, frame="galactic")
    with pytest.raises(ValueError, match="outside GALPROP grid"):
        model.spectrum(position)


def test_to_naima_format(synthetic_model):
    model = Porter2017ISRF(data_dir=synthetic_model)
    position = SkyCoord(l=0 * u.deg, b=0 * u.deg,
                        distance=0.5 * u.kpc, frame="galactic")
    seed = model.to_naima(position)
    assert len(seed) == 3
    assert seed[0] == "Porter2017_R12_Total"
    assert seed[1].unit.is_equivalent(u.eV)
    assert seed[2].unit.is_equivalent(u.eV / u.cm**3)


def test_coordinate_convention():
    from astropy.coordinates import Galactocentric
    frame = Galactocentric(galcen_distance=8.5 * u.kpc, z_sun=0 * u.pc,
                          roll=0 * u.deg)
    position = SkyCoord(l=17.8 * u.deg, b=-0.7 * u.deg,
                        distance=4 * u.kpc, frame="galactic")
    radius, phi, height = skycoord_to_galprop(position, frame)
    assert np.isclose(radius.to_value(u.kpc), 4.848466, atol=1e-4)
    assert np.isclose(phi.to_value(u.deg), 345.393446, atol=1e-3)
    assert np.isclose(height.to_value(u.kpc), -0.048861, atol=1e-4)
