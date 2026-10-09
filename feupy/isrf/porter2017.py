# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spatially interpolated Porter et al. (2017) GALPROP ISRF spectra.

Read the 3D GALPROP SED-only FITS data, indexed by
``robitaille_DL07_PAHISMMix.dat``, and interpolate in (z, R, phi).
The FITS spectra are treated as S(epsilon) = epsilon**2 dn/depsilon in
units of eV / cm**3, following the original analysis script.

The GALPROP convention used here has Sun at X=+8.5 kpc and
X=R_sun-d*cos(b)*cos(l), Y=-d*cos(b)*sin(l), Z=d*sin(b).
The CMB is not included by this class automatically.
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, nnls
from .utils import (
    blackbody_sed,
    integrate_isrf_sed,
    sed_to_photon_density,
)

__all__ = ["Porter2017ISRF", "read_galprop_index", "read_flux_fits", "skycoord_to_galprop"]

_COMPONENTS = ("Total", "Direct", "Scattered", "Transient", "Thermal")
_SED_UNIT = u.eV / u.cm**3
_INDEX_FILENAME = "robitaille_DL07_PAHISMMix.dat"
DEFAULT_DATA_DIR = (
    "$FEUPY_DATA/isrf/porter2017/"
    "Porter_etal_ApJ_846_67_2017_SEDonly"
)

def read_galprop_index(index_file: str | Path) -> dict:
    """Read a GALPROP 3D ISRF master index.

    Parameters
    ----------
    index_file : str or `~pathlib.Path`
        Path to ``robitaille_DL07_PAHISMMix.dat``.

    Returns
    -------
    index : dict
        Spatial axes, metadata, and ordered Flux FITS filenames.
    """
    index_file = Path(index_file).expanduser()
    with index_file.open(encoding="utf-8") as handle:
        if handle.readline().strip() != "!!3D":
            raise ValueError(f"Invalid GALPROP 3D index header: {index_file}")
        n_cells, n_wavelength, n_filters, n_r, n_phi, n_z = map(
            int, handle.readline().split()
        )
        radius = np.fromstring(handle.readline(), sep=" ", dtype=float)
        azimuth = np.fromstring(handle.readline(), sep=" ", dtype=float)
        height = np.fromstring(handle.readline(), sep=" ", dtype=float)
        if (radius.size, azimuth.size, height.size) != (n_r, n_phi, n_z):
            raise ValueError("Index axis lengths do not match header")
        if n_cells != n_r * n_phi * n_z:
            raise ValueError("Index cell count does not match grid dimensions")
        for axis, label in ((radius, "radius"), (azimuth, "azimuth"), (height, "height")):
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{label} axis must be finite and strictly increasing")
        luminosity, dust_mass = map(float, handle.readline().split())
        n_stellar = int(handle.readline())
        stellar_components = []
        for _ in range(n_stellar):
            row = handle.readline().split()
            stellar_components.append((row[0], float(row[1])))
        geometry = int(handle.readline())
        region_data = np.fromstring(handle.readline(), sep=" ", dtype=float)
        flux_files = [None] * n_cells
        cells = [None] * n_cells
        for _ in range(n_cells):
            cell_id = int(handle.readline())
            if not 0 <= cell_id < n_cells or flux_files[cell_id] is not None:
                raise ValueError(f"Duplicate or invalid cell ID: {cell_id}")
            x, y, z_cell = map(float, handle.readline().split())
            dx, dy, dz = map(float, handle.readline().split())
            filenames = [handle.readline().strip() for _ in range(11)]
            if any(not name for name in filenames):
                raise ValueError(f"Incomplete filename block for cell {cell_id}")
            flux_files[cell_id] = filenames[-1]
            cells[cell_id] = {
                "index": cell_id, "x": x, "y": y, "z": z_cell,
                "dx": dx, "dy": dy, "dz": dz,
                "flux_file": filenames[-1],
            }
    if any(name is None for name in flux_files):
        raise ValueError("Missing cell IDs in GALPROP index")
    return {
        "n_cells": n_cells, "n_wavelength": n_wavelength,
        "n_filters": n_filters, "n_r": n_r, "n_phi": n_phi, "n_z": n_z,
        "radius": radius * u.kpc, "azimuth": azimuth * u.deg,
        "height": height * u.kpc,
        "r": radius, "phi": azimuth, "z": height,
        "luminosity": luminosity, "dust_mass": dust_mass,
        "stellar_components": stellar_components, "geometry": geometry,
        "region_data": region_data, "flux_files": flux_files,
        "cells": cells, "directory": index_file.parent,
    }


def read_flux_fits(filename: str | Path, component: str = "Total") -> tuple[u.Quantity, u.Quantity]:
    """Read one Flux FITS spectrum, ordered by increasing photon energy.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Path to a ``*_Flux.fits.gz`` file.
    component : str, optional
        One of ``Total``, ``Direct``, ``Scattered``, ``Transient``, ``Thermal``.

    Returns
    -------
    energy : `~astropy.units.Quantity`
        Photon energies in eV, strictly increasing.
    sed : `~astropy.units.Quantity`
        S(epsilon) in eV / cm**3, ordered consistently with energy.
    """
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of {_COMPONENTS}")
    with fits.open(filename, memmap=False) as hdul:
        data = hdul["Energy Density"].data
        wavelength = np.asarray(data["Wavelength"], dtype=float) * u.micron
        sed = np.asarray(data[component], dtype=float) * _SED_UNIT
    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    order = np.argsort(energy.value)
    energy, sed = energy[order], sed[order]
    if (not np.all(np.isfinite(energy.value)) or
        np.any(energy.value <= 0) or np.any(np.diff(energy.value) <= 0) or
        not np.all(np.isfinite(sed.value)) or np.any(sed.value < 0)):
        raise ValueError(f"Invalid energy or SED values in {filename}")
    return energy, sed


def skycoord_to_galprop(position, frame):
    """
    Convert sky coordinates to GALPROP cylindrical coordinates.

    Parameters
    ----------
    position : astropy.coordinates.SkyCoord
        Source position with a physical distance.

    frame : astropy.coordinates.Galactocentric
        Galactocentric reference frame.

    Returns
    -------
    radius : astropy.units.Quantity
        Galactocentric cylindrical radius in kpc.

    azimuth : astropy.units.Quantity
        Galactocentric azimuth in degrees, in [0, 360).

    height : astropy.units.Quantity
        Height relative to the Galactic plane in kpc.

    Notes
    -----
    The coordinate convention follows the legacy GALPROP
    implementation used in this project:

        X = -x_astropy
        Y = -y_astropy
        Z =  z_astropy

    where x_astropy, y_astropy, and z_astropy are Cartesian
    coordinates in the Astropy Galactocentric frame.
    """
    from astropy.coordinates import SkyCoord

    if not isinstance(position, SkyCoord):
        raise TypeError(
            "position must be an astropy.coordinates.SkyCoord."
        )

    # Require physical distances.
    if not position.cartesian.x.unit.is_equivalent(u.kpc):
        raise ValueError(
            "SkyCoord must contain a physical distance."
        )

    # Transform to the configured Galactocentric frame.
    galactocentric = position.transform_to(frame)

    # Convert to the legacy GALPROP Cartesian convention.
    x = -galactocentric.x.to(u.kpc)
    y = -galactocentric.y.to(u.kpc)
    z = galactocentric.z.to(u.kpc)

    # Cylindrical coordinates.
    radius = np.hypot(x, y)

    azimuth = (
        np.arctan2(
            y.to_value(u.kpc),
            x.to_value(u.kpc),
        )
        * u.rad
    ).to(u.deg)

    azimuth = azimuth % (360 * u.deg)

    return radius, azimuth, z



class Porter2017ISRF:
    """Interpolate the Porter et al. (2017) GALPROP ISRF.

    Parameters
    ----------
    model : str, optional
        Subdirectory containing the GALPROP model, e.g. ``R12`` or ``F98``.
    data_dir : str or `~pathlib.Path`, optional
        Root directory containing the GALPROP model subdirectories.
        Defaults to ``$FEUPY_DATA/isrf/porter2017/
        Porter_etal_ApJ_846_67_2017_SEDonly``.
        The environment variable ``FEUPY_DATA`` is expanded
        automatically. Alternatively, the model directory itself
        may be supplied.
    component : str, optional
        Default spectral component. Other components load lazily on request.
    galcen_distance : `~astropy.units.Quantity`, optional
        Solar Galactocentric radius used for coordinate transformations.
    bounds_error : bool, optional
        If True, reject positions outside the spatial grid (recommended).
        If False, linear extrapolation is used outside the grid, and results
        are not guaranteed physical.

    Notes
    -----
    Data are loaded into RAM on first access for each component. One R12
    component uses approximately 24 MB before interpolation overhead.
    """

    def __init__(
        self,
        model: str = "R12",
        data_dir: str | Path = DEFAULT_DATA_DIR,
        component: str = "Total",
        galcen_distance: u.Quantity = 8.5 * u.kpc,
        bounds_error: bool = True,
    ) -> None:
        if component not in _COMPONENTS:
            raise ValueError(
                f"component must be one of {_COMPONENTS}"
            )
    
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
    
        data_dir = str(data_dir)
    
        if "$FEUPY_DATA" in data_dir and "FEUPY_DATA" not in os.environ:
            raise EnvironmentError(
                "FEUPY_DATA is not defined. "
                "Set it to the root directory of feupy-data."
            )
    
        root = Path(
            os.path.expandvars(data_dir)
        ).expanduser().resolve()
    
        self.model = str(model)
    
        self.model_dir = (
            root
            if (root / _INDEX_FILENAME).is_file()
            else root / self.model
        )
    
        self.index_file = self.model_dir / _INDEX_FILENAME
    
        if not self.index_file.is_file():
            raise FileNotFoundError(
                f"GALPROP index not found: {self.index_file}"
            )
    
        self.index = read_galprop_index(self.index_file)
    
        self.component = component
        self.bounds_error = bool(bounds_error)
    
        self.frame = Galactocentric(
            galcen_distance=u.Quantity(galcen_distance).to(u.kpc),
            z_sun=0 * u.pc,
            roll=0 * u.deg,
        )
    
        self._energy: u.Quantity | None = None
        self._interpolators: dict[str, RegularGridInterpolator] = {}

    @property
    def energy(self) -> u.Quantity:
        """Common, increasing photon-energy grid (eV)."""
        if self._energy is None:
            filename = self.model_dir / self.index["flux_files"][0]
            self._energy, _ = read_flux_fits(filename, component=self.component)
        return self._energy.copy()

    @property
    def available_components(self) -> tuple[str, ...]:
        """Available physical spectral components."""
        return _COMPONENTS

    def _get_interpolator(self, component: str) -> RegularGridInterpolator:
        if component not in _COMPONENTS:
            raise ValueError(f"component must be one of {_COMPONENTS}")
        if component in self._interpolators:
            return self._interpolators[component]
        idx = self.index
        shape = (idx["n_z"], idx["n_r"], idx["n_phi"], idx["n_wavelength"])
        spectra = np.empty(shape, dtype=np.float64)
        energy_ref = self.energy
        for cell_id, relative_filename in enumerate(idx["flux_files"]):
            filename = self.model_dir / relative_filename
            energy, sed = read_flux_fits(filename, component=component)
            if not np.allclose(energy.value, energy_ref.value, rtol=1e-10, atol=0):
                raise ValueError(f"Inconsistent photon-energy grid in {filename}")
            iz, ir, iphi = np.unravel_index(cell_id, shape[:3], order="C")
            spectra[iz, ir, iphi, :] = sed.to_value(_SED_UNIT)
        # Append periodic phi=360 endpoint; this is a separate array in memory.
        extended = np.concatenate((spectra, spectra[:, :, :1, :]), axis=2)
        phi = idx["azimuth"].to_value(u.deg)
        if not np.isclose(phi[0], 0) or np.any(phi >= 360):
            raise ValueError("Expected GALPROP azimuth grid in [0, 360) starting at 0")
        interpolator = RegularGridInterpolator(
            (
                idx["height"].to_value(u.kpc),
                idx["radius"].to_value(u.kpc),
                np.concatenate((phi, [360.0])),
            ),
            extended,
            method="linear",
            bounds_error=self.bounds_error,
            fill_value=None,
        )
        self._interpolators[component] = interpolator
        return interpolator

    def spectrum(
        self,
        position: SkyCoord,
        component: str | None = None,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Evaluate the ISRF spectrum at one or more sky positions.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Galactic or celestial position(s) with physical distances.
        component : str, optional
            Spectral component. Defaults to the component set at init.

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Increasing photon-energy grid (Nenergy,).
        sed : `~astropy.units.Quantity`
            S(epsilon) in eV / cm**3. Shape is ``position.shape + (Nenergy,)``.
        """
        chosen = self.component if component is None else component
        radius, azimuth, height = skycoord_to_galprop(position, self.frame)
        radius_values, azimuth_values, height_values = np.broadcast_arrays(
            radius.to_value(u.kpc),
            azimuth.to_value(u.deg),
            height.to_value(u.kpc),
        )
        shape = radius_values.shape
        points = np.column_stack((
            height_values.ravel(),
            radius_values.ravel(),
            azimuth_values.ravel(),
        ))
        if self.bounds_error:
            z_min, z_max = self.index["height"].to_value(u.kpc)[[0, -1]]
            r_min, r_max = self.index["radius"].to_value(u.kpc)[[0, -1]]
            if np.any((points[:, 0] < z_min) | (points[:, 0] > z_max) |
                      (points[:, 1] < r_min) | (points[:, 1] > r_max)):
                raise ValueError(
                    "Position outside GALPROP grid; choose a position within "
                    f"R=[{r_min}, {r_max}] kpc and z=[{z_min}, {z_max}] kpc"
                )
        values = self._get_interpolator(chosen)(points)
        if shape == ():
            values = values[0]
        else:
            values = values.reshape(shape + (self.energy.size,))
        return self.energy, values * _SED_UNIT

    def photon_density(
        self,
        position: SkyCoord,
        component: str | None = None,
    ) -> tuple[u.Quantity, u.Quantity]:
        """Return energy and differential photon density dn/depsilon."""
        energy, sed = self.spectrum(position, component=component)
        return energy, sed_to_photon_density(energy, sed)

    def energy_density(
        self,
        position: SkyCoord,
        component: str | None = None,
    ) -> u.Quantity:
        """Integrate the ISRF spectrum over log photon energy."""
        energy, sed = self.spectrum(position, component=component)
        return integrate_isrf_sed(energy, sed)
    
    def fit_blackbody_components(
        self,
        position: SkyCoord,
        temperatures: dict[str, u.Quantity] | None = None,
        component: str | None = None,
        threshold: float = 0.01,
        x0: tuple[float, ...] | None = None,
        energy_range: tuple[u.Quantity, u.Quantity] | None = None,
        method: str = "nnls",
    ) -> dict:
        """Approximate a GALPROP SED by nonnegative diluted blackbodies.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Scalar position with physical distance.
        temperatures : dict, optional
            Ordered mapping from field names to temperatures. The default
            is FIR=40 K, NIR=500 K, VIS=3500 K, UV=20000 K, as validated
            against the legacy notebook at the reference position.
        component : str, optional
            GALPROP component; defaults to the instance component.
        threshold : float, optional
            Select bins with SED > threshold * max(SED). Default 0.01.
        x0 : sequence of float, optional
            Starting bolometric densities in eV/cm3 for iterative methods.
            When omitted, use a nonnegative linear least-squares solution.
            Not used for ``method='nnls'``.
        energy_range : tuple of `~astropy.units.Quantity`, optional
            Inclusive photon-energy range used for the fit.
        method : {'nnls', 'linear', 'log', 'relative'}, optional
            ``nnls`` solves nonnegative linear least squares directly;
            ``linear``, ``log``, and ``relative`` use bounded nonlinear
            least squares with their corresponding residual definitions.

        Returns
        -------
        fit : dict
            Full-grid component SEDs, summed fit, bolometric blackbody
            normalizations, and common quality metrics. Integrated
            densities are calculated only on the GALPROP energy grid.

        Notes
        -----
        Fixed-temperature blackbody amplitudes are bolometric energy
        densities, not integrals over separate GALPROP energy bands.
        A small integrated difference does not guarantee an accurate
        inverse-Compton spectrum; validate against the tabulated field.
        """
        if not isinstance(position, SkyCoord) or not position.isscalar:
            raise ValueError("position must be a scalar SkyCoord")
        if temperatures is None:
            temperatures = {
                "FIR": 40 * u.K,
                "NIR": 500 * u.K,
                "VIS": 3500 * u.K,
                "UV": 20000 * u.K,
            }
        if not temperatures:
            raise ValueError("temperatures must not be empty")
        if method not in {"nnls", "linear", "log", "relative"}:
            raise ValueError("method must be 'nnls', 'linear', 'log', or 'relative'")
        if not np.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError("threshold must be finite and in [0, 1)")

        labels = list(temperatures)
        temperatures = {
            name: u.Quantity(value).to(u.K)
            for name, value in temperatures.items()
        }
        for name, temp in temperatures.items():
            if not np.isfinite(temp.value) or not temp.isscalar or temp <= 0 * u.K:
                raise ValueError(f"Invalid temperature for {name!r}")

        energy, sed = self.spectrum(position, component=component)
        sed_values = sed.to_value(_SED_UNIT)
        if not np.all(np.isfinite(sed_values)) or np.any(sed_values < 0):
            raise ValueError("GALPROP SED must be finite and nonnegative")
        if not np.any(sed_values > 0):
            raise ValueError("GALPROP SED has no positive bins")

        basis = np.column_stack([
            blackbody_sed(
                energy,
                temperature=temperatures[name],
                energy_density=1 * _SED_UNIT,
            ).to_value(_SED_UNIT)
            for name in labels
        ])
        if not np.all(np.isfinite(basis)):
            raise ValueError("Blackbody basis contains nonfinite values")

        mask = sed_values > threshold * sed_values.max()
        if energy_range is not None:
            if len(energy_range) != 2:
                raise ValueError("energy_range must contain (minimum, maximum)")
            emin, emax = (
                u.Quantity(bound).to_value(u.eV)
                for bound in energy_range
            )
            if not (np.isfinite(emin) and np.isfinite(emax) and 0 < emin < emax):
                raise ValueError("Invalid energy_range")
            e_values = energy.to_value(u.eV)
            mask &= (e_values >= emin) & (e_values <= emax)

        if np.count_nonzero(mask) < len(labels):
            raise ValueError("Insufficient selected bins for blackbody fitting")
        design = basis[mask]
        y = sed_values[mask]
        initial, _ = nnls(design, y)

        if method == "nnls":
            densities = initial
            success, message, nfev = True, "NNLS converged", 0
        else:
            if x0 is None:
                x0_values = initial
            else:
                x0_values = np.asarray(x0, dtype=float)
                if (x0_values.shape != (len(labels),)
                        or not np.all(np.isfinite(x0_values))
                        or np.any(x0_values < 0)):
                    raise ValueError("x0 must contain one finite, nonnegative value per field")
            # Avoid zero-valued starting vectors for log residuals.
            if method == "log":
                x0_values = np.maximum(x0_values, 1e-12)

            def residuals(params):
                model_values = design @ params
                if method == "linear":
                    return model_values - y
                if method == "relative":
                    return (model_values - y) / y
                return (
                    np.log10(np.maximum(model_values, np.finfo(float).tiny))
                    - np.log10(y)
                )

            optimizer = least_squares(
                residuals,
                x0=x0_values,
                bounds=(0, np.inf),
            )
            densities = optimizer.x
            success = bool(optimizer.success)
            message = str(optimizer.message)
            nfev = int(optimizer.nfev)

        fitted_selected = design @ densities
        residual_linear = fitted_selected - y
        residual_relative = residual_linear / y
        residual_log = (
            np.log10(np.maximum(fitted_selected, np.finfo(float).tiny))
            - np.log10(y)
        )
        residual_by_method = {
            "nnls": residual_linear,
            "linear": residual_linear,
            "relative": residual_relative,
            "log": residual_log,
        }
        cost = 0.5 * float(np.sum(residual_by_method[method] ** 2))

        fit_sed = (basis @ densities) * _SED_UNIT
        component_seds = {
            name: (basis[:, index] * densities[index]) * _SED_UNIT
            for index, name in enumerate(labels)
        }
        reference_density = integrate_isrf_sed(energy, sed)
        fitted_density = integrate_isrf_sed(energy, fit_sed)
        relative_difference = (
            fitted_density / reference_density - 1
        ).to_value(u.one)
        bolometric_density = np.sum(densities) * _SED_UNIT

        return {
            "energy": energy,
            "sed": sed,
            "fit_sed": fit_sed,
            "components": component_seds,
            "temperatures": temperatures,
            "energy_densities": {
                name: float(densities[index]) * _SED_UNIT
                for index, name in enumerate(labels)
            },
            "method": method,
            "threshold": threshold,
            "mask": mask,
            "reference_integrated_density": reference_density,
            "fitted_integrated_density": fitted_density,
            "bolometric_density": bolometric_density,
            "rms_log_residual": float(np.sqrt(np.mean(residual_log ** 2))),
            "rms_relative_residual": float(np.sqrt(np.mean(residual_relative ** 2))),
            "relative_integrated_difference": float(relative_difference),
            "success": success,
            "message": message,
            "cost": cost,
            "nfev": nfev,
        }

    def to_naima_blackbodies(
        self,
        position: SkyCoord,
        temperatures: dict[str, u.Quantity] | None = None,
        component: str | None = None,
        threshold: float = 0.01,
        energy_range: tuple[u.Quantity, u.Quantity] | None = None,
        method: str = "nnls",
        x0: tuple[float, ...] | None = None,
        include_cmb: bool = False,
    ) -> list:
        """Return fitted thermal fields in Naima seed-photon-field format.

        Each field is ``[name, temperature, bolometric_energy_density]``.
        Set ``include_cmb=True`` only when a separate CMB field is needed;
        the GALPROP SED handled by this class does not automatically add it.
        """
        fit = self.fit_blackbody_components(
            position,
            temperatures=temperatures,
            component=component,
            threshold=threshold,
            energy_range=energy_range,
            method=method,
            x0=x0,
        )
        if not fit["success"]:
            raise RuntimeError(f"Blackbody fit failed: {fit['message']}")
        fields = [
            [name, temp, fit["energy_densities"][name]]
            for name, temp in fit["temperatures"].items()
        ]
        if include_cmb:
            fields.append("CMB")
        return fields

    def to_naima(
        self,
        position: SkyCoord,
        component: str | None = None,
        name: str | None = None,
    ) -> list:
        """Return a tabulated seed photon field accepted by Naima.

        Returns
        -------
        seed : list
            ``[name, photon_energy, spectral_energy_density]``.

        Notes
        -----
        Naima converts S(epsilon) in eV/cm**3 to dn/depsilon internally.
        Only scalar sky positions are accepted. Add ``'CMB'`` separately
        to ``seed_photon_fields`` when needed.
        """
        if not position.isscalar:
            raise ValueError("to_naima requires a scalar SkyCoord")
        chosen = self.component if component is None else component
        energy, sed = self.spectrum(position, component=chosen)
        label = name if name is not None else f"Porter2017_{self.model}_{chosen}"
        return [label, energy, sed]
