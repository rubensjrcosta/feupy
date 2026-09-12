# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""CTAO visibility estimation utilities."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import (
    AltAz,
    SkyCoord,
    get_body,
    get_sun,
    solar_system_ephemeris,
)
from astropy.time import Time
from gammapy.data import observatory_locations

try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, **kwargs):
        return iterable


__all__ = [
    "CTAOVisibilityEstimator",
    "make_ctao_visibility_table",
]

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Core estimator (ONLY physics)
# -----------------------------------------------------------------------------


class CTAOVisibilityEstimator:
    """
    Compute annual visibility for CTAO observatories.

    Includes:
    - Astronomical night (Sun altitude constraint)
    - Moon avoidance
    - Optional airmass weighting
    """

    ZENITH_BINS = {
        "20": (10, 30),
        "40": (30, 50),
        "60": (50, 70),
    }

    def __init__(
        self,
        target: SkyCoord,
        year: int = 2025,
        time_step_min: int = 30,
        sun_alt_limit: float = -18,
        moon_sep_limit: float = 30,
        use_airmass_weight: bool = False,
    ):
        self.target = target
        self.year = year
        self.dt = time_step_min

        self.sun_limit = sun_alt_limit * u.deg
        self.moon_limit = moon_sep_limit * u.deg
        self.use_airmass_weight = use_airmass_weight

        # Night time grid (18h → 6h)
        self._time_grid = [
            f"{h:02d}:{m:02d}:00"
            for h in list(range(18, 24)) + list(range(0, 6))
            for m in range(0, 60, self.dt)
        ]

    # ------------------------------------------------------------------
    # Time selection (Sun + Moon)
    # ------------------------------------------------------------------

    def _get_observable_times(self, date: datetime, location) -> Time:
        """Return valid observing times for a given date."""

        date_str = date.strftime("%Y-%m-%d")
        times = Time([f"{date_str} {t}" for t in self._time_grid])

        altaz = AltAz(obstime=times, location=location)

        # Sun constraint (astronomical night)
        sun_alt = get_sun(times).transform_to(altaz).alt
        night_mask = sun_alt < self.sun_limit

        # Moon constraint
        with solar_system_ephemeris.set("builtin"):
            moon = get_body("moon", times).transform_to(altaz)

        target_altaz = self.target.transform_to(altaz)

        moon_mask = ~(
            (moon.alt > 0 * u.deg) & (moon.separation(target_altaz) < self.moon_limit)
        )

        return times[night_mask & moon_mask]

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_visibility(
        self,
        observatory: str,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """
        Compute annual visibility per zenith bin.

        Parameters
        ----------
        observatory : str
            "cta_south" or "cta_north"
        show_progress : bool

        Returns
        -------
        dict
            Visibility (hours) per zenith bin
        """
        location = observatory_locations[observatory]
        visibility = {k: 0.0 for k in self.ZENITH_BINS}

        step_hours = self.dt / 60.0

        start = datetime(self.year, 1, 1)
        end = datetime(self.year + 1, 1, 1)
        n_days = (end - start).days

        iterator = (
            tqdm(range(n_days), desc=f"{observatory}")
            if show_progress
            else range(n_days)
        )

        for d in iterator:
            date = start + timedelta(days=d)

            times = self._get_observable_times(date, location)

            if len(times) == 0:
                continue

            altaz = AltAz(obstime=times, location=location)
            target_altaz = self.target.transform_to(altaz)

            zenith = 90 * u.deg - target_altaz.alt

            # Airmass weighting (optional)
            if self.use_airmass_weight:
                airmass = 1 / np.cos(zenith.to(u.rad))
                weights = 1 / airmass
            else:
                weights = np.ones_like(zenith.value)

            for label, (zmin, zmax) in self.ZENITH_BINS.items():
                mask = (zenith >= zmin * u.deg) & (zenith < zmax * u.deg)
                visibility[label] += np.sum(weights[mask]) * step_hours

        return visibility


# -----------------------------------------------------------------------------
# Table utilities (separate responsibility)
# -----------------------------------------------------------------------------


def make_ctao_visibility_table(
    estimator: CTAOVisibilityEstimator,
    save_path: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Create a visibility table for CTAO North and South.

    Parameters
    ----------
    estimator : CTAOVisibilityEstimator
    save_path : str, optional
    show_progress : bool

    Returns
    -------
    pandas.DataFrame
    """
    rows = []

    for obs in ["cta_south", "cta_north"]:
        vis = estimator.compute_visibility(obs, show_progress)

        for zbin, hours in vis.items():
            rows.append(
                {
                    "Observatory": "CTAO South" if obs == "cta_south" else "CTAO North",
                    "Zenith (deg)": int(zbin),
                    "Visibility (hours)": hours,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Observatory", "Zenith (deg)"])

    # Optional save
    if save_path:
        path = Path(save_path)

        if path.suffix == ".csv":
            df.to_csv(path, index=False)

        elif path.suffix == ".tex":
            path.write_text(
                df.to_latex(
                    index=False,
                    float_format=lambda x: f"{x:.2f}",
                    caption="Annual CTAO visibility per zenith bin.",
                    label="tab:ctao_visibility",
                )
            )

    return df
