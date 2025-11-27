# Licensed under a 3-clause BSD style license - see LICENSE.rst
""" CTAO IRFs class."""

import pandas as pd

import numpy as np

import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

from gammapy.irf import load_irf_dict_from_file
from gammapy.data import observatory_locations
from gammapy.data import observatory_locations

from datetime import datetime, timedelta


__all__ = ["Irfs"]

# Define options for IRFs
_IRFS_OPTIONS = [
    ['South', 'AverageAz', '20deg', '0.5h'],
     ['South', 'AverageAz', '20deg', '5h'],
     ['South', 'AverageAz', '20deg', '50h'],
     ['South', 'NorthAz', '20deg', '0.5h'],
     ['South', 'NorthAz', '20deg', '5h'],
     ['South', 'NorthAz', '20deg', '50h'],
     ['South', 'SouthAz', '20deg', '0.5h'],
     ['South', 'SouthAz', '20deg', '5h'],
     ['South', 'SouthAz', '20deg', '50h'],
     ['South', 'AverageAz', '40deg', '0.5h'],
     ['South', 'AverageAz', '40deg', '5h'],
     ['South', 'AverageAz', '40deg', '50h'],
     ['South', 'NorthAz', '40deg', '0.5h'],
     ['South', 'NorthAz', '40deg', '5h'],
     ['South', 'NorthAz', '40deg', '50h'],
     ['South', 'SouthAz', '40deg', '0.5h'],
     ['South', 'SouthAz', '40deg', '5h'],
     ['South', 'SouthAz', '40deg', '50h'],
     ['South', 'AverageAz', '60deg', '0.5h'],
     ['South', 'AverageAz', '60deg', '5h'],
     ['South', 'AverageAz', '60deg', '50h'],
     ['South', 'NorthAz', '60deg', '0.5h'],
     ['South', 'NorthAz', '60deg', '5h'],
     ['South', 'NorthAz', '60deg', '50h'],
     ['South', 'SouthAz', '60deg', '0.5h'],
     ['South', 'SouthAz', '60deg', '5h'],
     ['South', 'SouthAz', '60deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '20deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '20deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '20deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '40deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '40deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '40deg', '50h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '5h'],
     ['South-SSTSubArray', 'AverageAz', '60deg', '50h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '5h'],
     ['South-SSTSubArray', 'NorthAz', '60deg', '50h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '5h'],
     ['South-SSTSubArray', 'SouthAz', '60deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '20deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '20deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '20deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '40deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '40deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '40deg', '50h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '5h'],
     ['South-MSTSubArray', 'AverageAz', '60deg', '50h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '5h'],
     ['South-MSTSubArray', 'NorthAz', '60deg', '50h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '5h'],
     ['South-MSTSubArray', 'SouthAz', '60deg', '50h'],
     ['North', 'AverageAz', '20deg', '0.5h'],
     ['North', 'AverageAz', '20deg', '5h'],
     ['North', 'AverageAz', '20deg', '50h'],
     ['North', 'NorthAz', '20deg', '0.5h'],
     ['North', 'NorthAz', '20deg', '5h'],
     ['North', 'NorthAz', '20deg', '50h'],
     ['North', 'SouthAz', '20deg', '0.5h'],
     ['North', 'SouthAz', '20deg', '5h'],
     ['North', 'SouthAz', '20deg', '50h'],
     ['North', 'AverageAz', '40deg', '0.5h'],
     ['North', 'AverageAz', '40deg', '5h'],
     ['North', 'AverageAz', '40deg', '50h'],
     ['North', 'NorthAz', '40deg', '0.5h'],
     ['North', 'NorthAz', '40deg', '5h'],
     ['North', 'NorthAz', '40deg', '50h'],
     ['North', 'SouthAz', '40deg', '0.5h'],
     ['North', 'SouthAz', '40deg', '5h'],
     ['North', 'SouthAz', '40deg', '50h'],
     ['North', 'AverageAz', '60deg', '0.5h'],
     ['North', 'AverageAz', '60deg', '5h'],
     ['North', 'AverageAz', '60deg', '50h'],
     ['North', 'NorthAz', '60deg', '0.5h'],
     ['North', 'NorthAz', '60deg', '5h'],
     ['North', 'NorthAz', '60deg', '50h'],
     ['North', 'SouthAz', '60deg', '0.5h'],
     ['North', 'SouthAz', '60deg', '5h'],
     ['North', 'SouthAz', '60deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '20deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '20deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '20deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '40deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '40deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '40deg', '50h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '5h'],
     ['North-MSTSubArray', 'AverageAz', '60deg', '50h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '5h'],
     ['North-MSTSubArray', 'NorthAz', '60deg', '50h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '5h'],
     ['North-MSTSubArray', 'SouthAz', '60deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '20deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '20deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '20deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '40deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '40deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '40deg', '50h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '5h'],
     ['North-LSTSubArray', 'AverageAz', '60deg', '50h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '5h'],
     ['North-LSTSubArray', 'NorthAz', '60deg', '50h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '0.5h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '5h'],
     ['North-LSTSubArray', 'SouthAz', '60deg', '50h']
]

class Irfs:
    """Class to handle Instrument Response Functions (IRFs) for CTAO."""
    
    IRFS_OPTIONS = _IRFS_OPTIONS
    IRF_VERSION = "prod5 v0.1"
    
    _SITE_ARRAY = {
        'South': '14MSTs37SSTs', 
        'South-SSTSubArray': '37SSTs', 
        'South-MSTSubArray': '14MSTs', 
        'North': '4LSTs09MSTs', 
        'North-MSTSubArray': '09MSTs',
        'North-LSTSubArray': '4LSTs'
    }
    _OBS_TIME = {'0.5h': '1800s', '5h': '18000s', '50h': '180000s'}
    
    _DIR_FITS = '$PYTHONPATH/data/irfs/cta-prod5-zenodo-v0.1/fits/'

    def __init__(self):
        self.irfs = None
        self.irfs_label = None
        self.obs_loc = None

    @classmethod
    def get_irfs(cls, irfs_opt):
        """Load IRF file based on specified options."""
        dir_fits = f'CTA-Performance-prod5-v0.1-{irfs_opt[0]}-{irfs_opt[2]}.FITS/'
        isite = irfs_opt[0].split('-')[0]
        irfs_file_name = (
            f'Prod5-{isite}-{irfs_opt[2]}-{irfs_opt[1]}-'
            f'{cls._SITE_ARRAY[irfs_opt[0]]}.{cls._OBS_TIME[irfs_opt[3]]}-v0.1.fits.gz'
        )
        file_path = f'{cls._DIR_FITS}{dir_fits}{irfs_file_name}'
        cls.irfs = load_irf_dict_from_file(file_path)
        cls.irfs_label = cls.get_irfs_label(irfs_opt)
        cls.obs_loc = cls.get_obs_loc(irfs_opt)
        return cls.irfs

    @staticmethod
    def get_irfs_label(required_irfs, which='both'):
        """Generate an IRF label based on options."""
        ss = 'CTAO '
        array_label = required_irfs[0].replace('SubArray', 's')
        azimuth_label = required_irfs[1].replace('AverageAz', '')
        extra = ''
        if which == 'zenith':
            extra = f' ({required_irfs[2]})'
        elif which == 'livetime':
            extra = f' ({required_irfs[3]})'
        elif which == 'both':
            extra = f' ({required_irfs[2]}-{required_irfs[3]})'
        return f"{ss}{array_label}{azimuth_label}{extra}"
    
    @staticmethod
    def get_obs_loc(irfs_opt):
        """Get observatory location based on site."""
        return observatory_locations['cta_south'] if 'South' in irfs_opt[0] else observatory_locations['cta_north']

    
    @staticmethod
    def get_irf_groups(irfs_opts):
        """Generate IRF groups with options, labels, and locations."""
        irfs_groups, irfs, irfs_labels, obs_locations = [], [], [], []
        
        arrays = irfs_opts[0] if isinstance(irfs_opts[0], list) else [irfs_opts[0]]
        azimuths = irfs_opts[1] if isinstance(irfs_opts[1], list) else [irfs_opts[1]]
        zeniths = irfs_opts[2] if isinstance(irfs_opts[2], list) else [irfs_opts[2]]
        obs_times = irfs_opts[3] if isinstance(irfs_opts[3], list) else [irfs_opts[3]]

        for array in arrays:
            for azimuth in azimuths:
                for zenith in zeniths:
                    for obs_time in obs_times:
                        irfs_opt = [array, azimuth, zenith, obs_time]
                        irfs_groups.append(irfs_opt)
                        irfs.append(Irfs.get_irfs(irfs_opt))
                        irfs_labels.append(Irfs.get_irfs_label(irfs_opt))
                        obs_locations.append(Irfs.get_obs_loc(irfs_opt))
                        
        return irfs_groups, irfs, irfs_labels, obs_locations
    
    
    @staticmethod
    def get_irfs_array(required_irfs):
        ss = 'CTAO '
        ss_0 = required_irfs[0].replace('SubArray', 's')
        ss_1 = required_irfs[1].replace('AverageAz', '')
        return  ss + ss_0 + ss_1 
    
    @staticmethod
    def get_irfs_name(required_irfs, which='both'):
        ss = 'CTAO-'
        ss_0 = required_irfs[0].replace('SubArray', 's')
        ss_1 = required_irfs[1].replace('AverageAz', '')
        _ss = ''
        if which == 'zenith':
            _ss = required_irfs[2]
        elif which == 'livetime':
            _ss = required_irfs[3]
        elif which =='both': 
            _ss = required_irfs[2] + '_' + required_irfs[3]
        return  ss + ss_0 + ss_1  + '_' + _ss
    


def calculate_annual_visibility_for_zenith_ranges(
        observatory_name, source_position, year=2025, time_step=30):
    """
    Compute annual visibility (in hours) of a source for predefined zenith-angle
    ranges at a given observatory.

    Parameters
    ----------
    observatory_name : str
        Observatory key as defined in `observatory_locations`.
    source_position : SkyCoord
        Celestial coordinates of the source.
    year : int, optional
        Year for which visibility is calculated (default: 2025).
    time_step : int, optional
        Time step in minutes for sampling visibility during the night.

    Returns
    -------
    dict
        Total visibility hours for each zenith-angle range.
    """

    location = observatory_locations[observatory_name]

    # Zenith-angle bins (degrees)
    zenith_ranges = {'20': (10, 30), '40': (30, 50), '60': (50, 70)}
    annual_visibility = {k: 0 for k in zenith_ranges}

    # Night-time sampling: 18:00 → 06:00
    time_points = [
        f"{h:02d}:{m:02d}:00"
        for h in list(range(18, 24)) + list(range(0, 6))
        for m in range(0, 60, time_step)
    ]

    # Loop over every day of the year
    current_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)
    one_day = timedelta(days=1)

    while current_date < end_date:
        date = current_date.strftime("%Y-%m-%d")
        times = Time([f"{date} {t}" for t in time_points])

        altaz = AltAz(obstime=times, location=location)
        source_altaz = source_position.transform_to(altaz)

        zenith_angles = 90 * u.deg - source_altaz.alt

        # Count time spent inside each zenith bin
        for label, (zmin, zmax) in zenith_ranges.items():
            mask = (zenith_angles >= zmin * u.deg) & (zenith_angles < zmax * u.deg)
            annual_visibility[label] += np.sum(mask) * (time_step / 60)

        current_date += one_day

    return annual_visibility


def get_visibility_table_from_position(source_position, year, save_path=None):
    """
    Compute annual visibility for a source at CTAO South and North, returning
    the results as a pandas DataFrame and optionally saving them as CSV or LaTeX.

    Parameters
    ----------
    source_position : SkyCoord
        ICRS coordinates of the target source.
    year : int
        Year for which visibility is computed.
    save_path : str, optional
        Output file path (.csv or .tex). If None, no file is saved.

    Returns
    -------
    pandas.DataFrame
        Visibility duration per zenith-angle bin and observatory.
    """

    observatories = ['cta_south', 'cta_north']
    rows = []

    print("Annual Visibility Durations")

    for obs in observatories:
        vis = calculate_annual_visibility_for_zenith_ranges(
            observatory_name=obs,
            source_position=source_position,
            year=year
        )

        print(f"\n{obs}:")
        for zbin, duration in vis.items():
            print(f"Zenith Angle {zbin}°: {duration:.2f} hours")
            rows.append({
                "Observatory": obs,
                "Zenith Range (deg)": zbin,
                "Visibility (hours)": duration
            })

    df = pd.DataFrame(rows)

    # Publication-friendly names
    df["Observatory"] = df["Observatory"].replace({
        "cta_south": "CTAO South",
        "cta_north": "CTAO North"
    })

    # Save outputs
    if save_path:
        if save_path.endswith(".csv"):
            df.to_csv(save_path, index=False)
            print(f"\nTable saved in: {save_path}")

        elif save_path.endswith(".tex"):
            latex = df.to_latex(
                index=False,
                float_format=lambda x: f"{x:.2f}",
                caption="Annual visibility for each zenith range and CTAO site.",
                label="tab:annual_visibility",
                escape=True
            )
            with open(save_path, "w") as f:
                f.write(latex)
            print(f"\nLaTeX table saved in: {save_path}")

        else:
            print("[Warning] Unsupported file format. Use .csv or .tex.")

    return df

