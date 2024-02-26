#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FeuPy TeV Astronomy Python package configutation."""


# In[ ]:


from astropy import units as u


# In[ ]:


INTROSTR = '*** Basic parameters ***\n\n'

CU = 6.1e-17 * u.Unit("TeV-1 cm-2 s-1") 
# CU is the flux of the Crab Nebula at 100 TeV
UNIT_DEG = 'deg' 
# Degree unit
FRAME_ICRS = 'icrs'



COLOR_LHAASO = "red"
MARKER_LHAASO = "o"

COLOR_CTA = "blue"
MARKER_CTA = "s"


IRF_OPTIONS = [
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

           

# # catalogs_tags = ["gamma-cat", "hgps", "2hwc", "3hwc", "3fgl", "4fgl", "2fhl", "3fhl"]



# datasets_sources_gammapy = "counterparts_gammapy"
# datasets_sources_outside_gammapy = "counterparts_outside_gammapy"
# datasets_sources_joint = "counterparts_joint"

# dict_separation = "separation"
# dict_leg_style = "leg_style"
# path_github = "/home/born-again/Documents/GitHub"
# path_my_modules = f"{path_github}/my_modules"

# path_fp = f"{path_github}/flux_points_outside_gammapy_catalogs"

# path_fp_HAWC = f"{path_fp}/HAWC"
# path_fp_LHAASO = f"{path_fp}/LHASSO_publishNature"



# dir_config = "config"
# dir_plot_style = "plot_style"
# dir_utilities = "utilities"
# dir_spectral_models = "spectral_models"
# dir_hawc_analysis = "hawc_analysis"
# dir_hess_analysis = "hess_analysis"
# dir_lhaaso_analysis = "lhaaso_analysis"
# dir_cta_simulation = "cta_simulation"
# dir_gammapy_catalogs = "gammapy_catalogs"
# file_setup_analysis = "setup_analysis"






# format_csv  = '.csv'
# format_fits = '.fits'
# format_dat  = '.dat'
# format_png  = '.png'
# format_pdf  = '.pdf'
# format_svg  = '.svg'
# format_yaml = '.yaml'



# sed_type_e2dnde = 'e2dnde'
# sed_type_dnde   = 'dnde'


# frame_icrc = "icrs" # International Celestial Reference System (ICRS)

# # dir_analysis = "analysis"
# dir_analysis = "analysis"

# dir_flux_points_tables = "flux_points_tables"

# # parent directories names
# dir_flux_points = "flux_points"
# dir_sky_maps = "sky_maps"
# dir_counts = "counts"

# dir_tables = "tables"
# dir_figures = "figures"
# dir_datasets  = "datasets"
# dir_models  = "models"

# dir_catalogs_roi = "catalogs_roi"

# dir_SED = "SED_models"
# dir_SED_from_catalogs = "SED_from_catalogs"




# irf_z = [20,40,60]
# irf_h = [0.5, 5, 50]
# irf_loc = [("cta_north", "North"),("cta_south", "South")]

