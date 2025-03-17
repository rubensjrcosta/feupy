# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog utilities classes."""
import numpy as np

# from astropy.units import Quantity
# from astropy import units as u
# from astropy.table import Table, vstack, Column

# from gammapy.modeling.models import Models
# from gammapy.estimators import FluxPoints
# from gammapy.datasets import Datasets, FluxPointsDataset
# from gammapy.catalog.gammacat import SourceCatalogObjectGammaCat
# from gammapy.catalog.hess import SourceCatalogObjectHGPS
# from feupy.catalog import SOURCE_REGISTRY, CATALOG_REGISTRY
# from feupy.utils.units import Hz_to_eV, Jy_to_erg_by_cm2_s
# from feupy.utils.scripts import unpickling

import logging
from typing import List, Optional
# from feupy.catalog import SOURCE_REGISTRY, CATALOG_REGISTRY
from feupy.catalog import FEUPY_CATALOG_REGISTRY

log = logging.getLogger(__name__)

catalog_2fhl = FEUPY_CATALOG_REGISTRY.get_cls('2fhl')()
catalog_3fhl = FEUPY_CATALOG_REGISTRY.get_cls('3fhl')()

catalog_3fgl = FEUPY_CATALOG_REGISTRY.get_cls('3fgl')()
catalog_4fgl = FEUPY_CATALOG_REGISTRY.get_cls('4fgl')()

catalog_2hwc = FEUPY_CATALOG_REGISTRY.get_cls('2hwc')()
catalog_3hwc = FEUPY_CATALOG_REGISTRY.get_cls('3hwc')()
catalog_ehwc = FEUPY_CATALOG_REGISTRY.get_cls('ehwc')()
catalog_extra_hawc = FEUPY_CATALOG_REGISTRY.get_cls('hwc-2021ApJ')()

catalog_hgps = FEUPY_CATALOG_REGISTRY.get_cls('hgps')()
catalog_extra_hess = FEUPY_CATALOG_REGISTRY.get_cls('hess-2019A&A')()

catalog_gamma_cat = FEUPY_CATALOG_REGISTRY.get_cls('gamma-cat')()

catalog_vtscat = FEUPY_CATALOG_REGISTRY.get_cls('vtscat')()

catalog_veritas = FEUPY_CATALOG_REGISTRY.get_cls('veritas-2018ApJ')()

catalog_lhaaso = FEUPY_CATALOG_REGISTRY.get_cls('LHAASO')()
catalog_1lhaaso = FEUPY_CATALOG_REGISTRY.get_cls('1LHAASO')()
catalog_extra_lhaaso = FEUPY_CATALOG_REGISTRY.get_cls('LHAASO-2024icrc')()

catalog_psrcat = FEUPY_CATALOG_REGISTRY.get_cls('psrcat')()

# CATALOGS_MARKERS = {
#     'gamma-cat': 'd',
#     'hgps': '8',
#     '2hwc': '^',
#     '3fgl': '<',
#     '4fgl': '>',
#     '2fhl': 'v',
#     '3fhl': 'h',
#     '3hwc': 'p',
#     'veritas': 'P',
#     'extraHAWC': '8',
#     'publish-nature-lhaaso': 'D',
#     '1LHAASO': 's',
#     'ATNF': '*',
#     'CTAO': 'o',
# }

# SOURCE_CATALOGS_MARKERS = {
#     'gamma-cat': 'd',
#     'HESS': '8',
#     '2HWC': '^',
#     '3FGL': '<',
#     '4FGL': '>',
#     '2FHL': 'v',
#     '3HL': 'h',
#     '3HWC': 'p',
#     'VER': 'P',
#     'eHAWC': '8',
#     'HAWC': '8',
#     'LHAASO': 'D',
#     '1LHAASO': 's',
#     'ATNF': '*',
#     'CTAO': 'o'
# }





def load_catalogs(catalogs: Optional[List] = FEUPY_CATALOG_REGISTRY) -> List:
    """Load a list of catalogs from the provided registry.

    Parameters:
    -----------
    catalogs : list, optional
        A list of catalog definitions from the registry. 
        Defaults to FEUPY_CATALOG_REGISTRY if not provided.

    Returns:
    --------
    List:
        A list of catalog class instances.

    Raises:
    -------
    ValueError: If the catalog class cannot be loaded properly.
    """
    source_catalogs = []
    
    for index, catalog in enumerate(catalogs):
        try:
            catalog_cls = catalogs.get_cls(catalog.tag)()
            source_catalogs.append(catalog_cls)
            # log.info(f"Successfully loaded catalog '{catalog.tag}' at index {index}.")
        except Exception as e:
            log.error(f"Failed to load catalog '{catalog.tag}' at index {index}: {e}")
            raise ValueError(f"Error loading catalog '{catalog.tag}' at index {index}: {e}")
    
    log.info(f"Loaded {len(source_catalogs)} catalogs.")
    return source_catalogs




# def set_gammacat_hess_name(source, which='dataset'):

#     if isinstance(source, SourceCatalogObjectGammaCat):
#         cat_tag = 'gamma-cat'
#     elif isinstance(source, SourceCatalogObjectHGPS):
#         cat_tag = 'hgps'
#     else:
#         raise ValueError('Source needs to be SourceCatalogObjectGammaCat or SourceCatalogObjectHGPS')
#     if which == 'dataset' or which=='source':    
#         name = f'{source.name} ({cat_tag})'
#     elif which == 'model':
#         name = f'{source.name} ({cat_tag}-{source.spectral_model().tag[1]})'
#     else:
#         raise ValueError('which needs to be dataset or model')
#     return name

# def load_catalog(tag=None):
#     """
#     """
#     return  CATALOG_REGISTRY.get_cls(tag)()


# def _get_catalogs(catalogs_registry = None):
#     source_catalogs = []
#     for index, catalog in enumerate(catalogs_registry):
#         catalog_cls = catalogs_registry.get_cls(catalog.tag)()
#         source_catalogs.append(catalog_cls)
#     return source_catalogs
    

# def load_catalogs(cats="all"):
#     """load catalogs.

#     Parameters
#     ----------
#     cats : {'all', 'gamma', 'gammapy', 'feupy'}
#         list of possible options
#     """
#     if cats == "all":
#         return _get_catalogs(catalogs_registry = CATALOG_REGISTRY)
#     elif cats == "gammapy":
#         return _get_catalogs(catalogs_registry = CATALOG_REGISTRY_GAMMAPY)
#     elif cats == "feupy":
#         return _get_catalogs(catalogs_registry = CATALOG_REGISTRY_FEUPY)
#     elif cats == "gamma":
#         source_catalogs =  _get_catalogs(catalogs_registry = CATALOG_REGISTRY)
#         source_catalogs.remove(source_catalogs[-1])
#         return source_catalogs

# # In[14]:


# def catalogs_info():
#     """
#     """
#     print (f"Source catalogs in Gammapy: {len(CATALOG_REGISTRY_GAMMAPY)}\n")
#     for gindex, catalog in enumerate(CATALOG_REGISTRY_GAMMAPY):
#         print(f"(catalog index: {gindex}) {CATALOG_REGISTRY_GAMMAPY.get_cls(catalog.tag)()}")
#     print (f"Source catalogs in Feupy: {len(CATALOG_REGISTRY_FEUPY)}\n")
#     for index, catalog in enumerate(CATALOG_REGISTRY_FEUPY):
#         print(f"(catalog index: {index+gindex+1}) {CATALOG_REGISTRY_FEUPY.get_cls(catalog.tag)()}")


