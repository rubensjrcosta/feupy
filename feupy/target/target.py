#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Target class."""


# In[27]:


from astropy import units as u
from astropy.units import Quantity

from astropy.coordinates import SkyCoord

from gammapy.modeling.models import (
    SkyModel,
    SpectralModel, 
    SpatialModel, 
    TemporalModel
)

from feupy.utils.string_handling import name_to_txt

from feupy.utils.coordinates import skcoord_to_dict
from gammapy.modeling.models import Models


# In[7]:


__all__ = [
    "Target",
]


# In[ ]:





# In[8]:


class Target:
    """Observation target information.
    
    Parameters
    ----------
    name : `str`
        Name of the source
    pos_ra : `~astropy.units.Quantity`
        Right ascension (J2000) (degrees) of the source position
    pos_dec : `~astropy.units.Quantity`
        Declination (J2000) (degrees) of the source position
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral Model of the source
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial Model of the source
    temporal_model : `~gammapy.modeling.models.TemporalModel`
        Temporal Model of the source
    """
    
    all = []
    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg)
    def __init__(
        self, 
        name: str, 
        pos_ra, 
        pos_dec,
        model: SkyModel= None,
    ):

        # Run validations to the received arguments
        assert  0 <= pos_ra.value <= 360, f"Right Ascension {pos_ra} is not in the range: (0,360) deg!"
        assert -90 <= pos_dec.value <= 90, f"Declination {pos_dec} is not in the range: (-90,90) deg!"

        # Assign to self object
        self.__name = name 
        self.position = SkyCoord(Quantity(pos_ra, "deg"), Quantity(pos_dec, "deg"))
#         self.position_dict = skcoord_to_dict(self.position)
        self.model = model
        self.dict = self._to_dict()
            
    
    def _to_dict(self):
        _dict = {"name": self.name,
                "position": skcoord_to_dict(self.position) }
        if self.model:
            _dict["model"] = self.model.to_dict()
        return _dict
    
    @property
    def name(self):
        return self.__name
        
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if model:
            self.spectral_model = model.spectral_model
            self.spatial_model = model.spatial_model
            self.temporal_model = model.temporal_model
            
    @property
    def info(self):
        """Target report (`str`)."""
        ss = '*** Basic parameters ***\n\n'
        if self.name is not None:
            ss += 'name={}\n'.format(self.name)
        else: ss += 'name=None\n'
        if self.position is not None: 
            ss += "pos_ra={:.2f}\n".format(self.position.ra).replace(' ', '')
            ss += "pos_dec={:.2f}\n".format(self.position.dec).replace(' ', '')
        else: ss += "position=None\n"
        if self.model:
            ss += "\n*** Model information ***\n\n"
            ss += str(self.model)
        return ss.replace('=', ' = ')

    
    
    def __repr__(self):
        ss = f'{self.__class__.__name__}('
        ss += f'name={self.name!r}, '
        ss += f"position={self.position!r}, "
        ss += f"model={self.model!r})"
        return ss
        


# In[9]:


# from feupy.catalog.pulsar.atnf import SourceCatalogATNF

# catalog = SourceCatalogATNF()
# source = catalog['PSR J1826-1256']
# name = source.name
# pos_ra = source.position.ra
# pos_dec = source.position.dec
# from gammapy.modeling.models import SkyModel
# from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel
# model = SkyModel(spectral_model=ExpCutoffPowerLawSpectralModel(), name=name)
# print(model)
# target = Target(name, pos_ra, pos_dec, model)
# target.model

