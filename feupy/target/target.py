#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Target class."""


# In[1]:


from astropy import units as u
from astropy.units import Quantity

from astropy.coordinates import SkyCoord

from gammapy.modeling.models import (
    SkyModel,
    SpectralModel, 
    SpatialModel, 
    TemporalModel
)

from feupy.utils.coordinates import skcoord_to_dict

from feupy.utils.string_handling import name_to_txt


# In[3]:


__all__ = [
    "Target",
]


# In[2]:


class Target:
    """Target object.

    This class can be used directly or used as a base class for 
    the ROI class.
    
    ROI class represented by `~feupy.roi.ROI`
    
    Attributes
    ----------
    name : str
        Target name 
    pos_ra : `~astropy.units.Quantity`
        Right ascension (J2000) (degrees) of the target position
    pos_dec : `~astropy.units.Quantity`
        Declination (J2000) (degrees) of the target position
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model of the target
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial model of the target
    temporal_model : `~gammapy.modeling.models.TemporalModel`
        Temporal model of the target
        
    Methods
    -------
    
    """
    all = []
    # Validating the units of arguments to functions
    @u.quantity_input(pos_ra=u.deg, pos_dec=u.deg)
    def __init__(
        self, 
        name: str, 
        pos_ra, 
        pos_dec,
        spectral_model: SpectralModel=None,
        spatial_model: SpatialModel=None,
        temporal_model: TemporalModel=None,
    ):

        # Assign to self object
        self.__name = name 
        self.position = SkyCoord(Quantity(pos_ra, "deg"), Quantity(pos_dec, "deg"))
        self.spectral_model = spectral_model
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.sky_model = self._sky_model()
        self.dict = self._to_dict()
        
        Target.all.append(self)
        
    @property
    def name(self):
        return self.__name
                   
    @property
    def spectral_model(self):
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, spectral_model):
        self._spectral_model = spectral_model 
        
    @property
    def spatial_model(self):
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, spatial_model):
        self._spatial_model = spatial_model
        
    @property
    def temporal_model(self):
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, temporal_model):
        self._temporal_model = temporal_model
    
    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectrum'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,spectrum"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "spectrum" in ops:
            ss += self._info_spectrum()

        return ss

    def _info_basic(self):
        """Print basic info."""
        return (
            f"\n*** Basic info ***\n\n"
            f"Source name : {self.name}\n"
        )

    def _info_position(self):
        """Print position info."""
        return (
            f"\n*** Position info ***\n\n"
            f"RA: {self.position.ra:.3f}\n"
            f"DEC: {self.position.dec:.3f}\n"
        )

    def _info_spectrum(self):
        """Print spectral info."""
        ss = "\n*** Spectral info ***\n\n"
        if self.spectral_model is not None:
            model = self.spectral_model
            parameters = model.parameters
            ss += f"Spectrum type:  {model.tag[0]}\n"
            for par in parameters:
                name = par.name
                val = par.value
                err = par.error
                
                try:
                    unit = f"{par.unit:unicode}"
                except: unit = ""

                ss += f"{name}: {val:.3} +- {err} {unit}\n"
    
        else:
            ss += "No spectrum available"

        return ss
    
    def _sky_model(self):
        if self.spectral_model is not None:
            tag = self.spectral_model.tag[1]
            return SkyModel(
                    name=f"{name_to_txt(self.name)} - {tag}",
                    spectral_model=self.spectral_model,
                    spatial_model=self.spatial_model, 
                    temporal_model=self.temporal_model
                )
        else: return None
        
    def _to_dict(self):
        _dict = {"name": self.name,
                "position": skcoord_to_dict(self.position) }
        if self.sky_model is not None:
            _dict["model"] = self.sky_model.to_dict()
        return _dict
    
    def __repr__(self):
        ss = f"{self.__class__.__name__}("
        ss += f"name={self.name!r}, "
        ss += "pos_ra=Quantity('{:.2f}'), ".format(self.position.ra).replace(' ', '')
        ss += "pos_dec=Quantity('{:.2f}'))\n".format(self.position.dec).replace(' ', '')
        return ss 


# In[ ]:


def config(self, value):
    if isinstance(value, dict):
        self._config = AnalysisConfig(**value)
    elif isinstance(value, AnalysisConfig):
        self._config = value
    else:
        raise TypeError("config must be dict or AnalysisConfig.")


# In[5]:


def test_target():
    return Target(
        name="23HWC J1825-134", 
        pos_ra=27.46* u.Unit('deg'), 
        pos_dec=12.2* u.Unit('deg'),
    
    )


# In[7]:





# In[10]:





# In[ ]:




