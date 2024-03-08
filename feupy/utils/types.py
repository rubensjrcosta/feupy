#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for types validation."""


# In[2]:


from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity

from feupy.cta import Irfs


# In[4]:


__all__ = [
    "AngleType",
    "EnergyType",
    "QuantityType",
    "TimeType",
    "IrfType",
]


# In[6]:


class AngleType(Angle):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Angle(v)


class EnergyType(Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v = Quantity(v)
        if v.unit.physical_type != "energy":
            raise ValueError(f"Invalid unit for energy: {v.unit!r}")
        return v

class QuantityType(Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Quantity(v)
        
            
class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)

class IrfType(Irfs):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not v in Irfs.IRFS_OPTIONS:
            ss = f"The value of the IRF option is invalid: {v!r}\n "
            ss += f"Select one value from the following list: {Irfs.IRFS_OPTIONS!r}"
            raise ValueError(ss)
        return v


# In[ ]:




