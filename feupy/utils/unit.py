#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rom astropy import units as u
import numpy as np
from math import floor
def energy_unit_format(E):
    """Format energy quantities to a string representation that is more comfortable to read.

    This is done by switching to the most relevant unit (keV, MeV, GeV, TeV) and changing the float precision.

    Parameters
    ----------
    E: `~astropy.units.Quantity`
        Quantity or list of quantities.

    Returns
    -------
    str : str
        A string or tuple of strings with energy unit formatted.
    """
    try:
        iter(E)
    except TypeError:
        pass
    else:
        return tuple(map(energy_unit_format, E))

    i = floor(np.log10(E.to_value(u.eV)) / 3)  # a new unit every 3 decades
    unit = (u.eV, u.keV, u.MeV, u.GeV, u.TeV, u.PeV)[i] if i < 5 else u.PeV

    v = E.to_value(unit)
    i = floor(np.log10(v))
    prec = (2, 1, 0)[i] if i < 3 else 0

    return f"{v:0.{prec}f} {unit}"

