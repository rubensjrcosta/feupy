#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.units import Quantity

def table_row_to_dict(row, make_quantity=True):
    """Make one source data dictionary.

    Parameters
    ----------
    row : `~astropy.table.Row`
        Row.
    make_quantity : bool, optional
        Make quantity values for columns with units.
        Default is True.

    Returns
    -------
    data : dict
        Row data.
    """
    
    data = {}
    for name, col in row.columns.items():
        val = row[name]

        if make_quantity and col.unit:
            val = Quantity(val, unit=col.unit)
        data[name] = val
    return data

def _skycoord_from_table(table):
    keys = table.colnames

    if {"RAJ2000", "DEJ2000"}.issubset(keys):
        lon, lat, frame = "RAJ2000", "DEJ2000", "icrs"
    elif {"RA", "DEC"}.issubset(keys):
        lon, lat, frame = "RA", "DEC", "icrs"
    elif {"ra", "dec"}.issubset(keys):
        lon, lat, frame = "ra", "dec", "icrs"
    else:
        raise KeyError("No column GLON / GLAT or RA / DEC or RAJ2000 / DEJ2000 found.")

    unit = table[lon].unit.to_string() if table[lon].unit else "deg"

    return SkyCoord(table[lon], table[lat], unit=unit, frame=frame)

# In[ ]:




