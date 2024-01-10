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


# In[ ]:


from gammapy.utils.scripts import make_path
def append_nones(length, list_):
    """
    Appends Nones to list to get length of list equal to `length`.
    If list is too long raise AttributeError
    """
    diff_len = length - len(list_)
    if diff_len < 0:
        raise AttributeError('Length error list is too long.')
    return list_ + [None] * diff_len


# In[ ]:


import numpy as np

def remove_nan(mytable):
    """Remove lines containing a nan"""
    has_nan = np.zeros(len(mytable), dtype=bool)
    for col in mytable.itercols():
        if col.info.dtype.kind == 'f':
            has_nan |= np.isnan(col)
    return mytable[~has_nan]

