# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Legacy utilities for Astropy table manipulation.

WARNING
-------
This module is deprecated and kept only for backward compatibility.

New code MUST use:
    feupy.utils.tables.utils

These functions may be removed in future versions without notice.
"""

import warnings

__all__ = [
    "column_to_string",
    "append_nones",
]


# ============================================================
# Legacy: Column utilities
# ============================================================

def column_to_string(column):
    """
    DEPRECATED: Convert column to string representation.

    Use:
        feupy.utils.tables.utils.column_to_string
    """
    warnings.warn(
        "column_to_string is deprecated. Use utils.tables.utils instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return f"[{','.join(str(x) for x in column)}]"


# ============================================================
# Legacy: List utilities
# ============================================================

def append_nones(length, list_):
    """
    DEPRECATED: Pad list with None values.

    Use:
        feupy.utils.tables.utils.pad_list_to_length
    """
    warnings.warn(
        "append_nones is deprecated. Use pad_list_to_length instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    diff = length - len(list_)

    if diff < 0:
        raise ValueError("Input list is longer than target length.")

    return list_ + [None] * diff