# Licensed under a 3-clause BSD style license - see LICENSE
"""Deprecated table utilities kept for backward compatibility."""

import warnings

__all__ = [
    "column_to_string",
    "append_nones",
]


def column_to_string(column):
    """Convert a column to a compact string representation.

    .. deprecated::
        This function is kept only for backward compatibility.
    """
    warnings.warn(
        "column_to_string is deprecated.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"[{','.join(str(value) for value in column)}]"


def append_nones(length, input_list):
    """Pad a list to a given length using None values.

    .. deprecated::
        Use :func:`feupy.utils.tables.utils.pad_list_to_length` for new code.
    """
    warnings.warn(
        "append_nones is deprecated. Use pad_list_to_length instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    diff = length - len(input_list)

    if diff < 0:
        raise ValueError("Input list is longer than the requested target length.")

    return input_list + [None] * diff
