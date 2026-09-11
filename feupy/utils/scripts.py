# Licensed under a 3-clause BSD style license - see LICENSE
"""Utilities for scripts and command-line tools."""

import pickle

from gammapy.utils.scripts import make_path

__all__ = [
    "is_documented_by",
    "pickling",
    "unpickling",
]


def is_documented_by(original):
    """Copy docstrings from one or more objects to a target.

    Parameters
    ----------
    original : object or list of objects
        Function, class, or list of objects whose docstrings are copied.

    Returns
    -------
    decorator : callable
        Decorator that assigns the combined docstring to the target.
    """

    def wrapper(target):
        doc = "*** Docstring of internal function/class ***\n"

        if isinstance(original, list):
            for item in original:
                doc += f"{item.__qualname__}:\n{item.__doc__}\n"
        else:
            doc += f"{original.__doc__}\n"

        if target.__doc__:
            doc += f"\n*** Docstring of {target.__qualname__} ***\n{target.__doc__}"

        target.__doc__ = doc
        return target

    return wrapper


def pickling(object_instance, file_name):
    """Serialize an object to a pickle file.

    Parameters
    ----------
    object_instance : object
        Object to serialize.
    file_name : str or `~pathlib.Path`
        Output filename without the ``.pkl`` extension.
    """
    filename = make_path(f"{file_name}.pkl")

    with filename.open("wb") as file:
        pickle.dump(object_instance, file)


def unpickling(file_name):
    """Load an object from a pickle file.

    Parameters
    ----------
    file_name : str or `~pathlib.Path`
        Input filename without the ``.pkl`` extension.

    Returns
    -------
    object
        Deserialized object.
    """
    filename = make_path(f"{file_name}.pkl")

    with filename.open("rb") as file:
        return pickle.load(file)
