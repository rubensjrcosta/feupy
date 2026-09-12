# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Container class for catalog source objects."""

import collections.abc
import copy
import logging
from pathlib import Path

import yaml
from astropy.coordinates import SkyCoord

from feupy.catalogs import FEUPY_CATALOG_REGISTRY
from feupy.catalogs.utils import get_catalog_tag

__all__ = ["Sources"]

log = logging.getLogger(__name__)


class Sources(collections.abc.MutableSequence):
    """Collection of catalog sources.

    Parameters
    ----------
    sources : source object, list of source objects, or `Sources`, optional
        Sources used to initialize the collection.
    """

    def __init__(self, sources=None):
        if sources is None:
            sources = []
        elif isinstance(sources, Sources):
            sources = sources._sources
        elif self._is_source(sources):
            sources = [sources]
        elif not isinstance(sources, list):
            log.error("Failed Invalid type: %r", sources)
            raise TypeError(f"Invalid type: {sources!r}")

        labels = []
        for source in sources:
            label = self._get_source_label(source)
            if label in labels:
                message = (
                    f"Source name '{source.name}' from "
                    f"{get_catalog_tag(source)} already exists!"
                )
                log.error("Failed %s", message)
                raise ValueError(message)
            labels.append(label)

        self._sources = sources

    @staticmethod
    def _is_source(source):
        """Check whether an object is a supported catalog source."""
        return any(
            isinstance(source, catalog.source_object_class)
            for catalog in FEUPY_CATALOG_REGISTRY
        )

    @staticmethod
    def _get_source_label(source):
        """Return the unique label associated with a source."""
        tag = get_catalog_tag(source)
        return f"{source.name} ({tag})"

    def __getitem__(self, key):
        return self._sources[self.index(key)]

    def __delitem__(self, key):
        del self._sources[self.index(key)]

    def __setitem__(self, key, source):
        if not self._is_source(source):
            log.error("Failed Invalid type: %r", type(source))
            raise TypeError(f"Invalid type: {type(source)!r}")

        label = self._get_source_label(source)
        current_index = self.index(key)

        if label in self.labels and self.labels[current_index] != label:
            message = (
                f"Source name '{source.name}' from "
                f"{get_catalog_tag(source)} already exists!"
            )
            log.error("Failed %s", message)
            raise ValueError(message)

        self._sources[current_index] = source

    def __len__(self):
        return len(self._sources)

    def insert(self, index, source):
        """Insert a source into the collection.

        Parameters
        ----------
        index : int
            Position at which the source is inserted.
        source : source object
            Source to insert.

        Raises
        ------
        TypeError
            If ``source`` is not a supported catalog source.
        ValueError
            If the source label already exists in the collection.
        """
        if not self._is_source(source):
            log.error("Failed Invalid type: %r", type(source))
            raise TypeError(f"Invalid type: {type(source)!r}")

        label = self._get_source_label(source)
        if label in self.labels:
            message = (
                f"Source name '{source.name}' from "
                f"{get_catalog_tag(source)} already exists!"
            )
            log.error("Failed %s", message)
            raise ValueError(message)

        self._sources.insert(index, source)

    def index(self, key):
        """Return the index associated with a key.

        Parameters
        ----------
        key : int, slice, str, or source object
            Index, source name, or source object.

        Returns
        -------
        index : int or slice
            Corresponding collection index.
        """
        if isinstance(key, (int, slice)):
            return key
        if isinstance(key, str):
            return self.names.index(key)
        if key in self._sources:
            return self._sources.index(key)

        log.error("Failed Invalid type: %r", type(key))
        raise TypeError(f"Invalid type: {type(key)!r}")

    def copy(self):
        """Return a deep copy of the source collection."""
        return copy.deepcopy(self)

    @property
    def names(self):
        """Source names."""
        return [source.name for source in self._sources]

    @property
    def labels(self):
        """Source labels including their catalog tags."""
        return [self._get_source_label(source) for source in self._sources]

    @property
    def positions(self):
        """Source positions as a `~astropy.coordinates.SkyCoord` object."""
        ra = [source.position.icrs.ra for source in self._sources]
        dec = [source.position.icrs.dec for source in self._sources]
        return SkyCoord(ra, dec, frame="icrs")

    def select(self, names):
        """Select sources by name.

        Parameters
        ----------
        names : iterable of str
            Source names to select.

        Returns
        -------
        sources : `Sources`
            Selected sources.
        """
        names = set(names)
        return Sources([source for source in self if source.name in names])

    def write(self, filename, overwrite=False):
        """Write the source collection to a YAML file.

        Parameters
        ----------
        filename : path-like
            Output filename.
        overwrite : bool, optional
            Whether to overwrite an existing file.

        Raises
        ------
        OSError
            If the file exists and ``overwrite`` is False.
        """
        path = Path(filename)
        if path.exists() and not overwrite:
            raise OSError(f"File exists: {filename}")

        data = {
            "Sources": {
                index: {
                    "name": source.name,
                    "catalog": get_catalog_tag(source),
                }
                for index, source in enumerate(self._sources)
            }
        }

        with path.open("w") as stream:
            yaml.safe_dump(data, stream)

    def read(self, filename):
        """Read sources from a YAML file.

        Parameters
        ----------
        filename : path-like
            Input YAML filename.
        """
        with open(filename) as stream:
            data = yaml.safe_load(stream)

        self._sources = [
            FEUPY_CATALOG_REGISTRY.get_cls(entry["catalog"])()[entry["name"]]
            for entry in data["Sources"].values()
        ]
