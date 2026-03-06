# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Catalog-aware marker assignment."""

from feupy.core.sources import Sources
from feupy.catalog.utils import get_catalog_tag
from gammapy.utils.scripts import recursive_merge_dicts
from .plotting import make_marker_dict

def map_catalog_to_marker(tag):
    tag = tag.lower()

    if "fhl" in tag or "fgl" in tag:
        return "v"
    if "hgps" in tag or "hess" in tag:
        return "p"
    if "veritas" in tag or "vtscat" in tag:
        return ">"
    if "hwc" in tag:
        return "8"
    if "gamma" in tag:
        return "h"
    if "lhaaso" in tag:
        return "s"
    if tag in {"psrcat", "2pc", "3pc"}:
        return "*"

    raise ValueError(f"No marker defined for catalog '{tag}'.")


def make_catalog_marker_dict(sources, datasets=None, **kwargs):
    ref_markers = {}

    for tag in {get_catalog_tag(s) for s in sources}:
        marker = map_catalog_to_marker(tag)

        selected = Sources(
            [s for s in sources if get_catalog_tag(s) == tag]
        )
        labels = selected.labels

        if datasets:
            for src in selected:
                labels.extend(
                    [d.name for d in datasets if d.name.startswith(src.name)]
                )

        markers = make_marker_dict(labels, marker=marker, **kwargs)
        ref_markers = recursive_merge_dicts(ref_markers, markers)

    return ref_markers
