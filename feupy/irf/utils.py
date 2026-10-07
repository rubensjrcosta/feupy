# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for CTAO IRFs."""

import logging
from itertools import product

from feupy.irf import CTAOIRFManager

log = logging.getLogger(__name__)

__all__ = ["get_irf_groups"]


def get_irf_groups(
    groups,
    production="prod5",
    condition="dark",
):
    """Generate all combinations of CTAO IRF options.

    Parameters
    ----------
    groups : list
        Sequence containing the IRF options in the order
        ``[array, azimuth, zenith, observation_time]``.
        Each element may be either a single value or a list
        of values.
    production : {"prod5", "prod6"}, optional
        CTAO IRF production. Default is ``"prod5"``.
    condition : {"dark", "halfmoon"}, optional
        Observing condition. This option is used by Prod6.
        Default is ``"dark"``.

    Returns
    -------
    required_irfs : list of list
        List containing all generated IRF option combinations.
    irfs : list
        List of IRFs corresponding to each combination.
    """
    manager = CTAOIRFManager(
        production=production,
        condition=condition,
    )

    normalized = [
        values
        if isinstance(values, list)
        else [values]
        for values in groups
    ]

    required_irfs = [
        list(options)
        for options in product(*normalized)
    ]

    irfs = [
        manager.get_irf(options)["irf"]
        for options in required_irfs
    ]

    return required_irfs, irfs