# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spectral model helper utilities.
"""

__all__ = [
    "get_ecut_from_ecpl",
]


def get_ecut_from_ecpl(sky_model, fmt="{:.2f} \\pm {:.2f}"):
    """
    Extract cutoff energy from an ECPL spectral model.

    Parameters
    ----------
    sky_model : `~gammapy.modeling.models.SkyModel`
        Sky model containing an ExpCutoffPowerLawSpectralModel.

    fmt : str, optional
        Format string for value ± error.

    Returns
    -------
    ecut : str
        Formatted cutoff energy.
    """

    spec = sky_model.spectral_model

    if not hasattr(spec, "lambda_"):
        raise TypeError(
            "Spectral model does not contain parameter 'lambda_' "
            "(expected ExpCutoffPowerLawSpectralModel)."
        )

    lam = spec.lambda_.value
    lam_err = spec.lambda_.error

    # Gammapy parameters never return None, but may return 0
    if lam_err is None or lam_err <= 0:
        raise ValueError("Parameter 'lambda_' has no associated error.")

    ecut = 1.0 / lam
    ecut_err = ecut * (lam_err / lam)

    return fmt.format(ecut, ecut_err)