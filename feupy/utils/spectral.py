# Licensed under a 3-clause BSD style license - see LICENSE
"""Spectral model helper utilities."""

__all__ = ["get_ecut_from_ecpl"]


def get_ecut_from_ecpl(sky_model, fmt="{:.2f} \\pm {:.2f}"):
    """Extract the cutoff energy from an ECPL spectral model.

    Parameters
    ----------
    sky_model : `~gammapy.modeling.models.SkyModel`
        Sky model containing an exponential cutoff power-law spectral model.
    fmt : str, optional
        Format string used for the cutoff energy and its uncertainty.
        Default is ``"{:.2f} \\\\pm {:.2f}"``.

    Returns
    -------
    ecut : str
        Formatted cutoff energy and uncertainty.

    Raises
    ------
    TypeError
        If the spectral model does not contain a ``lambda_`` parameter.
    ValueError
        If ``lambda_`` has no positive uncertainty.
    """
    spectral_model = sky_model.spectral_model

    if not hasattr(spectral_model, "lambda_"):
        raise TypeError(
            "Spectral model does not contain parameter 'lambda_' "
            "(expected ExpCutoffPowerLawSpectralModel)."
        )

    lambda_value = spectral_model.lambda_.value
    lambda_error = spectral_model.lambda_.error

    if lambda_error is None or lambda_error <= 0:
        raise ValueError("Parameter 'lambda_' has no associated error.")

    ecut = 1.0 / lambda_value
    ecut_error = ecut * (lambda_error / lambda_value)

    return fmt.format(ecut, ecut_error)
