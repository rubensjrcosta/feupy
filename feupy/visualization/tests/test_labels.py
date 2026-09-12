# Licensed under a 3-clause BSD style license - see LICENSE.rst

from feupy.visualization.utils.labels import (
    DEFAULT_XAXIS_LABEL,
    DEFAULT_YAXIS_LABEL,
)


def test_yaxis_labels_keys():
    assert set(DEFAULT_YAXIS_LABEL) == {"e2dnde", "dnde"}


def test_yaxis_labels_content():
    assert "E^{2}" in DEFAULT_YAXIS_LABEL["e2dnde"]
    assert "\\Phi(E)" in DEFAULT_YAXIS_LABEL["e2dnde"]
    assert "\\Phi(E)" in DEFAULT_YAXIS_LABEL["dnde"]


def test_xaxis_labels_keys():
    assert set(DEFAULT_XAXIS_LABEL) == {"erg", "TeV"}


def test_xaxis_labels_content():
    assert DEFAULT_XAXIS_LABEL["erg"].startswith("Energy [")
    assert DEFAULT_XAXIS_LABEL["TeV"].startswith("Energy [")
    assert "erg" in DEFAULT_XAXIS_LABEL["erg"]
    assert "TeV" in DEFAULT_XAXIS_LABEL["TeV"]
