# Licensed under a 3-clause BSD style license - see LICENSE.rst

from feupy.visualization import labels


def test_yaxis_labels_keys():
    assert "e2dnde" in labels.DEFAULT_YAXIS_LABEL
    assert "dnde" in labels.DEFAULT_YAXIS_LABEL


def test_yaxis_labels_content():
    assert "E^{2}" in labels.DEFAULT_YAXIS_LABEL["e2dnde"]
    assert "\\Phi(E)" in labels.DEFAULT_YAXIS_LABEL["dnde"]


def test_xaxis_labels_keys():
    assert "erg" in labels.DEFAULT_XAXIS_LABEL
    assert "TeV" in labels.DEFAULT_XAXIS_LABEL


def test_xaxis_labels_content():
    assert "Energy" in labels.DEFAULT_XAXIS_LABEL["erg"]
    assert "Energy" in labels.DEFAULT_XAXIS_LABEL["TeV"]
