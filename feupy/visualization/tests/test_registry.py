# Licensed under a 3-clause BSD style license - see LICENSE.rst

from feupy.visualization.styles.markers.registry import (
    CATALOG_STYLE_REGISTRY,
    CatalogStyle,
    CatalogStyleRegistry,
)


def test_catalog_style_defaults():
    style = CatalogStyle("4FGL")

    assert style.tag == "4fgl"
    assert style.marker == "o"
    assert style.color is None


def test_catalog_style_custom_values():
    style = CatalogStyle("HGPS", marker="p", color="red")

    assert style.tag == "hgps"
    assert style.marker == "p"
    assert style.color == "red"


def test_registry_register_and_get():
    registry = CatalogStyleRegistry()
    style = CatalogStyle("4FGL", marker="v")

    registry.register(style)

    assert registry.get("4fgl") is style
    assert registry.get("4FGL") is style


def test_registry_unknown_tag():
    registry = CatalogStyleRegistry()

    assert registry.get("unknown") is None


def test_registry_tags():
    registry = CatalogStyleRegistry()
    registry.register(CatalogStyle("4FGL"))
    registry.register(CatalogStyle("HGPS"))

    assert registry.tags() == ["4fgl", "hgps"]


def test_registry_overwrites_existing_tag():
    registry = CatalogStyleRegistry()

    first = CatalogStyle("4FGL", marker="v")
    second = CatalogStyle("4fgl", marker="s")

    registry.register(first)
    registry.register(second)

    assert registry.get("4fgl") is second
    assert registry.get("4fgl").marker == "s"
    assert registry.tags() == ["4fgl"]


def test_global_registry_instance():
    assert isinstance(CATALOG_STYLE_REGISTRY, CatalogStyleRegistry)
