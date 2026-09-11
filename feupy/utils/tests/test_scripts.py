# Licensed under a 3-clause BSD style license - see LICENSE

from feupy.utils import scripts
from feupy.utils.scripts import is_documented_by, pickling, unpickling


def test_all():
    expected = {
        "is_documented_by",
        "pickling",
        "unpickling",
    }

    assert set(scripts.__all__) == expected

    for name in scripts.__all__:
        assert hasattr(scripts, name)


def test_is_documented_by_single():
    def original():
        """Original docstring."""

    @is_documented_by(original)
    def target():
        """Target docstring."""

    assert "Original docstring." in target.__doc__
    assert "Target docstring." in target.__doc__


def test_is_documented_by_list():
    def first():
        """First docstring."""

    def second():
        """Second docstring."""

    @is_documented_by([first, second])
    def target():
        """Target docstring."""

    assert "first:" in target.__doc__
    assert "First docstring." in target.__doc__
    assert "second:" in target.__doc__
    assert "Second docstring." in target.__doc__
    assert "Target docstring." in target.__doc__


def test_is_documented_by_without_target_docstring():
    def original():
        """Original docstring."""

    @is_documented_by(original)
    def target():
        pass

    assert "Original docstring." in target.__doc__


def test_pickling_and_unpickling(tmp_path):
    data = {
        "name": "Crab",
        "values": [1, 2, 3],
    }
    filename = tmp_path / "result"

    pickling(data, filename)
    result = unpickling(filename)

    assert (tmp_path / "result.pkl").is_file()
    assert result == data
