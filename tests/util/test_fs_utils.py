from pathlib import Path

import pytest

from quantfinlib.util._fs_utils import get_project_root


@pytest.fixture
def root_dir():
    return get_project_root()


@pytest.fixture
def relative_path():
    return "quantfinlib"


def test_get_project_root_without_argument(root_dir):
    assert isinstance(root_dir, Path)
    assert root_dir.exists()
    expected = root_dir
    assert get_project_root() == expected
    assert isinstance(get_project_root(), Path)


def test_get_project_root_with_argument(root_dir, relative_path):
    assert isinstance(root_dir, Path)
    assert isinstance(relative_path, str) | isinstance(relative_path, Path)
    expected = root_dir / relative_path
    output = get_project_root(relative_path)
    assert output == expected
    assert isinstance(output, Path)


def test_get_project_root_empty_string_argument(root_dir):
    assert isinstance(root_dir, Path)
    assert root_dir.exists()
    relative_path = ""
    expected = root_dir
    output = get_project_root(relative_path)
    assert output == expected
    assert isinstance(output, Path)


def test_get_project_root_with_path_object_argument(root_dir, relative_path):
    assert isinstance(root_dir, Path)
    assert isinstance(relative_path, str) | isinstance(relative_path, Path)
    pathlib_obj = Path(relative_path)
    expected = root_dir / pathlib_obj
    output = get_project_root(pathlib_obj)
    assert output == expected
    assert isinstance(output, Path)


def test_get_project_root_nested_directories_argument(root_dir):
    assert isinstance(root_dir, Path)
    expected = root_dir / "directory1" / "directory2" / "file.txt"
    output = get_project_root("directory1/directory2/file.txt")
    assert output == expected
    assert isinstance(output, Path)


def test_get_project_root_nonexistent_argument(root_dir):
    assert isinstance(root_dir, Path)
    assert not (root_dir / "nonexistent").exists()
    expected = root_dir / "nonexistent"
    output = get_project_root("nonexistent")
    assert output == expected
    assert isinstance(output, Path)


if __name__ == "__main__":
    pytest.main([__file__])
