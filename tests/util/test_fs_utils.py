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


def test_get_project_root_with_argument(root_dir, relative_path):
    assert isinstance(root_dir, Path)
    assert isinstance(relative_path, str) | isinstance(relative_path, Path)
    expected_path = root_dir / relative_path
    assert get_project_root(relative_path) == expected_path


def test_get_project_root_empty_string_argument(root_dir):
    assert isinstance(root_dir, Path)
    assert root_dir.exists()
    relative_path = ""
    assert get_project_root(relative_path) == root_dir


def test_get_project_root_with_path_object_argument(root_dir, relative_path):
    assert isinstance(root_dir, Path)
    assert isinstance(relative_path, str) | isinstance(relative_path, Path)
    pathlib_obj = Path(relative_path)
    expected_path = root_dir / pathlib_obj
    assert get_project_root(pathlib_obj) == expected_path


def test_get_project_root_nested_directories_argument(root_dir):
    assert isinstance(root_dir, Path)
    expected_path = root_dir / "directory1" / "directory2" / "file.txt"
    assert get_project_root("directory1/directory2/file.txt") == expected_path


def test_get_project_root_nonexistent_argument(root_dir):
    assert isinstance(root_dir, Path)
    assert not (root_dir / "nonexistent").exists()


if __name__ == "__main__":
    pytest.main([__file__])
