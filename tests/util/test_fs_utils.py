from pathlib import Path

import pytest

from quantfinlib.util.fs_utils import get_project_root


@pytest.fixture
def root_dir():
    return get_project_root()


@pytest.fixture
def relative_path():
    return "quantfinlib"


def test_get_project_root_without_argument(root_dir):
    # Assert that the root directory is a Path object
    assert isinstance(root_dir, Path)

    # Assert that the root directory exists
    assert root_dir.exists()


def test_get_project_root_with_argument(root_dir, relative_path):
    # Get the expected path by appending the relative file path to the root directory
    expected_path = root_dir / relative_path

    # Call the method and assert that the result matches the expected path
    assert get_project_root(relative_path) == expected_path


def test_get_project_root_with_path_object_argument(root_dir, relative_path):
    # Create a Path object for the relative file path
    file_path_obj = Path(relative_path)

    # Get the expected path by appending the relative file path to the root directory
    expected_path = root_dir / file_path_obj

    # Call the method and assert that the result matches the expected path
    assert get_project_root(file_path_obj) == expected_path
