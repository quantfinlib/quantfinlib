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
    expected_path = root_dir / relative_path
    assert get_project_root(relative_path) == expected_path


def test_get_project_root_with_path_object_argument(root_dir, relative_path):
    pathlib_obj = Path(relative_path)
    expected_path = root_dir / pathlib_obj
    assert get_project_root(pathlib_obj) == expected_path
