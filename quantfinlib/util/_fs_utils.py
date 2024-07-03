from pathlib import Path
from typing import Union


def get_project_root(relative_filepath: Union[Path, str, None] = None) -> Path:
    """Return the project's root directory: '../quantfinlib/'.

    Parameters
    ----------
    dependent_file_path (Union[Path, str, None], optional):
    additional filepath relative to the root, by default None

    Returns
    -------
    Path: The project's root directory with optional dependent file path.
    """
    root_path = Path(__file__).parent.parent.parent
    if relative_filepath is not None:
        root_path = root_path / relative_filepath
    return root_path
