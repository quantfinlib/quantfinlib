from pathlib import Path

import pandas as pd
import pytest

from quantfinlib.datasets.dataloader import load_multi_index, load_VIX


@pytest.fixture
def vix_data():  # Fixture to provide a sample DataFrame for testing
    data = {
        "DATE": pd.date_range("2022-01-01", periods=10),
        "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "Close": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        "Low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
    }
    return pd.DataFrame(data)


def test_load_VIX_local(monkeypatch):
    monkeypatch.setattr(
        "quantfinlib.util._fs_utils.get_project_root",
        lambda: Path("/non/existent/path"),
    )
    # monkeypatch.setattr(pd, "read_pickle", lambda path, **kwargs: vix_data)
    df = load_VIX()

    assert isinstance(df, pd.DataFrame)

    required_columns = ["DATE", "HIGH", "LOW", "CLOSE", "OPEN"]
    assert all(column in df.columns for column in required_columns)

    assert set(df.columns) == set(required_columns)


# write test_load_VIX_online
def test_load_VIX_online():

    try:
        df = load_VIX(load_latest=True)
    except RuntimeError:
        pytest.fail("load_VIX() raised RuntimeError unexpectedly!")

    assert isinstance(df, pd.DataFrame)

    required_columns = ["DATE", "HIGH", "LOW", "CLOSE", "OPEN"]
    assert all(column in df.columns for column in required_columns)

    assert set(df.columns) == set(required_columns)


def test_load_multi_index():
    df = load_multi_index()
    assert isinstance(df, pd.DataFrame)
    assert "DATE" in df.columns


if __name__ == "__main__":
    pytest.main([__file__])
