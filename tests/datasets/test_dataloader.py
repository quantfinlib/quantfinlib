from pathlib import Path

import pandas as pd
import pytest

from quantfinlib.datasets import load_equity_indices, load_treasury_rates, load_vix


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
    df = load_vix()

    assert isinstance(df, pd.DataFrame)

    required_columns = ["HIGH", "LOW", "CLOSE", "OPEN"]
    assert all(column in df.columns for column in required_columns)

    assert set(df.columns) == set(required_columns)


@pytest.mark.skip(reason="Test is deprecated")
def test_load_VIX_online():

    try:
        df = load_vix(load_latest=True)
    except RuntimeError:
        pytest.fail("load_VIX() raised RuntimeError unexpectedly!")

    assert isinstance(df, pd.DataFrame)

    required_columns = ["HIGH", "LOW", "CLOSE", "OPEN"]
    assert all(column in df.columns for column in required_columns)

    assert set(df.columns) == set(required_columns)


def test_load_treasury_rates():
    df = load_treasury_rates()
    assert isinstance(df, pd.DataFrame)
    required_columns = ["1m", "2m", "3m", "4m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
    assert all(c in df.columns for c in required_columns)
    assert set(df.columns) == set(required_columns)


def test_load_multi_index():
    df = load_equity_indices()
    assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
