from pathlib import Path

import pandas as pd
import pytest

from quantfinlib.datasets import load_equity_indices, load_treasury_rates, load_vix


def test_load_vix_local(monkeypatch: pytest.MonkeyPatch):
    df = load_vix()
    assert isinstance(df, pd.DataFrame)
    required_columns = ["HIGH", "LOW", "CLOSE", "OPEN"]
    assert set(df.columns) == set(required_columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.empty

    # Test FileNotFoundError
    monkeypatch.setattr(
        "quantfinlib.datasets._dataloader.VIX_INDEX_LOCAL_PATH",
        Path("non_existent/path"),
    )
    with pytest.raises(FileNotFoundError) as e:
        _ = load_vix()
    assert str(e.value) == "VIX dataset file 'path' does not exist at 'non_existent'"


def test_load_treasury_rates(monkeypatch: pytest.MonkeyPatch):
    df = load_treasury_rates()
    assert isinstance(df, pd.DataFrame)
    required_columns = ["1m", "2m", "3m", "4m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
    assert set(df.columns) == set(required_columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.empty

    # Test FileNotFoundError
    monkeypatch.setattr(
        "quantfinlib.datasets._dataloader.TREASURY_RATES_LOCAL_PATH",
        Path("non_existent/path"),
    )
    with pytest.raises(FileNotFoundError) as e:
        _ = load_treasury_rates()
    assert str(e.value) == "Daily Treasury rates dataset file 'path' does not exist at 'non_existent'"


def test_load_multi_index(monkeypatch: pytest.MonkeyPatch):
    df = load_equity_indices()
    assert isinstance(df, pd.DataFrame)
    assert "VIX" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.empty

    # Test FileNotFoundError
    monkeypatch.setattr(
        "quantfinlib.datasets._dataloader.MULTI_INDEX_LOCAL_PATH",
        Path("non_existent/path"),
    )
    with pytest.raises(FileNotFoundError) as e:
        _ = load_equity_indices()
    assert str(e.value) == "Multi-index dataset file 'path' does not exist at 'non_existent'"


if __name__ == "__main__":
    pytest.main([__file__])
