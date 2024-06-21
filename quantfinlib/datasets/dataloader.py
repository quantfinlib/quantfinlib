from pathlib import Path

import pandas as pd

from quantfinlib.util.fs_utils import get_project_root


def load_VIX(load_latest: bool = False) -> pd.DataFrame:
    """Load the VIX dataset with Open, Close,High, and Low prices.
    Source: https://www.cboe.com/tradable_products/vix/vix_historical_data/ (updated daily)

    Parameters
    ----------
    load_latest : bool, optional
        Load the latest VIX data from the CBOE website, by default False

    Returns
    -------
    pd.DataFrame
        The VIX dataset.

    Examples
    --------
    >>> df = load_vix()
    """
    if load_latest:
        try:
            df = pd.read_csv(
                "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            )
            LOAD_SUCCESS = True
        except Exception as e:
            LOAD_SUCCESS = False
            raise RuntimeError(
                f"Failed to load the latest VIX data: {e}. Proceed with the local data."
            )

    if not load_latest or not LOAD_SUCCESS:
        data_filename = "VIX_History.pkl"
        data_path = (
            get_project_root() / "quantfinlib" / "datasets" / "data" / data_filename
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"VIX dataset file '{data_filename}' does not exist at '{data_path}'"
            )

        df = pd.read_pickle(data_path, compression="gzip")

    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


if __name__ == "__main__":
    print(load_VIX().head())
    print(load_VIX(load_latest=True).head())
