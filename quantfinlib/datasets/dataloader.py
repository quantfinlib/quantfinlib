import logging
from pathlib import Path

import pandas as pd

from quantfinlib.util.fs_utils import get_project_root

logger = logging.getLogger("main")

VIX_ONLINE_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
VIX_LOCAL_PATH = (
    get_project_root() / "quantfinlib" / "datasets" / "data" / "VIX_History.pkl"
)


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
    LOAD_LATEST_SUCCESS = False
    if load_latest:
        try:
            df = pd.read_csv(VIX_ONLINE_URL)
            LOAD_LATEST_SUCCESS = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the latest VIX data: {e}. Proceed with the local data."
            )

    if not load_latest or not LOAD_LATEST_SUCCESS:
        if not VIX_LOCAL_PATH.exists():
            raise FileNotFoundError(
                f"VIX dataset file '{VIX_LOCAL_PATH.name}' does not exist at '{VIX_LOCAL_PATH.parent}'"
            )

        df = pd.read_pickle(VIX_LOCAL_PATH, compression="gzip")

    df["DATE"] = pd.to_datetime(df["DATE"])
    logger.info(
        f"Loaded VIX dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
    )
    return df


if __name__ == "__main__":
    print(load_VIX().head())
    print(load_VIX(load_latest=True).head())
