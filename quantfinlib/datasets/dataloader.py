"""Functions to load example datasets in quantfinlib."""

import pandas as pd

from quantfinlib.util._fs_utils import get_project_root
from quantfinlib.util.logger_config import get_logger

logger = get_logger()

VIX_ONLINE_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
VIX_LOCAL_PATH = get_project_root("quantfinlib/datasets/data/VIX_ochl.pkl")


MULTI_INDEX_LOCAL_PATH = get_project_root(
    "quantfinlib/datasets/data/multi_index_close.pkl"
)


def load_multi_index() -> pd.DataFrame:
    """Load the multi-index dataset with daily Close prices for various indices.
    Source: https://www.kaggle.com/datasets/mukhazarahmad/worldwide-stock-market-indices-data

    Returns
    -------
    pd.DataFrame N x 11
        The multi-index dataset, 1 row per day, with columns: DATE, %INDEX_NAME%
    """
    df = pd.read_pickle(MULTI_INDEX_LOCAL_PATH, compression="gzip")
    logger.info(
        f"Loaded multi-index dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
    )
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.reset_index(drop=True, inplace=True)
    return df


def load_VIX(load_latest: bool = False) -> pd.DataFrame:
    """Load the VIX Index daily dataset with Open, Close,High, and Low prices.
    Source: https://www.cboe.com/tradable_products/vix/vix_historical_data/ (updated daily)

    Parameters
    ----------
    load_latest : bool, optional
        Load the latest VIX data from the CBOE website, by default False

    Returns
    -------
    pd.DataFrame N x 5
        The VIX dataset, 1 row per day, with columns: DATE, OPEN, HIGH, LOW, CLOSE

    Examples
    --------
    >>> df_vix = load_vix()
    >>> df_vix_latest = load_vix(load_latest=True)
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
    df.reset_index(drop=True, inplace=True)
    logger.info(
        f"Loaded VIX dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
    )
    return df


if __name__ == "__main__":
    print(load_VIX().head())
    print(load_VIX(load_latest=True).head())
    logger.info("VIX dataset loaded successfully.")

    print(load_multi_index().head())
    logger.info("Multi-index dataset loaded successfully.")
