"""Functions to load example datasets in quantfinlib."""

from pathlib import Path
from string import Template

import pandas as pd
import pkg_resources

from quantfinlib.util._fs_utils import get_project_root
from quantfinlib.util.logger_config import get_logger

logger = get_logger()

VIX_INDEX_LOCAL_PATH = get_project_root("quantfinlib/datasets/resources/VIX_ochl.pkl")
MULTI_INDEX_LOCAL_PATH = get_project_root("quantfinlib/datasets/resources/multi_index_close.pkl")
TREASURY_RATES_LOCAL_PATH = get_project_root("quantfinlib/datasets/resources/daily_treasury_rates.pkl")


def _load_pickle_to_df(filename: Path) -> pd.DataFrame:
    df = pd.read_pickle(filename, compression="gzip")
    return df


def load_vix() -> pd.DataFrame:
    """Load the VIX Index daily dataset with Open, Close,High, and Low index values.
    Source: https://www.cboe.com/tradable_products/vix/vix_historical_data/ (updated daily)

    Parameters
    ----------

    Returns
    -------
    pd.DataFrame N x 5
        The VIX dataset, 1 row per day, with columns: DATE, OPEN, HIGH, LOW, CLOSE

    Examples
    --------
    >>> df_vix = load_vix()
    """

    if not VIX_INDEX_LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"VIX dataset file '{VIX_INDEX_LOCAL_PATH.name}' does not exist at '{VIX_INDEX_LOCAL_PATH.parent}'"
        )
    else:
        logger.info(f"Reading VIX index data from {VIX_INDEX_LOCAL_PATH.name}...")
        df = _load_pickle_to_df(VIX_INDEX_LOCAL_PATH)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df.reset_index(drop=True, inplace=True)
        # df = df.set_index("DATE").sort_index()
        logger.debug(
            f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
        )
    return df


def load_treasury_rates() -> pd.DataFrame:
    """Load the daily Treasury rates dataset with various maturities.
    Souurce: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/${year}/all?type=daily_treasury_yield_curve&field_tdr_date_value=${year}&page&_format=csv

    Returns
    -------
        pd.DataFrame N x 14
            The daily Treasury rates dataset, 1 row per day, with columns: DATE, 1m, 2m, 3m, 4m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y

    Example
    -------
    >>> df_treasury_rates = load_treasury_rates()

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the specified path.
    """
    if not TREASURY_RATES_LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"Daily Treasury rates dataset file '{TREASURY_RATES_LOCAL_PATH.name}' does not exist at '{TREASURY_RATES_LOCAL_PATH.parent}'"
        )
    else:
        logger.info(f"Reading daily Treasury rates data from {TREASURY_RATES_LOCAL_PATH.name}...")
        df = _load_pickle_to_df(TREASURY_RATES_LOCAL_PATH)

        df = df.rename(
            columns={
                "Date": "DATE",
                "1 Mo": "1m",
                "2 Mo": "2m",
                "3 Mo": "3m",
                "4 Mo": "4m",
                "6 Mo": "6m",
                "1 Yr": "1y",
                "2 Yr": "2y",
                "3 Yr": "3y",
                "5 Yr": "5y",
                "7 Yr": "7y",
                "10 Yr": "10y",
                "20 Yr": "20y",
                "30 Yr": "30y",
            }
        )

        df["DATE"] = pd.to_datetime(df["DATE"])
        df.reset_index(drop=True, inplace=True)
        # df = df.set_index("DATE").sort_index()
        logger.debug(
            f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
        )
        return df


def load_equity_indices() -> pd.DataFrame:
    """Load the multi-index dataset with daily Close prices for various indices.
    Source: https://www.kaggle.com/datasets/mukhazarahmad/worldwide-stock-market-indices-data

    Returns
    -------
    pd.DataFrame N x 11
        The multi-index dataset, 1 row per day, with columns: DATE, %INDEX_NAME%

    Example
    -------
    >>> df_indices = load_equity_indices()
    """
    if not MULTI_INDEX_LOCAL_PATH.exists():
        raise FileNotFoundError(
            f"Multi-index dataset file '{MULTI_INDEX_LOCAL_PATH.name}' does not exist at '{MULTI_INDEX_LOCAL_PATH.parent}'"
        )
    else:
        logger.info(f"Reading multi-index data from {MULTI_INDEX_LOCAL_PATH.name}...")
        df = _load_pickle_to_df(MULTI_INDEX_LOCAL_PATH)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df.reset_index(drop=True, inplace=True)
        # df = df.set_index("DATE").sort_index()

        logger.info(
            f"Loaded multi-index dataset with {df.shape[0]} rows and {df.shape[1]} columns. Latest date: {df['DATE'].max()}"
        )

    return df


if __name__ == "__main__":
    print(load_vix().head())
    logger.info("VIX dataset loaded successfully.")

    print(load_equity_indices().head())
    logger.info("Multi-index dataset loaded successfully.")

    print(load_treasury_rates().head())
    logger.info("Treasury rates dataset loaded successfully.")
