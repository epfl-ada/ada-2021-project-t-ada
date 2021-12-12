"""
Functions to clean data.
"""
import pandas as pd

from .constants import USELESS_COLS


def drop_useless_columns(
    df: pd.DataFrame,
    colnames: list = USELESS_COLS,
) -> pd.DataFrame:
    """Drops useless columns from a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        colnames (list, optional): names of columns to remove.
        Defaults to USELESS_COLS.

    Returns:
        pd.DataFrame: dataframe with columns dropped.
    """
    return df.drop(colnames, axis=1)
