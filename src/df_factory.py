"""
Functions to create dataframes.
"""
import bz2
import json

import pandas as pd
from tqdm import tqdm

from .constants import QID_COL, QUOTATION_COL, TEST_DATA_PATH

pd.options.mode.chained_assignment = None

# Init progress bar
tqdm.pandas()


def create_df_from_bz2(filename: str) -> pd.DataFrame:
    """Creates a dataframe from a bz2 file.

    Args:
        filename (str): path to the bz2 file.

    Returns:
        pd.DataFrame: dataframe.
    """
    with bz2.open(filename, 'rb') as f:
        data = f.readlines()
    data = list(map(json.loads, data))
    df = pd.DataFrame(data)
    df.set_index('quoteID', inplace=True)
    return df


def create_df_test() -> pd.DataFrame:
    """Creates the dataframe from the test dataset of quotes from the NYT.

    Returns:
        pd.DataFrame: test dataframe of quotes.
    """
    return create_df_from_bz2(TEST_DATA_PATH)


def create_df_speaker(df: pd.DataFrame, qid: str) -> pd.DataFrame:
    """Returns the dataframe corresponding to a speaker from its QID.

    Args:
        df (pd.Dataframe): main dataframe.
        qid (str): QID (can be found on Wikidata).

    Returns:
        pd.DataFrame: dataframe of the speaker.
    """
    if QID_COL in df.columns:
        return df[df[QID_COL] == qid]

    mask = df.qids.progress_apply(lambda x: x != [] and qid == x[0])
    return df[mask]


def add_col_qid(df: pd.DataFrame) -> None:
    """Adds the column with the first QID.

    Args:
        df (pd.DataFrame): dataframe.
    """
    df[QID_COL] = df.qids.progress_apply(lambda x: x[0] if x else None)


def create_df_joined_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the dataframe with one row by speaker with joined quotes.

    Args:
        df (pd.Dataframe): main dataframe.

    Returns:
        pd.DataFrame: dataframe with joined quotes.
    """
    # Adds qid column if not exists
    if QID_COL not in df.columns:
        add_col_qid(df)

    # Group by qid and join quotes
    return df.groupby(QID_COL, as_index=False)[QUOTATION_COL].progress_apply(
        lambda x: ' '.join(x)
    )
