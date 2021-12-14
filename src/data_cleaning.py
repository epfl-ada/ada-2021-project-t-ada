"""
Functions to clean data.
"""
import pandas as pd
from tqdm import tqdm

from .constants import TOKENS_COL, USELESS_COLS

# Init progress bar
tqdm.pandas()


def drop_useless_columns(
    df: pd.DataFrame,
    colnames: list = USELESS_COLS,
) -> None:
    """Drops useless columns from a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        colnames (list, optional): names of columns to remove.
        Defaults to USELESS_COLS.
    """
    if all(col in df.columns for col in USELESS_COLS):
        df.drop(colnames, axis=1, inplace=True)


def remove_abnormalities(df: pd.DataFrame, verbose: bool = False) -> None:
    """Preprocesses and checks any abnormalities in the data:

    - Check and remove duplicated rows.
    - Check presence of missing entries.

    Args:
        df (pd.DataFrame): dataframe.
        verbose (bool, optional): True to show messages. Defaults to False.
    """
    # Remove duplicated rows
    if not df.index.is_unique:
        df.drop_duplicates(inplace=True)
        print('Info: duplicated rows were removed')
    elif verbose:
        print('No duplicated rows')

    # Check for missing entries anywhere in the dataframe
    if df.isna().values.any():
        print('Warning: presence of missing entries')
    elif verbose:
        print('No missing entries')


def convert_columns_type(df: pd.DataFrame, verbose: bool = False) -> None:
    """Converts columns types.

    Args:
        df (pd.DataFrame): dataframe.
        verbose (bool, optional): True to show old and new types.
        Defaults to False.
    """
    # Print old types
    if verbose:
        print('Old types:')
        print(df.dtypes)

    # Change the types to the appropriate ones
    df = df.convert_dtypes()

    # Change type of date into datetime type
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Print new types
    if verbose:
        print('\nNew types:')
        print(df.dtypes)


def drop_pron_tokens(df: pd.DataFrame) -> None:
    """Drops the -PRON- from the tokens column of a dataframe.

    Args:
        df (pd.DataFrame): dataframe.
    """
    df[TOKENS_COL] = df[TOKENS_COL].progress_apply(
        lambda x: [token for token in x if token != '-PRON-'])
