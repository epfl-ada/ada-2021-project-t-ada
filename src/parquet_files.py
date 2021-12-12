"""
Functions to manage parquet files.
"""
import pandas as pd
from tqdm import tqdm


from .constants import QID, SPEAKER_COLUMNS

# Init progress bar
tqdm.pandas()


def create_df_from_parquet(
    filename: str,
    columns: list = SPEAKER_COLUMNS,
) -> pd.DataFrame:
    """Creates a dataframe from a parquet file.

    Args:
        filename (str): path to the bz2 file.

    Returns:
        pd.DataFrame: dataframe.
    """
    df = pd.read_parquet(filename, columns=columns)
    return df


def get_number_speakers_several_parties(df: pd.DataFrame) -> int:
    """Returns the number of speakers affiliated to several parties.

    Args:
        df (pd.DataFrame): dataframe of speakers.

    Returns:
        int: number of speakers affiliated to several parties.
    """
    mask = df.party.progress_apply(
        lambda x: len(x) > 1 if x is not None else 0
    )
    return df[mask].shape[0]


def affiliate_speakers_last_party(df: pd.DataFrame) -> None:
    """Affiliates the speakers to the last party of the list.

    Args:
        df (pd.DataFrame): dataframe of speakers.
    """
    df['party'] = df['party'].progress_apply(
        lambda x: x[-1] if x is not None else x
    )


def create_df_us_party(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the dataframe with american speakers with the party associated.

    It classifies the parties of speakers as following:
    - democrat: if one of the party of the speaker is democrat
    - republican: if one of the party of the speaker is republican
    - other: if the party of the speaker is not democrat neither republican
    - none: if the speaker has no party

    Args:
        df (pd.DataFrame): dataframe of speakers.

    Returns:
        pd.DataFrame: dataframe with american speakers and their party.
    """
    # Drop non-american speakers and nationality column
    mask_us = df.nationality.apply(lambda x: x is not None and QID['us'] in x)
    df_us = df[mask_us]
    df_us = df_us.drop(columns='nationality')

    # Set conditions for labelling
    mask_democrat = df_us['party'] == QID['democrat']
    mask_republican = df_us['party'] == QID['republican']
    mask_no_party = df_us.party.isna()

    # Create new column and setting it to 'other party' as default
    df_us['party_name'] = 'other party'
    df_us.loc[mask_democrat, 'party_name'] = 'democratic party'
    df_us.loc[mask_republican, 'party_name'] = 'republican party'
    df_us.loc[mask_no_party, 'party_name'] = 'no party'

    # Drop previous party column
    df_us = df_us.drop(columns='party')

    # Fixing column type
    df_us.party_name = df_us.party_name.astype('category')

    return df_us
