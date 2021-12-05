"""
Sentiment analysis functions.
"""
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from .constants import COMPOUND_SCORE_COL, QUOTATION_COL

pd.options.mode.chained_assignment = None

# Init progress bar
tqdm.pandas()

# Download lexicon
nltk.download('vader_lexicon')

# NLTK’s Pre-Trained Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


def get_compound_score(text: str) -> float:
    """Returns the compound score for a quotation.

    Args:
        text (str): text or quotation.

    Returns:
        float: compound score.
    """
    return sia.polarity_scores(text)['compound']


def add_col_compound_score(df: pd.DataFrame,
                           text_col: str = QUOTATION_COL) -> None:
    """Adds the column of compound score for sentiment analysis to a dataframe
    of quotes.

    Args:
        df (pd.DataFrame): dataframe.
        text_col (str, optional): name of the column containing quotations.
        Defaults to 'quotation'.
    """
    df[COMPOUND_SCORE_COL] = df[text_col].progress_apply(get_compound_score)
