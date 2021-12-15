"""
Sentiment analysis functions.
"""
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tqdm import tqdm

from .constants import COMPOUND_SCORE_COL, PARTY_NAME_COL, QUOTATION_COL

pd.options.mode.chained_assignment = None

# Init progress bar
tqdm.pandas()

# Download lexicon
nltk.download('vader_lexicon')

# NLTK’s Pre-Trained Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


def get_polarity_scores(text) -> dict:
    """Returns the polarity scores for a quotation.

    Args:
        text (str): text or quotation.

    Returns:
        dict: polarity scores (compound, neg, neu, pos).
    """
    return sia.polarity_scores(text)


def get_compound_score(text: str) -> float:
    """Returns the compound score for a quotation.

    Args:
        text (str): text or quotation.

    Returns:
        float: compound score.
    """
    return get_polarity_scores(text)['compound']


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


def run_ttest(df: pd.DataFrame, alpha: float = 0.05) -> str:
    """Runs ttest on a dataframe of topics and returns a table with the results
    per topic between democratic and republican parties in html format.

    Args:
        df (pd.DataFrame): dataframe of topics.
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        str: table of results in html format.
    """
    # Create dataframe per party
    df_democrats = df[df[PARTY_NAME_COL] == 'democratic party']
    df_republicans = df[df[PARTY_NAME_COL] == 'republican party']

    # Run ttest on each topic
    headers = ['Topic', 't-statistic', 'p-value', 'Same opinion?']
    results = list()
    for topic in df.columns.drop([PARTY_NAME_COL, 'label']):
        # Get scores
        scores_democrats = df_democrats[topic].dropna()
        scores_republicans = df_republicans[topic].dropna()

        # Run ttest
        topic_name = topic.replace('_compound_score', '').capitalize()
        statistic, pvalue = ttest_ind(scores_democrats, scores_republicans)
        same_opinion = '✅' if pvalue > alpha else '❌'
        results.append([topic_name, statistic, pvalue, same_opinion])

    return tabulate(results, headers=headers, tablefmt='html', floatfmt='.4f')


def create_df_avg_compound_score(df_topics: pd.DataFrame) -> pd.DataFrame:
    """Creates the dataframe with the average of compound scores on each topic
    for democrats and republicans from a daframe of topics.

    Args:
        df_topics (pd.DataFrame): dataframe of topics.

    Returns:
        pd.DataFrame: dataframe with average of compound scores.
    """
    # Create dataframe with only democrats and republicans
    df_demo_repu = df_topics[
        (df_topics['party_name'] == 'democratic party') | (
            df_topics['party_name'] == 'republican party')
    ]

    # Average compound scores
    df_avg = df_demo_repu.groupby(['label', 'party_name']).mean()

    # Drop rows with all NaN and fill NaN by 0
    df_avg = df_avg.dropna(how='all')

    return df_avg


def pca_analysis(
    df: pd.DataFrame,
    n_components: int = 2,
    stardardize: bool = False,
) -> tuple:
    """Performs PCA on the dataframe and returns the PCA with the principal
    components stored in a dataframe.

    Args:
        df (pd.DataFrame): dataframe.
        n_components (int, optional): number of components. Defaults to 2.
        stardardize (bool, optional): stardardize the scores.
        Defaults to False.

    Returns:
        tuple: pca, dataframe with principal components
    """
    # Replace missing entries with 0
    df_clean = df.fillna(0)

    # Perform PCA
    pca = PCA(n_components)
    if stardardize:
        scaler = StandardScaler()
        data = pca.fit_transform(scaler.fit_transform(df_clean))
    else:
        data = pca.fit_transform(df_clean)

    df_pca = pd.DataFrame(data, index=df_clean.index)

    # Set columns names
    df_pca.columns = [f'PC{i + 1}' for i in range(n_components)]

    return pca, df_pca
