"""
Text processing functions.
"""
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.phrases import Phrases
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .constants import BOW_COL, QUOTATION_COL, TOKENS_COL

pd.options.mode.chained_assignment = None

# Init progress bar
tqdm.pandas()

# Tokenizer and lemmatizer function
nlp = spacy.load('en_core_web_sm')

# Stopwords list
STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS


def clean_col_text(df: pd.DataFrame, text_col: str = QUOTATION_COL) -> None:
    """Preprocesses a text column in a dataframe:

    - Remove the line breaks
    - Remove the punctuation
    - Remove the capital letters

    Args:
        df (pd.DataFrame): dataframe.
        text_col (str, optional): name of the column. Defaults to 'quotation'.
    """
    # Remove line breaks
    df[text_col] = df[text_col].replace('\n', ' ')

    # Remove numbers
    df[text_col] = df[text_col].str.replace(r'\w*\d\w*', '', regex=True)

    # Remove punctuation
    df[text_col] = df[text_col].str.replace(r'[^\w\s]', '', regex=True)

    # Remove capital letters
    df[text_col] = df[text_col].str.lower()


def get_tokens(text: str) -> list:
    """Preprocessing of a quote for LDA. It returns the list of tokens for the
    text after filtering.

    Args:
        text (str): text or quotation.

    Returns:
        list: tokens.
    """
    doc = nlp(text)

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [
        token.lemma_ for token in doc if token.is_alpha and not token.is_stop
    ]

    # Remove common words from the stopword list and keep only words of length
    # 3 or more.
    doc = [token for token in doc if token not in STOPWORDS and len(token) > 2]

    return doc


def add_col_tokens(df: pd.DataFrame, text_col: str = QUOTATION_COL) -> None:
    """Adds the column of tokens to a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe.
        text_col (str, optional): name of the column containing quotations.
        Defaults to 'quotation'.
    """
    df[TOKENS_COL] = df[text_col].progress_apply(get_tokens)


def add_bigrams_to_list(tokens: list, bigrams: Phrases) -> list:
    """Returns the list of tokens extended with bigrams.

    Args:
        tokens (list): list of tokens.
        bigrams (Phrases): phrases object to get bigrams.

    Returns:
        list: tokens + bigrams.
    """
    for token in bigrams[tokens]:
        if '_' in token:
            # Token is a bigram, add to document.
            tokens.append(token)
    return tokens


def add_bigrams(df: pd.DataFrame) -> None:
    """Adds bigrams to the tokens list.

    Args:
        df (pd.DataFrame): dataframe with `tokens` column.
    """
    assert TOKENS_COL in df.columns

    # Add bigrams to docs (only ones that appear 15 times or more).
    bigrams = Phrases(df[TOKENS_COL], min_count=15)
    df[TOKENS_COL] = df[TOKENS_COL].progress_apply(
        lambda x: add_bigrams_to_list(x, bigrams)
    )


def create_dictionary_from_words(
    words: list,
    verbose: bool = False,
) -> Dictionary:
    """Creates a dictionary from a list of words.

    Args:
        words (list): list of words.
        verbose (bool, optional): True to print the dictionary.
        Defaults to False.

    Returns:
        Dictionary: Dictionary object.
    """
    dictionary = Dictionary(words)

    if verbose:
        print(dictionary)
        print('Tokens id:')
        print(dictionary.token2id)

    return dictionary


def create_dictionary_from_tokens_col(
    df: pd.DataFrame,
    min_wordcount: int = 5,
    max_freq: float = 0.5
) -> Dictionary:
    """Creates a dictionary from the `tokens` column.

    Args:
        df (pd.DataFrame): dataframe with `tokens` column.
        min_wordcount (int, optional): minimum word count allowed.
        Defaults to 5.
        max_freq (float, optional): maximum word frequency allowed.
        Defaults to 0.5.

    Returns:
        Dictionary: Dictionary object.
    """
    assert TOKENS_COL in df.columns

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(df[TOKENS_COL])

    # Filter out words that occur too frequently or too rarely.
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

    return dictionary


def add_col_bow(df: pd.DataFrame, dictionary: Dictionary) -> None:
    """Adds the column of Bag-of-words representation to a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe with `tokens` column.
        dictionary (Dictionary): dictionary.
    """
    assert TOKENS_COL in df.columns

    df[BOW_COL] = df[TOKENS_COL].progress_apply(dictionary.doc2bow)


def get_lda_model(df: pd.DataFrame, dictionary: Dictionary) -> LdaMulticore:
    """Returns the LdaMulticore object for a dataframe with a `bow` column.

    Args:
        df (pd.DataFrame): dataframe with `bow` column.
        dictionary (Dictionary): dictionary.

    Returns:
        LdaMulticore: LdaMulticore object.
    """
    assert BOW_COL in df.columns

    return LdaMulticore(
        corpus=df[BOW_COL],
        id2word=dictionary,
        random_state=0
    )


def get_tfidf_matrix(
    df: pd.DataFrame,
    text_col: str = QUOTATION_COL,
    vocabulary: list = None,
) -> csr_matrix:
    """Returns the TF-IDF matrix from quotations.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        text_col (str, optional): name of the column. Defaults to 'quotation'.
        vocabulary (list, optional): list of words to use. Defaults to None.

    Returns:
        csr_matrix: TF-IDF matrix.
    """
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_col])
    return tfidf_matrix


def reduce_tfidf_matrix(
    tfidf_matrix: csr_matrix,
    n_components: int = 2,
) -> tuple:
    """Reduces the TF-IDF matrix using a dimensionality reduction procedure.

    Args:
        tfidf_matrix (csr_matrix): TF-IDF matrix.
        n_components (int, optional): dimensionality of output. Defaults to 2.

    Returns:
        tuple: TF-IDF matrix reduced, explained variance ratio
    """
    truncatedSVD = TruncatedSVD(
        n_components=n_components, n_iter=10, random_state=0,
    )
    tfidf_matrix_reduced = truncatedSVD.fit_transform(tfidf_matrix)
    return tfidf_matrix_reduced, truncatedSVD.explained_variance_ratio_
