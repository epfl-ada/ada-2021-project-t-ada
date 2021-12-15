"""
Text processing functions.
"""
import numpy as np
import pandas as pd
import spacy
from empath import Empath
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.phrases import Phrases
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .constants import (BOW_COL, COMPOUND_SCORE_COL, QUOTATION_COL, TOKENS_COL,
                        TOPICS_COL, TOPICS_DICT)

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


def create_vocabulary_with_empath(
    topic_name: str,
    seed_words: list,
    size: int = 500,
) -> list:
    """Creates a vocabulary list about a topic using empath.

    Args:
        topic_name (str): name of the topic.
        seed_words (list): seed words to generate vocabulary.
        size (int, optional): number of generated words. Defaults to 500.

    Returns:
        list: list of words about the topic.
    """
    lexicon = Empath()
    lexicon.create_category(topic_name, seed_words, model='nytimes', size=size)
    vocabulary = lexicon.cats[topic_name]
    return vocabulary


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


def create_lexicon(topics_dict: dict = TOPICS_DICT, size: int = 500) -> Empath:
    """Creates a lexicon with empath from a dictionary of topics and seed
    words.

    Args:
        topics_dict (dict, optional): dictionary of topics and seed words.
        Defaults to TOPICS_DICT.
        size (int, optional): number of generated words. Defaults to 500.

    Returns:
        Empath: lexicon.
    """
    lexicon = Empath()
    for topic_name, seed_words in topics_dict.items():
        print('Topic:', topic_name)
        lexicon.create_category(
            topic_name, seed_words, model='nytimes', size=size
        )
    return lexicon


def get_topics_list(tokens: list, lexicon: Empath, categories: list) -> list:
    """Returns the list of topics from a list of tokens and a lexicon.

    Args:
        tokens (list): tokens list.
        lexicon (Empath): lexicon.
        categories (list): list of topics.

    Returns:
        list: list of topics.
    """
    result = lexicon.analyze(tokens, categories=categories)
    return [topic for topic in result if result[topic] > 0]


def add_topics_col(
    df: pd.DataFrame,
    lexicon: Empath,
    categories: list,
) -> None:
    """Adds the column of topics to a dataframe of quotes. It corresponds to
    a list of topics about a quote.

    Args:
        df (pd.DataFrame): dataframe with `tokens` column.
        lexicon (Empath): lexicon from Empath.
        categories (list): list of topics.
    """
    assert TOKENS_COL in df.columns

    df[TOPICS_COL] = df[TOKENS_COL].progress_apply(
        lambda x: get_topics_list(x, lexicon, categories)
    )


def create_df_topics(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """Creates the dataframe of topics from a dataframe of quotes.

    It contains one column per topic, one row per quote. The cell for quote q
    and topic t contains the compound score if t is in the topics associated to
    the quote, Nan otherwise.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        categories (list): list of topics.

    Returns:
        pd.DataFrame: dataframe of topics.
    """
    assert COMPOUND_SCORE_COL in df.columns and TOPICS_COL in df.columns

    # Init dataframe
    df_topics = pd.DataFrame()

    # Create each column
    for topic in tqdm(categories, desc='Create df topics', unit='topic'):
        colname = f"{topic.replace(' ', '_')}_{COMPOUND_SCORE_COL}"
        df_topics[colname] = df.apply(
            lambda x: x[COMPOUND_SCORE_COL] if topic in x[TOPICS_COL]
            else np.nan,
            axis=1
        )

    return df_topics
