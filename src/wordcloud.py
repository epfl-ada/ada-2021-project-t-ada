"""
Functions to create wordclouds.
"""
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS

from .constants import TOKENS_COL

# Init progress bar
tqdm.pandas()

# Update stopwords with words that are strongly present in both parties
STOPWORDS.update([
    'will', 'people', 'im', 'going', 'president', 'one', 'dont', 'know',
    'want', 'time', 'say', 'go', 'think', 'make', 'see', 'thing', 'need',
    'work', 'way', 'now', 'come', 'well', 'good', 'lot', 'much', 'something',
    'really', 'look', 'take', 'tell'
])

# Define a colormap for the word clouds of each category
COLORDICT = {
    'democratic party': 'Blues',
    'republican party': 'Reds',
    'other party': 'Greens',
    'no party': 'Purples',
}


def join_tokens(df: pd.DataFrame) -> str:
    """Joins the tokenized quotations of a dataframe.

    Args:
        df (pd.DataFrame): dataframe of quotes.

    Returns:
        str: joined tokens.
    """
    return ' '.join(''.join(tokens) for tokens in df[TOKENS_COL])


def plot_wordcloud(
    wordcloud: WordCloud,
    filename: str = None,
    title: str = 'Wordcloud',
    figsize: tuple = (20, 10)
) -> None:
    """Plots a word cloud.

    Args:
        wordcloud (WordCloud): wordcloud object.
        filename (str, optional): path to save the wordcloud. Defaults to None.
        title (str, optional): title. Defaults to 'Wordcloud'.
        figsize (tuple, optional): figsize. Defaults to (20, 10).
    """
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.title(title, fontsize=20)
    plt.axis('off')
    if filename is not None:
        if filename.endswith('.svg'):
            wordcloud_svg = wordcloud.to_svg()
            with open(filename, 'w') as f:
                f.write(wordcloud_svg)
        else:
            wordcloud.to_file(filename)
    plt.show()


def create_wordcloud_party(df: pd.DataFrame, party_name: str) -> WordCloud:
    """Generates a word cloud for a party.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        party_name (str): party name.

    Returns:
        WordCloud: wordcloud object.
    """
    assert 'party_name' in df.columns

    df_party = df[df['party_name'] == party_name]

    # Join quotations
    text = join_tokens(df_party)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        random_state=105,
        background_color='white',
        colormap=COLORDICT[party_name],
        stopwords=STOPWORDS,
        contour_width=3,
        contour_color='black').generate(text)

    return wordcloud
