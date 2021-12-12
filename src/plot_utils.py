"""
Plot functions using plotly.
"""
import pandas as pd
import plotly.express as px


def plot_bar_top_speakers(
    df: pd.DataFrame,
    title: str = 'Top speakers',
    filename: str = None,
    n: int = 10,
):
    """Plots the top speakers in a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        title (str, optional): title. Defaults to 'Top speakers'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
        n (int, optional): number of speakers. Defaults to 10.
    """
    assert 'label' in df.columns

    df_top_speakers = df['label'].value_counts().head(n).reset_index()
    df_top_speakers.columns = ['speaker', 'counts']

    fig = px.bar(
        df_top_speakers, x='speaker', y='counts',
        labels={'speaker': 'Speaker', 'counts': 'Number of quotations'},
        title=title,
    )

    if filename is not None:
        fig.write_html(filename)

    return fig


def plot_pie_top_speakers(
    df: pd.DataFrame,
    title: str = 'Top speakers',
    filename: str = None,
    n: int = 10,
):
    """Plots the top speakers in a dataframe of quotes.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        title (str, optional): title. Defaults to 'Top speakers'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
        n (int, optional): number of speakers. Defaults to 10.
    """
    assert 'label' in df.columns

    df_top_speakers = df['label'].value_counts().head(n).reset_index()
    df_top_speakers.columns = ['speaker', 'counts']

    fig = px.pie(
        df_top_speakers, names='speaker', values='counts',
        labels={'speaker': 'Speaker', 'counts': 'Number of quotations'},
        title=title,
    )

    if filename is not None:
        fig.write_html(filename)

    return fig


def plot_pie_parties(
    df: pd.DataFrame,
    title: str = 'Proportions of parties',
    filename: str = None,
):
    """Plots the proportion of parties represented in a dataframe.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        title (str, optional): title. Defaults to 'Proportions of parties'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """

    df_parties = df['party_name'].value_counts().reset_index()
    df_parties.columns = ['party', 'counts']
    df_parties['party'] = df_parties['party'].str.capitalize()

    fig = px.pie(
        df_parties, names='party', values='counts',
        labels={'party': 'Party', 'counts': 'Number of quotations'},
        title=title,
    )
    fig.update_traces(textinfo='percent+label')

    if filename is not None:
        fig.write_html(filename)

    return fig