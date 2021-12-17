"""
Plot functions using plotly.
"""
import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .constants import COMPOUND_SCORE_COL, TOPICS


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
    assert 'party_name' in df.columns

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


def plot_hist_compound(
    df: pd.DataFrame,
    title: str = 'Distribution of compound score',
    filename: str = None,
):
    """Plots the distribution of the compound score in a dataframe.

    Args:
        df (pd.DataFrame): dataframe of quotes.
        title (str, optional): title.
        Defaults to 'Distribution of compound score'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """
    assert COMPOUND_SCORE_COL in df.columns

    fig = px.histogram(
        df, x=COMPOUND_SCORE_COL, nbins=20,
        labels={
            COMPOUND_SCORE_COL: 'Compound score',
            'count': 'Number of quotations'
        },
        title=title,
    )

    if filename is not None:
        fig.write_html(filename)

    return fig


def plot_scatter_pca(
    df: pd.DataFrame,
    title: str = 'PCA',
    filename: str = None,
):
    """Plots the results of a PCA from a dataframe.

    Args:
        df (pd.DataFrame): dataframe of PCA.
        title (str, optional): title. Defaults to 'PCA'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """
    if df.columns.size == 3:
        # 3 components
        fig = px.scatter_3d(
            df.reset_index(), x='PC1', y='PC2', z='PC3', color='party_name',
            hover_data=['label'], title=title,
        )
        fig.update_traces(marker=dict(size=3))
    else:
        # 2 components
        fig = px.scatter(
            df.reset_index(), x='PC1', y='PC2', color='party_name',
            hover_data=['label'], title=title,
        )

    if filename is not None:
        fig.write_html(filename)

    return fig


def plot_mean_sentiment_scores_per_party(
    df: pd.DataFrame,
    title: str = 'Mean score per topic for the sentiment analysis',
    filename: str = None,
):
    """Plots the results of the sentiment analysis, showing the difference of
    opinion score per political party.

    Args:
        df (pd.DataFrame): dataframe of sentiment analysis scores.
        title (str, optional): title. Defaults to 'Mean score per topic for the
        sentiment analysis'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """
    # Compute mean and std per party
    means_per_party = df.groupby('party_name').mean()
    std_per_party = df.groupby('party_name').std()

    # Create figure
    fig = go.Figure()

    # Add trace for democrats scores
    fig.add_trace(go.Bar(
        name='Democrats',
        x=TOPICS, y=means_per_party.loc["democratic party"],
        error_y=dict(
            type='data',
            array=std_per_party.loc["democratic party"].values,
        )
    ))

    # Add trace for republican scores
    fig.add_trace(go.Bar(
        name='Republicans',
        x=TOPICS, y=means_per_party.loc["republican party"],
        error_y=dict(
            type='data',
            array=std_per_party.loc["republican party"].values,
        )
    ))

    # Update the figure layouts and plot
    fig.update_layout(
        barmode='group', title=title,
        yaxis_title='Mean score', xaxis_title='Topics'
    )

    if filename is not None:
        fig.write_html(filename)

    return fig


def dropdown_menu_for_plot(df: pd.DataFrame):
    """Creates an interactive dropdown menu for a bar plot with all the topics
    of the sentiment analysis.

    Returns the dropdown variables and the corresponding data to be plotted.

    Args:
        df (pd.DataFrame): dataframe of sentiment analysis scores.
    """
    # Initialize variables
    buttons_list = list([])
    plotted_data = []
    i = 0

    # Define the colors for the graph
    c_map = matplotlib.cm.get_cmap(name='tab20', lut=11)
    color_list = [matplotlib.colors.rgb2hex(c_map(i)) for i in range(c_map.N)]

    for topic in TOPICS:
        # Data to be plotted
        plotted_data.append(go.Bar(
            name=topic,
            x=df.index.astype('string'), y=df[topic + '_compound_score'],
            marker_color=color_list[i]
        ))

        # Interactive button
        x_visible = len(TOPICS) * [False]
        x_visible[i] = True
        buttons_list.append(dict(
            label=topic,
            method='update',
            args=[
                {'visible': x_visible},
                {'title': f'Count of {topic}-related quotes over the years'},
            ]
        ))
        i += 1

    return plotted_data, buttons_list


def plot_topics_count_stacked(
    df: pd.DataFrame,
    journal_name: str = 'NYT',
    filename: str = None
):
    """Plots the count of quotes for each topic for a given dataframe.

    Args:
        df (pd.DataFrame): dataframe of quotes count per topic over the years.
        journal_name (str, optional): name of the journal. Defaults to 'NYT'.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """
    [plotted_data, buttons_list] = dropdown_menu_for_plot(df)

    # Create graph layout
    menus = list([dict(active=-1, buttons=buttons_list)])
    layout = dict(
        barmode='stack',
        title=f'Count of topic-related quotes over the years ({journal_name})',
        showlegend=False,
        yaxis_title='Normalized count',
        xaxis_title='Years',
        updatemenus=menus
    )

    # Plot the graph
    fig = go.Figure(data=plotted_data, layout=layout)

    if filename is not None:
        fig.write_html(filename)

    return fig


def plot_topics_R_vs_D(
    df_democrats: pd.DataFrame,
    df_republicans: pd.DataFrame,
    topic: str,
    filename: str = None
):
    """Plots the count of quotes over the years for Republicans and Democrats
    (bar chart), for a specific topic.

    Args:
        df_democrats (pd.DataFrame): dataframe of democrat speakers quotes
        count per topic over the years.
        df_republicans (pd.DataFrame): dataframe of republicans speakers quotes
        count per topic over the years.
        topic (str): name of the topic to plot.
        filename (str, optional): filename to save the figure.
        Defaults to None.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Democrats',
        x=df_democrats.index.astype('string'),
        y=df_democrats[topic + '_compound_score'],
    ))
    fig.add_trace(go.Bar(
        name='Republicans',
        x=df_republicans.index.astype('string'),
        y=df_republicans[topic + '_compound_score'],
    ))

    fig.update_layout(
        barmode='stack',
        title=f'Count of {topic}-related quotes over the years (R vs D)',
        yaxis_title='Normalized count',
        xaxis_title='Years'
    )

    if filename is not None:
        fig.write_html(filename)

    return fig
