"""
Functions to make tables.

To write in a html file:
python3 table_utils.py > table.html
"""
from tabulate import tabulate

from .constants import TOPICS_DICT


def make_html_table_topics(topics_dict: dict = TOPICS_DICT) -> str:
    """Makes a html table with the topics and the seed words.

    Args:
        topics_dict (dict, optional): topics and associated seed words.
        Defaults to TOPICS_DICT.

    Returns:
        str: html table with topics and seed words.
    """
    data = [
        [topic_name.capitalize(), ', '.join(seed_words)]
        for topic_name, seed_words in topics_dict.items()
    ]
    return tabulate(data, headers=['Topic', 'Seed words'], tablefmt='html')


if __name__ == '__main__':
    print(make_html_table_topics())
