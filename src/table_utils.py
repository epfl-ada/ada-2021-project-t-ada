"""
Functions to make tables.

To write in a html file:
python3 table_utils.py > table.html
"""
from tabulate import tabulate

from constants import TOPICS_DICT


def make_html_table_topics():
    data = [
        [topic_name.capitalize(), ', '.join(seed_words)]
        for topic_name, seed_words in TOPICS_DICT.items()
    ]
    return tabulate(data, headers=['Topic', 'Seed words'], tablefmt='html')


if __name__ == '__main__':
    print(make_html_table_topics())
