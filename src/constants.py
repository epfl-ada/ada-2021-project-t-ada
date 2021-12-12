"""
All constants.
"""
import os

# Paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)
TEST_FILENAME = 'quotes-2019-nytimes.json.bz2'
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data', TEST_FILENAME)

# Column names in dataframes
BOW_COL = 'bow'
COMPOUND_SCORE_COL = 'compound_score'
QID_COL = 'qid'
QUOTATION_COL = 'quotation'
SPEAKER_COL = 'speaker'
TOKENS_COL = 'tokens'

# Useless columns
USELESS_COLS = ['phase', 'probas']
