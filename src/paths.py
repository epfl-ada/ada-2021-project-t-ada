"""
All paths.
"""
import os

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)

DATA_DIR = os.path.join(ROOT_DIR, 'data')
QUOTEBANK_DIR = os.path.join(ROOT_DIR, 'Quotebank')

TEST_FILENAME = 'quotes-2019-nytimes.json.bz2'
TEST_DATA_PATH = os.path.join(DATA_DIR, TEST_FILENAME)

SELECTED_DIR = os.path.join(DATA_DIR, 'selected')

CNN_DIR = os.path.join(DATA_DIR, 'CNN')
FOX_DIR = os.path.join(DATA_DIR, 'FOX')
NYT_DIR = os.path.join(DATA_DIR, 'NYT')

TOKENS_DIR = os.path.join(DATA_DIR, 'tokens')

CNN_TOKENS_PATH = os.path.join(TOKENS_DIR, 'CNN-tokenizer.json.bz2')
FOX_TOKENS_PATH = os.path.join(TOKENS_DIR, 'FOX-tokenizer.json.bz2')
NYT_TOKENS_PATH = os.path.join(TOKENS_DIR, 'NYT-tokenizer.json.bz2')

PARQUET_PATH = os.path.join(ROOT_DIR, 'speaker_attributes.parquet')
