"""
All constants.
"""
# Column names in dataframes
BOW_COL = 'bow'
COMPOUND_SCORE_COL = 'compound_score'
QID_COL = 'qid'
QIDS_COL = 'qids'
QUOTATION_COL = 'quotation'
SPEAKER_COL = 'speaker'
TOKENS_COL = 'tokens'
TOPICS_COL = 'topics'

# Useless columns
USELESS_COLS = ['phase', 'probas', 'urls']

# Columns to keep in parquet files
SPEAKER_COLUMNS = [
    'aliases', 'id', 'nationality', 'US_congress_bio_ID', 'party', 'label'
]

# Usefull QIDS
QID = {
    'us': 'Q30',
    'democrat': 'Q29552',
    'republican': 'Q29468',
}

# List of parties
PARTIES_LIST = [
    'democratic party',
    'republican party',
    'other party',
    'no party',
]

# Dictionnary of topics and seed terms
TOPICS_DICT = {
    'immigration': [
        'refugee',
        'immigration',
        'border',
        'citizenship',
        'naturalization',
    ],
    'healthcare': [
        'health',
        'medical',
        'treatment',
        'disease',
        'aid',
        'hospital',
        'insurance',
        'reimbursement',
    ],
    'climate': [
        'melting',
        'global warming',
        'temperature',
        'rise',
        'change',
        'ecology',
        'meteorology',
        'urgency',
        'co2',
        'greenhouse gas',
        'climate event',
    ],
    'trump': [
        'president',
        'donald trump',
        'republican',
        '2016 presidential election',
    ],
    'abortion': [
        'pregnancy',
        'woman',
        'life',
        'choice',
        'family',
        'child',
        'foetus',
        'body',
        'right',
        'terminate',
        'abort',
        'rape',
    ],
    'women right': [
        'abortion',
        'sexism',
        'salary gap',
        'sexual harassment',
        'abuse',
        'gender equality',
        'gender',
        'woman',
        'female',
        'patriarchy',
        'feminism',
    ],
    'violence': [
        'police violence',
        'gun',
        'second amendment',
        'shooting',
        'death',
        'police brutality',
        'firearm',
    ],
    'racism': [
        'discrimination',
        'privilage',
        'race',
        'ethnicity',
        'equality',
        'afroamerican',
        'white',
        'black',
        'hate crime',
        'color',
    ],
    'war': [
        'military',
        'irak',
        'afghanistan',
        'palestine',
        'middle east',
        'soldier',
        'arm',
        'weapon',
        'missile',
        'conflict',
        'operation',
        'troop',
        'bomb',
        'force',
    ],
    'tax': [
        'income',
        'revenue',
        'free trade',
        'taxpayer',
        'imposition',
        'fee',
        'social welfare',
        'tax evasion',
        'tariff',
        'deductible',
        'vat',
    ],
    'coal': [
        'energy',
        'pollution',
        'mine',
        'industry',
        'fossil fuel',
        'electricity',
        'carbon',
    ],
}
