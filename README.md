<h1 align="center">
Determining the political orientation of newspapers through quotes analysis
</h1>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4277311.svg)](https://doi.org/10.5281/zenodo.4277311)

## Team members

- Albrecht Alice
- Deschamps Quentin
- Juanico Alice
- Testa Laura

## Data story

You can find the data story on this [website](https://quentin18.github.io/newspapers/).
The repository associated is on this [link](https://github.com/Quentin18/newspapers).

## Structure of the repository

The analysis is divided into two notebooks:

- `project_pt1_loading.ipynb`: loading and selecting the data
- `project_pt2_analyses.ipynb`: analyses

All the functions used in the notebooks are implemented in the `src` directory,
which contains multiple modules for the different tasks of the analysis.

## Setup

We used [Google Colab](https://research.google.com/colaboratory/) to run the
notebooks. The different libraries used are listed in the `requirements.txt`
file and can be downloaded with the following command:
```
pip3 install -r requirements.txt
```

## Project

### Abstract

In the USA, the political orientation of American people is mostly divided into two main parties: the republicans and the democrats. The opinion of the population is obviously based on the media such CNN, FOX News or The New York Times. These famous American journals have to be neutral in order not to influence their readers but to report the news well. Here, we will use the quotebank dataset containing a set of quotations published in the newspapers between 2015 and 2020 to verify this statement. Using different techniques such as sentiment analysis and Principal Component Analysis, we will highlight the separation between pro-democratic and pro-republican in the quotations and their speakers and check whether one political orientation is more strongly represented than the other one. The determination of the political orientation will be based on sentiment analysis about several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue.

### Research questions

Our project should answer the following questions:
- Who are the main speakers quoted in the newspapers and from what political party are they?
- What is the opinion of newspapers on big subjects and debated topics? Are these opinions drifting towards one political party?
- Are there any specific topics that are frequently addressed by newspapers? Are those topics relevant to define the newspapers' political opinion?

### Additional datasets

The parquet provided from the ADA course containing additional information on speakers

### Methods

The determination of the political orientation of the journals is based on several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue. Therefore, we chose mulitple topics and use Empath to generate a dictionnary from given seed terms. The dictionaries will help to select quotations that comment on a specific topic. Here is the exhaustive list of reviewed topics:
- immigration
- healthcare
- climate
- trump
- abortion
- women right
- violence
- racism
- war
- tax
- coal

Then for each journal:
1. Determine who are the main speakers and the proportion given by the journal to each party 
2. Perform a sentiment analysis on each of the selected quotations to obtain a quote score vector for the selected set of topics
3. Analyze the mean sentiment scores per topic for each party, and determine if the opinion on the topic really significantly differs from one party to another. 
4. Analyze the mean sentiment score per topic of the entire journal
5. PCA visualization for speakers quotes: visualize the difference of opinions between the 2 parties

Used libraries (for now): `spaCy`, `NLTK`, `WordCloud`, `Pandas`, `NumPy`, `Matplotlib`

### Organization

See [timeline](timeline.md).

## References

See [references](references.md)
