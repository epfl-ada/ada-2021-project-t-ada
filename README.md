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

In the USA, the political orientation of American people is mostly divided into two main parties: the republicans and the democrats. The famous New York Times newspaper claims to be neutral, presenting both democratic and republican opinions. Here, we will use the quotebank dataset containing a set of quotations published in the newspapers between 2015 and 2020 to verify this statement. Using different techniques such as sentiment analysis and Principal Component Analysis, we will highlight the separation between pro-democratic and pro-republican in the quotations and their speakers and check whether one political orientation is more strongly represented than the other one. The determination of the political orientation will be based on sentiment analysis about several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue.

### Research questions

Our project should answer the following questions:
- Who are the main speakers quoted in the journal and from what political party are they? Are they identified leaders?
- What is the opinion of the New York Times on big subjects and debated topics? Is its opinion drifting towards one political party?

### Additional datasets

1. The parquet provided from the ADA course containing additional information on speakers
2. Dictionaries on several common topics. The dictionaries will help to select quotations that comment on a specific topic. Here is a non-exhaustive list of the topics to review:
- immigration
- health care
- climate change
- gun control
- Trump
- abortion
- women’s rights
- racisms
- police violence
- war and military action
- taxes
- coal industry

These are topics on which the dictionaries will be based. Not all topics are necessarily going to be included in our analysis, only the ones that highlight the differences of political opinions between parties. We will probably have to eliminate some topics if there are too few quotations about them or if the sentiment analysis does not show any significant difference between the parties.

### Methods

Further steps to perform for the project:
- Obtain a quote score vector for the selected set of topics: use a pre-trained sentimental analysis model (or try to train the model with our data). The combination of all topics will help determinig the political opinions.
- PCA visualization for speakers quotes: visualize the difference of opinions between the 2 parties, to confirm a clear Republican/Democrat separation.
- Deduce the potential political bias of the New York Times: average all scores per topic to obtain a NYT score vector, compare this vector to the Democrat and Republican score vectors.
- Build a political network: search for the main speakers of the NYT and look at their political party. Link the speakers between them and build a network. 
- Conclude if one political party is more strongly represented in the NYT, using the political network.

Used libraries (for now): `spaCy`, `NLTK`, `WordCloud`, `Pandas`, `NumPy`, `Matplotlib`

### Organization

See [timeline](timeline.md).

## References

See [references](references.md)
