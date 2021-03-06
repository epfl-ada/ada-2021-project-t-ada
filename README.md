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

In the USA, the political orientation of American people is mostly divided into two main parties: the republicans and the democrats. The opinion of the population is obviously based on the media such as *CNN*, *FOX News* or *The New York Times*. These famous American newspapers have to be neutral in order not to influence their readers but to report the news well.

For this project, we use the [Quotebank](https://doi.org/10.5281/zenodo.4277311) dataset containing a set of quotations published in the newspapers between 2015 and 2020 to verify this statement. Using different techniques such as sentiment analysis and Principal Component Analysis, we highlight the separation between pro-democratic and pro-republican in the quotations and their speakers and check whether one political orientation is more strongly represented than the other one. The determination of the political orientation is based on sentiment analysis about several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue.

### Research questions

Our project answers the following questions:
- Who are the main speakers quoted in the newspapers and from which political party are they?
- What is the opinion of newspapers on big subjects and debated topics? Are these opinions drifting towards one political party?
- Are there any specific topics that are frequently addressed by newspapers? Are those topics relevant to define the newspapers' political opinion?

### Additional datasets

To obtain more informations about the speakers, we use the parquet files provided from the ADA course containing informations from [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).

### Methods

The determination of the political orientation of the newspapers is based on several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue. Therefore, we chose mulitple topics and use Empath to generate a dictionnary from given seed terms. The dictionaries help to select quotations that comment on a specific topic. Here is the exhaustive list of reviewed topics:
- Immigration
- Healthcare
- Climate
- Trump
- Abortion
- Women right
- Violence
- Racism
- War
- Tax
- Coal

Then for each newspaper:
1. Determine who are the main speakers and the proportion given by the newspaper to each party
2. Perform a sentiment analysis on each of the selected quotations to obtain a quote score vector for the selected set of topics
3. Analyze the mean sentiment scores per topic for each party, and determine if the opinion on the topic really significantly differs from one party to another.
4. Analyze the mean sentiment score per topic of the entire newspaper
5. PCA visualization for speakers quotes: difference of opinions between the 2 parties

### Organization

Quentin Deschamps:
- Creation of the GitHub website
- Perform sentiment analysis on the whole data using the different dictionaries, in order to score all the quotes per topic, obtaining a score vector per quotation.
- Improved code writing and final organization of the GitHub

Alice Juanico:
- For each party, compute from the set of quotations scores a mean score vector to determine the opinion (positive, negative, neutral) of the party on each topic. Check if the score distributions are statistically different between both parties (statistical analysis)
- Visualization coding part to obtain the interactive graphs for the website
- Improved the data story about interactive graph

Laura Testa:
- Preprocessing of the data and tokenizer to obtain analyzable quotes
- Perform a PCA to visualize the differences of opinion between Democrats and Republicans (using the sentiment analysis results)
- Writing the data story for the website and handle the presentation

Alice Albrecht:
- Loading the complete data for the 3 newspapers
- Create a dictionary for every topic using Empath, use them to filter our data to check if there are enough quotations per topics
- README and harmonization of the documentations

## References

See [references](references.md)
