# Determining the political orientation of the New York Times newspaper through quotes analysis
Albrecht Alice, Deschamps Quentin, Juanico Alice, Testa Laura


*Note: the code for milestone 2 is divided into two notebooks: `project_pt1_loading.ipynb` and `project_pt2_analyses.ipynb`. This choice is explained in the first notebook.*

### Abstract:
In the USA, the political orientation of American people is mostly divided into two main parties: the republicans and the democrats. The famous New York Times newspaper claims to be neutral, presenting both democratic and republican opinions. Here, we will use the quotebank dataset containing a set of quotations published in the newspapers between 2015 and 2020 to verify this statement. Using different techniques such as sentiment analysis and Principal Component Analysis, we will highlight the separation between pro-democratic and pro-republican in the quotations and their speakers and check whether one political orientation is more strongly represented than the other one. The determination of the political orientation will be based on sentiment analysis about several chosen topics that are commonly addressed in the USA and on which republicans and democrats tend to argue.

### Research questions:
Our project should answer the following questions:
- Who are the main speakers quoted in the journal and from what political party are they? Are they identified leaders?
- What is the opinion of the New York Times on big subjects and debated topics? Is its opinion drifting towards one political party?

### Additional datasets:
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

### Methods:
Further steps to perform for the project:
- Obtain a quote score vector for the selected set of topics: use a pre-trained sentimental analysis model (or try to train the model with our data). The combination of all topics will help determinig the political opinions.
- PCA visualization for speakers quotes: visualize the difference of opinions between the 2 parties, to confirm a clear Republican/Democrat separation.
- Deduce the potential political bias of the New York Times: average all scores per topic to obtain a NYT score vector, compare this vector to the Democrat and Republican score vectors.
- Build a political network: search for the main speakers of the NYT and look at their political party. Link the speakers between them and build a network. 
- Conclude if one political party is more strongly represented in the NYT, using the political network.

Used libraries (for now): `spaCy`, `NLTK`, `WordCloud`, `Pandas`, `NumPy`

### Proposed timeline & Organization within the team:

| Task                                                 | Dates                                            | Assigned To
| -----------------------------------------------------| ------------------------------------------------ | ------------------------------------------------
| Homework 2                                           | 12/11 → 26/11                                    | ALL
| Find/create a dictionary for every topic, use them to filter our data to check if there are enough quotations per topics  | 27/11 → 28/11  |ALBRECHT Alice, JUANICO Alice, TESTA Laura
| Perform sentiment analysis on the whole data using the different dictionaries, in order to score all the quotes per topic, obtaining a score vector per quotation.   | 27/11 → 30/12 | DESCHAMPS Quentin
| Write the Data Story on the sentimental analysis part, making the mathematical part more visual. | 30/12 → 02/12 | ALBRECHT Alice
| Perform a PCA to visualize the differences of opinion between Democrats and Republicans (using the sentiment analysis results). | 01/12 → 03/12 | TESTA Laura 
| Use the entire set of score vectors to determine the opinion (positive, negative, neutral) of the newspaper on each topic. Create visual outputs. | 01/12 → 05/12 | JUANICO Alice
| For each party, compute from the set of quotations scores a mean score vector to determine the opinion (positive, negative, neutral) of the party on each topic. Check if the score distributions are statistically different between both parties (statistical analysis). | 01/12 → 05/12 | JUANICO Alice
| Compare the opinion of the magazine and political parties to deduce the potential political bias of the NYT. | 05/12 → 07/12 | ALBRECHT Alice
| Write the Data Story on the topics opinions and conclusions about the NYT. | 07/12 → 08/12 | JUANICO Alice
| Examine the link between the political speakers to deduce a political network. | 04/12 → 07/12 | DESCHAMPS Quentin, TESTA Laura
| Write the Data Story on the political network. | 08/12 → 09/12 | DESCHAMPS Quentin
| Harmonize the Jupyter notebook and other documentations. | 10/12 → 12/12 | ALBRECHT Alice, TESTA Laura
| Possible extension of the project. | 11/12 → 13/12 | DESCHAMPS Quentin, JUANICO Alice
| Write the Data Story on the possible extension of the project. | 14/12 → 15/12 | JUANICO Alice
| Write the README. | 13/12 → 16/12 | DESCHAMPS Quentin, TESTA Laura
| Complete and perfect the Data Story. | 14/12 → 16/12 | ALBRECHT Alice
| Check everything, rerun the entire code, reread the documentation (Jupyter notebook, Readme, Data Story), coherence in the visualization, etc. | 17/12 | ALL



 
### Questions for TAs:
- Is our strategy sufficient for assessing the political orientation of the newspaper or should we do additional analyses?
- Do you know any platform that regroups complete dictionaries?
- Do you think it is a good idea to train the model on our data set instead of using a pre-trained model or is it unnecessary?


