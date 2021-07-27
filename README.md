# Sentiment Analysis API
Details of the files are as follows:
1) EntHire.ipynb: Contains code for reading the data, preprocessing the data, and training different models.
2) enthirebestmodel.sav: The model that gave the best performance (Sklearn Adaboost classifier).
3) preprocessor.py: Contains functions necessary to preprocess test input (tokenization, stopword removal etc)
4) feature.py: Converts test input into features
5) util.py: Inports all necessary packages
6) symspell.pkl: Contains data (single and bi-gram dictionaries) for spelling correction
7) top_100_negative.pkl: Name of the file is sli misleading. This contains the top 1000 (not 100) negative words in the dataset.
8) top_100_positive.pkl: Name of the file is sli misleading. This contains the top 1000 (not 100) positive words in the dataset.
9) wordcost.pkl: Contains data necessary to split words without spaces using dynamic programming in preprocessor.py
10) words-by-frequency.txt: Contains data necessary to split words without spaces using dynamic programming in preprocessor.py.
11) main.py: Creates an API for Sentiment Analysis.<br/>
