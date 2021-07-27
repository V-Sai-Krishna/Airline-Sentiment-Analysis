import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import re
import nltk.data
from nltk.corpus import stopwords, wordnet
from math import log
import string
from statistics import mean
from random import shuffle
from nltk.sentiment import SentimentIntensityAnalyzer
import pkg_resources
from symspellpy import SymSpell, Verbosity
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
import pickle
nltk.download([
"stopwords",
"vader_lexicon",
"punkt",
'wordnet'
])