from util import *

def extract_features(text):
    sia = SentimentIntensityAnalyzer()
    top_100_positive = open("top_100_positive.pkl", "rb")
    top_100_negative = open("top_100_negative.pkl", "rb")
    poswordcount = 0
    negwordcount = 0
    compound_scores = list()
    positive_scores = list()
    negative_scores = list()
    neutral_scores = list()
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
              poswordcount += 1
            elif word.lower() in top_100_negative:
              negwordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])
        # negative_scores.append(sia.polarity_scores(sentence)["neg"])
        # neutral_scores.append(sia.polarity_scores(sentence)["neu"])
    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features=np.zeros((1,4))
    features[0][0]=mean(compound_scores) + 1
    features[0][1]=mean(positive_scores)
    features[0][2]=poswordcount
    features[0][3]=negwordcount
    return features