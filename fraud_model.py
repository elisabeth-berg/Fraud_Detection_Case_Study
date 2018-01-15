import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

class FraudModel(object):
    def __init__(self, alpha=0.1, n_jobs=-1, max_features='sqrt', n_estimators=1000):
        """
        INPUT:
        - alpha = Additive laplace smoothing parameter for NaiveBayes
        - n_jobs = Number of jobs to run RFC on
        - max_features = Number of featres to consider on RFC
        - n_estimators = Number of trees in RFC

        ATTRIBUTES:
        - RFC = Random Forest Classifier
        - MNB = Multinomial Naive Bayes Classifier
        """
        self.RFC = RandomForestClassifier(n_jobs=n_jobs, max_features=max_features,
                                            n_estimators=n_estimators)
        self.MNB = MultinomialNB(alpha=alpha)

    def fit(self, X, y):
        """
        INPUT:
        - X: dataframe representing feature matrix for training data
        - y: series representing labels for training data
        """
        # Random Forest
        self.RFC.fit(X, y)

        # NLP

    def predict_proba(self, X):
        """
        INPUT:
        - X: dataframe representing feature matrix for data

        OUTPUT:
        - blah
        """
        RFC_preds = self.RFC.predict_proba(X)

    def score(self, X, y):
        """
        INPUT:
        - X: dataframe representing feature matrix for testing data
        - y: series representing labels for testing data

        OUTPUT:
        - blah
        """
        self.predict_proba(X)

    def _weighted_probas(self, arr):
        """
        INPUT:
        - arr: np array of probabilities of classes

        OUTPUT:
        - weighted_arr: weighted "importance" of probabilities
        """
        pass
