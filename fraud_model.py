import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
from nlp import *
from data_cleanup import *

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
        # NLP
        desc_no_html = run_nlp(X)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        word_counts = self.tfidf.fit_transform(desc_no_html)

        # K-means
        desc_kmeans = KMeans(n_clusters=5, random_state=56, n_jobs=-1)
        desc_kmeans.fit(word_counts)
        self.cluster_centers = desc_kmeans.cluster_centers_
        X_cluster = compute_cluster_distance(word_counts, self.cluster_centers)

        RF_X = X.merge(X_cluster)

        # Random Forest
        self.RFC.fit(RF_X, y)

        # Naive Bayes
        self.MNB.fit(word_counts, y)


    def predict_proba(self, X):
        """
        INPUT:
        - X: dataframe representing feature matrix for data

        OUTPUT:
        - blah
        """
        desc_no_html = run_nlp(X)
        word_counts = self.tfidf.transform(desc_no_html)
        X_cluster = compute_cluster_distance(word_counts, self.cluster_centers)

        RF_X = X.merge(X_cluster)

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


def get_data(datafile):
    df = pd.read_json(datafile)
    # clean X data
    y = _get_labels(df)
    return X, y

def _get_labels(df):
    acc_type_dict = {'fraudster': 'fraud',
                 'fraudster_att': 'fraud',
                 'fraudster_event': 'fraud',
                 'premium': 'premium',
                 'spammer': 'spam',
                 'spammer_limited': 'spam',
                 'spammer_noinvite': 'spam',
                 'spammer_warn': 'spam',
                 'spammer_web': 'spam',
                 'tos_lock': 'tos',
                 'tos_warn': 'tos',
                 'locked': 'tos'}

    df['acct_label'] = df['acct_type'].map(acc_type_dict)
    return df['acct_label']

if __name__ == '__main__':
    train_X, train_y = get_data('data/train_data.json')
    fraud_model = FraudModel()
    fraud_model.fit(train_X, train_y)
    with open('fraud_model.pkl', 'w') as f:
        # Write the model to a file.
        pickle.dump(fraud_model, f)
