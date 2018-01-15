import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import string
import unicodedata
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import nltk
import scipy as scs
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
import pickle

"""
Input: series of strings (event descriptions)
Can be applied to training AND testing data AND new data
"""

def soupify(html):
    """ INPUT html
        OUTPUT string"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)

def update_data_frame(data):
    """ INPUT df with html
    OUTPUT new_df with description, html removed"""
    new_df = data.copy()
    new_df['description_no_HTML'] = new_df['description'].apply(soupify)
    return new_df

def distance_to_cluster(row, cluster_center):
    """ INPUT row of df
        OUTPUT data_point distance to cluster_center """
    return scs.spatial.distance.euclidean(row, cluster_center)

def distance_to_cluster_one(row, centers):
    cluster_center = centers[0]
    return distance_to_cluster(row, cluster_center)

def distance_to_cluster_two(row, centers):
    cluster_center = centers[1]
    return distance_to_cluster(row, cluster_center)

def distance_to_cluster_three(row, centers):
    cluster_center = centers[2]
    return distance_to_cluster(row, cluster_center)

def distance_to_cluster_four(row, centers):
    cluster_center = centers[3]
    return distance_to_cluster(row, cluster_center)

def distance_to_cluster_five(row, centers):
    cluster_center = centers[4]
    return distance_to_cluster(row, cluster_center)

def compute_cluster_distance(word_counts, cluster_centers):
    desc_matrix = word_counts.todense()

    column_list = ['cluster_one', 'cluster_two', 'cluster_three', 'cluster_four', 'cluster_five']
    cluster_descr_df = pd.DataFrame(data=None, columns=column_list)

    dist_one = np.apply_along_axis(lambda x: distance_to_cluster_one(x, cluster_centers), 1, desc_matrix)
    dist_two = np.apply_along_axis(lambda x: distance_to_cluster_two(x, cluster_centers), 1, desc_matrix)
    dist_three = np.apply_along_axis(lambda x: distance_to_cluster_three(x, cluster_centers), 1, desc_matrix)
    dist_four = np.apply_along_axis(lambda x: distance_to_cluster_four(x, cluster_centers), 1, desc_matrix)
    dist_five = np.apply_along_axis(lambda x: distance_to_cluster_five(x, cluster_centers), 1, desc_matrix)
    dist_list = [dist_one, dist_two, dist_three, dist_four, dist_five]

    for dist, column in zip(dist_list, column_list):
        cluster_descr_df[column] = dist

    return cluster_descr_df
