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
Input: pd dataframe of raw features with no 'acct_type'.
Can be applied to training AND testing data.
"""


def delivery_method_categorize(data):
    data['delivery_method_0'] = data['delivery_method'] == 0
    data['delivery_method_1'] = data['delivery_method'] == 1
    data['delivery_method_3'] = data['delivery_method'] == 3
    return data


def payout_type_categorize(data):
    data['payout_type_check'] = data['payout_type'] == 'CHECK'
    data['payout_type_ach'] = data['payout_type'] == 'ACH'
    return data


def currency_categorize(data):
    data['usd'] = data['currency'] == 'USD'
    data['gbp'] = data['currency'] == 'GBP'
    data['cad'] = data['currency'] == 'CAD'
    data['aud'] = data['currency'] == 'AUD'
    data['eur'] = data['currency'] == 'EUR'
    data['nzd'] = data['currency'] == 'NZD'
    return data


def user_type_categorize(data):
    data['user_type_1'] = data['user_type'] == 1
    data['user_type_2'] = data['user_type'] == 2
    data['user_type_3'] = data['user_type'] == 3
    data['user_type_4'] = data['user_type'] == 4
    data['user_type_5'] = data['user_type'] == 5
    return data


def email_categorize(data):
    """
    Define a "rare_email" domain as one that occurs one or zero times within the
    training data.
    """
    emails = pd.DataFrame(data['email_domain'].value_counts() <= 1)
    emails['rare_email'] = emails['email_domain']
    common_emails = emails.index[emails['rare_email'] == False]
    data['rare_email'] = [domain not in common_emails for domain in data['email_domain']]
    return data


def event_data(data):
    """
    Calculate event duration from event end and start timestamps.
    """
    data['event_duration'] = data['event_end'] - data['event_start']
    return data


def listed_categorize(data):
    """
    Categorizes the 'listed' column in the pandas dataframe.

    INPUT:
        - data: pandas dataframe with 'listed' column as 'y' or 'n'

    OUTPUT:
        - data: pandas dataframe with 'listed' column replaced with booleans
    """
    data['listed'] = data['listed'] == 'y'
    return data


def country_data(data):
    """
    Takes a pandas dataframe and does some undetermined stuff with the countries

    INPUT:
        - data: pandas dataframe to get country data from and add engineered
                columns to.

    OUTPUT:
        - data: pandas dataframe with engineered country features added.
    """
    data['venue_country_change'] = (data['venue_country'] != data['country'])
    data['is_us'] = data['country'] == 'US'
    data['is_gb'] = data['country'] == 'GB'
    data['is_ca'] = data['country'] == 'CA'
    return data


def categorize_descriptions(data, event_desc):
    """
    Takes data as a pandas data_frame made of one column: event_description,  the descriptions are categorized and returned as pandas dataframe with 10 columns representing each data_point as a row and the euclidean distance to its respective cluster center.
    """
    descr_means,  descr_tfidf = load_k_means_model_objects()
    new_df = update_data_frame(event_desc)
    desc_counts_tfidf = vecrtorize_new_data(new_df)
    desc_matrix = desc_counts_tfidf.todense()
    descr_centers = cluster_centers()

    dist_one = np.apply_along_axis(distance_to_cluster_one, 1, desc_matrix)
    dist_two = np.apply_along_axis(distance_to_cluster_two, 1, desc_matrix)
    dist_three = np.apply_along_axis(distance_to_cluster_three, 1, desc_matrix)
    dist_four = np.apply_along_axis(distance_to_cluster_four, 1, desc_matrix)
    dist_five = np.apply_along_axis(distance_to_cluster_five, 1, desc_matrix)

    cluster_descr_df['cluster_one'] = dist_one
    cluster_descr_df['cluster_two'] = dist_two
    cluster_descr_df['cluster_three'] = dist_three
    cluster_descr_df['cluster_four'] = dist_four
    cluster_descr_df['cluster_five'] = dist_five
    return cluster_descr_df


def soupify(html):
    """ INPUT html
        OUTPUT string"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(strip=True)


def update_data_frame(data):
    """ INPUT df with html
    OUTPUT new_df with cluster columns"""
    new_df = data.copy()
    new_df['description_no_HTML'] = new_df['description'].apply(soupify)
    #new_df['org_desc_no_HTML'] = new_df['org_desc'].apply(soupify)
    return new_df


def vectorize_text_descriptons(new_df):
    """INPUT df
        OUTPUT sparse matrix as cols"""
    vect_tfidf = TfidfVectorizer(stop_words='english', tokenizer=bag_of_words)
    words_counts_tfidf = vect_tfidf.fit_transform(new_df['description_no_HTML'])
    return words_counts_tfidf


def extract_bow_from_raw_text(text_as_string):
    """Extracts bag-of-words from a raw text string.
    Parameters with chunk sequencing of proper nouns
    ----------
    text (str): a text document given as a string
    Returns
    -------
    list : the list of the tokens extracted and filtered from the text,
    """
    if (text_as_string == None):
        return []

    if (len(text_as_string) < 1):
        return []

    nfkd_form = unicodedata.normalize('NFKD', text_as_string)
    text_input = str(nfkd_form.encode('ASCII', 'ignore'))

    sent_tokens = sent_tokenize(text_input)

    tokens = list(map(word_tokenize, sent_tokens))

    sent_tags = list(map(pos_tag, tokens))

    grammar = r"""
        SENT: {<(J|N).*>}
    """

    cp = RegexpParser(grammar)
    ret_tokens = list()
    stemmer_snowball = SnowballStemmer('english')

    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                t_tokenlist = [tpos[0].lower() for tpos in subtree.leaves()]
                t_tokens_stemsnowball = list(map(stemmer_snowball.stem, t_tokenlist))
                ret_tokens.extend(t_tokens_stemsnowball)

    return(ret_tokens)


def bag_of_words(col_name):
    """INPUT col_name
    OUTPUT list of strings of bag of words for each description"""
    bow = extract_bow_from_raw_text(col_name)
    return bow


def load_k_means_model_objects():
    descr_means = pickle.load(open("kmeans_desc_modelsgit ", "rb"))

    desc_tfidf = pickle.load(open("desc_vectorizer_final.pkl", "rb"))

    return descr_means, desc_tfidf


def vecrtorize_new_data(data):
    """return tfidf transformed to new data"""
    descr_counts_tfidf = descr_tfidf.transform(data['description_no_HTML'])
    return descr_counts_tfidf


def cluster_centers():
    """return cluster centers for descriptions """
    descr_means = pickle.load(open("kmeans_desc_models.pkl", "rb"))
    descr_centers = descr_means.cluster_centers_
    return descr_centers


def distance_to_cluster(row, cluster_center):
    """ INPUT row of df
        OUTPUT data_point distance to cluster_center """
    return scs.spatial.distance.euclidean(row, cluster_center)


def distance_to_cluster_one(row):
    descr_centers = cluster_centers()
    cluster_center = descr_centers[0]
    return distance_to_cluster(row, cluster_center)


def distance_to_cluster_two(row):
    descr_centers = cluster_centers()
    cluster_center = descr_centers[1]
    return distance_to_cluster(row, cluster_center)


def distance_to_cluster_three(row):
    descr_centers = cluster_centers()
    cluster_center = descr_centers[2]
    return distance_to_cluster(row, cluster_center)


def distance_to_cluster_four(row):
    descr_centers = cluster_centers()
    cluster_center = descr_centers[3]
    return distance_to_cluster(row, cluster_center)


def distance_to_cluster_five(row):
    descr_centers = cluster_centers()
    cluster_center = descr_centers[4]
    return distance_to_cluster(row, cluster_center)


def pop_descriptions(data):
    """
    Takes a pandas dataframe, removes the 'org_descr' and 'discription' columns
        and returns the two descriptions and the new data frame.

    INPUT:
        - data: pandas data frame to remove the descriptions from

    OUTPUT:
        - data: original pandas df with descriptions removed
        - event_description: pandas series of the event descriptions
    """
    data.drop(columns=['org_desc'])
    event_description = data.pop(item='description')

    return data, event_description


def final_columns(data):
    wanted_columns = ['delivery_method_0', 'delivery_method_1', 'delivery_method_3',
                      'payout_type_check', 'payout_type_ach', 'usd', 'gbp', 'cad',
                      'aud', 'eur', 'nzd', 'user_type_1', 'user_type_2', 'user_type_3',
                      'user_type_4', 'user_type_5', 'rare_email', 'event_duration',
                      'listed', 'venue_country_change','is_us', 'is_gb', 'is_ca',
                      'cluster_one', 'cluster_two', 'cluster_three', 'cluster_four',
                      'cluster_five', 'body_length', 'channels', 'fb_published',
                      'has_analytics', 'has_logo', 'listed', 'name_length', 'show_map',
                      'user_age']
    data = data[wanted_columns]
    return data


def clean_data(data):
    """
    Cleans the entire data set.

    INPUT:
        - data:
    """
    clean_data, event_description = pop_descriptions(data)
    clean_data = country_data(data)
    clean_data = listed_categorize(data)
    clean_data = event_data(data)
    clean_data = email_categorize(data)
    clean_data = user_type_categorize(data)
    clean_data = currency_categorize(data)
    clean_data = payout_type_categorize(data)

    clean_data = final_columns(data)

    return clean_data
