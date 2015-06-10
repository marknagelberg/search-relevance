import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
import evaluation
import cPickle
#Imports taken from the porter stemmer code https://www.kaggle.com/duttaroy/crowdflower-search-relevance/porter-stemmer/run/11533
from nltk.stem.porter import *
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    data["query_tokens_in_title"] = 0.0
    data["query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_description"] = 0.0
    data["percent_query_tokens_in_title"] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "query_tokens_in_title", float(len(query.intersection(title)))/float(len(title)))
            data.set_value(i, "percent_query_tokens_in_title", float(len(query.intersection(title)))/float(len(query)))
        if len(description) > 0:
            data.set_value(i, "query_tokens_in_description", float(len(query.intersection(description)))/float(len(description)))
            data.set_value(i, "percent_query_tokens_in_description", float(len(query.intersection(description)))/float(len(query)))
        data.set_value(i, "query_length", len(query))
        data.set_value(i, "description_length", len(description))
        data.set_value(i, "title_length", len(title))
        exact_query_in_title = 0
        if row["query"].lower() in row["product_title"].lower():
            exact_query_in_title = 1
        data.set_value(i, "exact_query_in_title", exact_query_in_title)
        exact_query_in_description = 0
        if row["query"].lower() in row["product_description"].lower():
            exact_query_in_description = 1
        data.set_value(i, "exact_query_in_description", exact_query_in_description)

        translate_to = u''
        translate_table = dict((ord(char), translate_to) for char in u' -')
        q_space_removed = row["query"].lower().translate(translate_table)
        t_space_removed = row["product_title"].lower().translate(translate_table)
        d_space_removed = row["product_description"].lower().translate(translate_table)

        if q_space_removed in t_space_removed:
            data.set_value(i, "space_removed_q_in_t", 1)
        else:
            data.set_value(i, "space_removed_q_in_t", 0)

        if q_space_removed in d_space_removed:
            data.set_value(i, "space_removed_q_in_d", 1)
        else:
            data.set_value(i, "space_removed_q_in_d", 0)

def get_string_similarity(s1, s2):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    s1_tokens = set(x.lower() for x in token_pattern.findall(s1))
    s2_tokens = set(x.lower() for x in token_pattern.findall(s2))
    if len(s1_tokens.union(s2_tokens)) == 0:
        return 0
    else:
        return float(len(s1_tokens.intersection(s2_tokens)))/float(len(s1_tokens.union(s2_tokens)))

def get_weighted_description_relevance(group, row):
    '''
    Takes a group of a particular query and a row within that 
    group and returns the weighted median relevance,
    weighted according to how  "close" description is to  other 
    rows within the group
    '''
    weighted_rating = 0.0
    num_similarities = 0
    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:
            similarity = get_string_similarity(row['product_description'], group_row['product_description'])
            weighted_rating += group_row['median_relevance'] * similarity
            num_similarities += 1
    return weighted_rating/float(num_similarities)

def get_weighted_title_relevance(group, row):
    '''
    Takes a group of a particular query and a row within that 
    group and returns the weighted median relevance,
    weighted according to how  "close" title is to  other 
    rows within the group
    '''
    weighted_rating = 0.0
    num_similarities = 0
    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:
            similarity = get_string_similarity(row['product_title'], group_row['product_title'])
            weighted_rating += group_row['median_relevance'] * similarity
            num_similarities += 1
    return weighted_rating/float(num_similarities)


def get_closest_description_relevance(group, row):
    '''
    Takes a group of a particular query and a row within that 
    group and returns the median relevance of the "closest" description in other 
    rows within the group
    '''
    return_rating = 0
    min_similarity = 0
    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:
            similarity = get_string_similarity(row['product_description'], group_row['product_description'])
            if similarity > min_similarity:
                min_similarity = similarity
                return_rating = group_row['median_relevance']
    return return_rating

def get_closest_title_relevance(group, row):
    '''
    Takes a group of a particular query and a row within that 
    group and returns the median relevance of the "closest" title in other 
    rows within the group
    '''
    return_rating = 0
    min_similarity = 0
    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:
            similarity = get_string_similarity(row['product_title'], group_row['product_title'])
            if similarity > min_similarity:
                min_similarity = similarity
                return_rating = group_row['median_relevance']
    return return_rating


        
def extract_training_features(train, test):
    train_group = train.groupby('query')
    test["q_mean_of_training_relevance"] = 0.0
    test["q_median_of_training_relevance"] = 0.0
    test["closest_title_relevance"] = 0
    for i, row in train.iterrows():
        group = train_group.get_group(row["query"])
        q_mean = group["median_relevance"].mean()
        train.set_value(i, "q_mean_of_training_relevance", q_mean)
        test.loc[test["query"] == row["query"], "q_mean_of_training_relevance"] = q_mean

        q_median = group["median_relevance"].median()
        train.set_value(i, "q_median_of_training_relevance", q_median)
        test.loc[test["query"] == row["query"], "q_median_of_training_relevance"] = q_median

        closest_title_relevance = get_closest_title_relevance(group, row)
        train.set_value(i, "closest_title_relevance", closest_title_relevance)

        closest_description_relevance = get_closest_description_relevance(group, row)
        train.set_value(i, "closest_description_relevance", closest_description_relevance)

        weighted_title_relevance = get_weighted_title_relevance(group, row)
        train.set_value(i, "weighted_title_relevance", weighted_title_relevance)

        weighted_description_relevance = get_weighted_description_relevance(group, row)
        train.set_value(i, "weighted_description_relevance", weighted_description_relevance)

    for i, row in test.iterrows():
        group = train_group.get_group(row["query"])
        closest_title_relevance = get_closest_title_relevance(group, row)
        test.set_value(i, "closest_title_relevance", closest_title_relevance)

        closest_description_relevance = get_closest_description_relevance(group, row)
        test.set_value(i, "closest_description_relevance", closest_description_relevance)

        weighted_title_relevance = get_weighted_title_relevance(group, row)
        test.set_value(i, "weighted_title_relevance", weighted_title_relevance)

        weighted_description_relevance = get_weighted_description_relevance(group, row)
        test.set_value(i, "weighted_description_relevance", weighted_description_relevance)

    train["all_words"] = train["query"] + " " + train["product_title"] + " " + train["product_description"]
    test["all_words"] = test["query"] + " " + test["product_title"] + " " + test["product_description"]


def stem_data(data):

    stemmer = PorterStemmer()

    for i, row in data.iterrows():

        q=(" ").join([z for z in BeautifulSoup(row["query"]).get_text(" ").split(" ")])
        t = (" ").join([z for z in BeautifulSoup(row["product_title"]).get_text(" ").split(" ")]) 
        d = (" ").join([z for z in BeautifulSoup(row["product_description"]).get_text(" ").split(" ")])

        q=re.sub("[^a-zA-Z0-9]"," ", q)
        t=re.sub("[^a-zA-Z0-9]"," ", t)
        d=re.sub("[^a-zA-Z0-9]"," ", d)

        q= (" ").join([stemmer.stem(z) for z in q.split(" ")])
        t= (" ").join([stemmer.stem(z) for z in t.split(" ")])
        d= (" ").join([stemmer.stem(z) for z in d.split(" ")])
        
        data.set_value(i, "query", q)
        data.set_value(i, "product_title", t)
        data.set_value(i, "product_description", d)


train = pd.read_csv("input/train.csv").fillna("")
test  = pd.read_csv("input/test.csv").fillna("")

stem_data(train)
stem_data(test)

extract_features(train)
extract_features(test)

#Extract features that can only be extracted on the training set
extract_training_features(train, test)

cPickle.dump(train, open('train_extracted_df.pkl', 'w'))
cPickle.dump(test, open('test_extracted_df.pkl', 'w'))

train.to_csv("Explore Training Set (With Transformations).csv", index=False)
test.to_csv("Explore Test Set (With Transformations).csv", index=False)



