import nltk
from nltk.corpus import stopwords
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


def extract_features(data, stemmed):
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

        if stemmed:
            translate_to = u''
            translate_table = dict((ord(char), translate_to) for char in u' -')
            q_space_removed = row["query"].lower().translate(translate_table)
            t_space_removed = row["product_title"].lower().translate(translate_table)
            d_space_removed = row["product_description"].lower().translate(translate_table)
        if not stemmed:
            q_space_removed = row["query"].lower().translate(None, ' -')
            t_space_removed = row["product_title"].lower().translate(None, ' -')
            d_space_removed = row["product_description"].lower().translate(None, ' -')

        if q_space_removed in t_space_removed:
            data.set_value(i, "space_removed_q_in_t", 1)
        else:
            data.set_value(i, "space_removed_q_in_t", 0)

        if q_space_removed in d_space_removed:
            data.set_value(i, "space_removed_q_in_d", 1)
        else:
            data.set_value(i, "space_removed_q_in_d", 0)

        two_grams_in_query = set(get_n_grams(row["query"], 2))
        two_grams_in_title = set(get_n_grams(row["product_title"], 2))
        two_grams_in_description = set(get_n_grams(row["product_description"], 2))

        data.set_value(i, "two_grams_in_q_and_t", len(two_grams_in_query.intersection(two_grams_in_title)))
        data.set_value(i, "two_grams_in_q_and_d", len(two_grams_in_query.intersection(two_grams_in_description)))

def get_string_similarity(s1, s2):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    s1_tokens = set(x.lower() for x in token_pattern.findall(s1))
    s2_tokens = set(x.lower() for x in token_pattern.findall(s2))
    if len(s1_tokens.union(s2_tokens)) == 0:
        return 0
    else:
        return float(len(s1_tokens.intersection(s2_tokens)))/float(len(s1_tokens.union(s2_tokens)))

def get_n_gram_string_similarity(s1, s2, n):
    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):
    '''
    Takes in a string and the degree of n gram n and returns a list of all the
    n grams in the string. String is separated by space.
    '''
    word_list = s.split()
    n_grams = []


    if n > len(word_list):
        return []
    
    for i, word in enumerate(word_list):
        n_gram = word_list[i:i+n]
        if len(n_gram) == n:
            n_grams.append(tuple(n_gram))
    return n_grams

def calculate_nearby_relevance_tuple(group, row, col_name):
    '''
    Takes a group of rows for a particular query and a row within that 
    group and returns several calculations based on the median relevance of
    "similar" entries. Returns a tuple of calculations. Tuple returns a rating that
    weights similarity across all other rows and returns a rating that is simply the 
    rating of the one "closest" row.
    '''
    return_rating = 0
    return_2gram_rating = 0
    max_similarity = 0
    max_2gram_similarity = 0
    weighted_ratings = {x:[0,0] for x in range(1,5)}
    weighted_2gram_ratings = {x:[0,0] for x in range(1,5)}
    for i, group_row in group.iterrows():
        #Weighted ratings takes the form {median relevance value: [number of comparisons with that relevance value, cumulative sum of similarity]}
        #We ultimately want to return the median relevance value with the highest cumulative_sum/num_comparisons
        if group_row['id'] != row['id']:
            similarity = get_string_similarity(row[col_name], group_row[col_name])
            twogram_similarity = get_n_gram_string_similarity(row[col_name], group_row[col_name], 2)
            weighted_ratings[group_row['median_relevance']][0] += 1
            weighted_2gram_ratings[group_row['median_relevance']][0] += 1
            weighted_ratings[group_row['median_relevance']][1] += similarity
            weighted_2gram_ratings[group_row['median_relevance']][0] += twogram_similarity
            if similarity > max_similarity:
                max_similarity = similarity
                return_rating = group_row['median_relevance']
            if twogram_similarity > max_2gram_similarity:
                max_2gram_similarity = twogram_similarity
                return_2gram_rating = group_row['median_relevance']


    max_weighted_similarity = 0
    max_weighted_median_rating = 0
    for rating in weighted_ratings:
        if weighted_ratings[rating][0] != 0:
            current_weighted_similarity = float(weighted_ratings[rating][1])/float(weighted_ratings[rating][0])
            if current_weighted_similarity > max_weighted_similarity:
                max_weighted_similarity = current_weighted_similarity
                max_weighted_median_rating = rating

    weighted_median_rating = float(sum(x * weighted_ratings[x][1] for x in weighted_ratings))/float(sum(weighted_ratings[x][0] for x in weighted_ratings))

    max_weighted_2gram_similarity = 0
    max_weighted_2gram_median_rating = 0
    for rating in weighted_2gram_ratings:
        if weighted_2gram_ratings[rating][0] != 0:
            current_weighted_2gram_similarity = float(weighted_2gram_ratings[rating][1])/float(weighted_2gram_ratings[rating][0])
            if current_weighted_2gram_similarity > max_weighted_2gram_similarity:
                max_weighted_2gram_similarity = current_weighted_2gram_similarity
                max_weighted_2gram_median_rating = rating

    weighted_2gram_median_rating = float(sum(x * weighted_2gram_ratings[x][1] for x in weighted_2gram_ratings))/float(sum(weighted_2gram_ratings[x][0] for x in weighted_2gram_ratings))

    return (return_rating, max_weighted_median_rating, weighted_median_rating, return_2gram_rating, max_weighted_2gram_similarity, weighted_2gram_median_rating)

        
def extract_training_features(train, test):
    train_group = train.groupby('query')
    test["q_mean_of_training_relevance"] = 0.0
    test["q_median_of_training_relevance"] = 0.0
    test["closest_title_relevance"] = 0
    test["closest_description_relevance"] = 0
    test["closest_2gram_title_relevance"] = 0
    test["closest_2gram_description_relevance"] = 0
    for i, row in train.iterrows():
        #Move the two blocks below outside this loop - can make them run much faster.
        group = train_group.get_group(row["query"])
        q_mean = group["median_relevance"].mean()
        train.set_value(i, "q_mean_of_training_relevance", q_mean)
        test.loc[test["query"] == row["query"], "q_mean_of_training_relevance"] = q_mean

        q_median = group["median_relevance"].median()
        train.set_value(i, "q_median_of_training_relevance", q_median)
        test.loc[test["query"] == row["query"], "q_median_of_training_relevance"] = q_median

        (closest_title_relevance, weighted_title_relevance, weighted_title_relevance_two, closest_2gram_title_relevance, weighted_2gram_title_relevance, weighted_2gram_title_relevance_two) = calculate_nearby_relevance_tuple(group, row, 'product_title')
        train.set_value(i, "closest_title_relevance", closest_title_relevance)
        train.set_value(i, "weighted_title_relevance", weighted_title_relevance)
        train.set_value(i, "weighted_title_relevance_two", weighted_title_relevance_two)
        train.set_value(i, "closest_2gram_title_relevance", closest_2gram_title_relevance)
        train.set_value(i, "weighted_2gram_title_relevance", weighted_2gram_title_relevance)
        train.set_value(i, "weighted_2gram_title_relevance_two", weighted_2gram_title_relevance_two)

        (closest_description_relevance, weighted_description_relevance, weighted_description_relevance_two, closest_2gram_description_relevance, weighted_2gram_description_relevance, weighted_2gram_description_relevance_two) = calculate_nearby_relevance_tuple(group, row, 'product_description')
        train.set_value(i, "closest_description_relevance", closest_description_relevance)
        train.set_value(i, "weighted_description_relevance", weighted_description_relevance)
        train.set_value(i, "weighted_description_relevance_two", weighted_description_relevance_two)
        train.set_value(i, "closest_2gram_description_relevance", closest_2gram_description_relevance)
        train.set_value(i, "weighted_2gram_description_relevance", weighted_2gram_description_relevance)
        train.set_value(i, "weighted_2gram_description_relevance_two", weighted_2gram_description_relevance_two)

    for i, row in test.iterrows():
        group = train_group.get_group(row["query"])
        (closest_title_relevance, weighted_title_relevance, weighted_title_relevance_two, closest_2gram_title_relevance, weighted_2gram_title_relevance, weighted_2gram_title_relevance_two) = calculate_nearby_relevance_tuple(group, row, 'product_title')
        test.set_value(i, "closest_title_relevance", closest_title_relevance)
        test.set_value(i, "weighted_title_relevance", weighted_title_relevance)
        test.set_value(i, "weighted_title_relevance_two", weighted_title_relevance_two)
        test.set_value(i, "closest_2gram_title_relevance", closest_2gram_title_relevance)
        test.set_value(i, "weighted_2gram_title_relevance", weighted_2gram_title_relevance)
        test.set_value(i, "weighted_2gram_title_relevance_two", weighted_2gram_title_relevance_two)

        (closest_description_relevance, weighted_description_relevance, weighted_description_relevance_two, closest_2gram_description_relevance, weighted_2gram_description_relevance, weighted_2gram_description_relevance_two) = calculate_nearby_relevance_tuple(group, row, 'product_description')
        test.set_value(i, "closest_description_relevance", closest_description_relevance)        
        test.set_value(i, "weighted_description_relevance", weighted_description_relevance)
        test.set_value(i, "weighted_description_relevance_two", weighted_description_relevance_two)
        test.set_value(i, "closest_2gram_description_relevance", closest_2gram_description_relevance)        
        test.set_value(i, "weighted_2gram_description_relevance", weighted_2gram_description_relevance)
        test.set_value(i, "weighted_2gram_description_relevance_two", weighted_2gram_description_relevance_two)

def stem_data(data):

    stemmer = PorterStemmer()

    for i, row in data.iterrows():

        q = (" ").join([z for z in BeautifulSoup(row["query"]).get_text(" ").split(" ")])
        t = (" ").join([z for z in BeautifulSoup(row["product_title"]).get_text(" ").split(" ")]) 
        d = (" ").join([z for z in BeautifulSoup(row["product_description"]).get_text(" ").split(" ")])

        q=re.sub("[^a-zA-Z0-9]"," ", q)
        t=re.sub("[^a-zA-Z0-9]"," ", t)
        d=re.sub("[^a-zA-Z0-9]"," ", d)

        q= (" ").join([stemmer.stem(z) for z in q.split()])
        t= (" ").join([stemmer.stem(z) for z in t.split()])
        d= (" ").join([stemmer.stem(z) for z in d.split()])
        
        data.set_value(i, "query", unicode(q))
        data.set_value(i, "product_title", unicode(t))
        data.set_value(i, "product_description", unicode(d))

def remove_stop_words(data):
    stop = stopwords.words('english')

    for i, row in data.iterrows():

        q = row["query"].lower().split(" ")
        t = row["product_title"].lower().split(" ")
        d = row["product_description"].lower().split(" ")

        q = (" ").join([z for z in q if z not in stop])
        t = (" ").join([z for z in t if z not in stop])
        d = (" ").join([z for z in d if z not in stop])

        data.set_value(i, "query", q)
        data.set_value(i, "product_title", t)
        data.set_value(i, "product_description", d)

def extract(train, test):

    print "Removing stop words in training data"
    remove_stop_words(train)
    print "Removing stop words in test data"
    remove_stop_words(test)

    print "Stemming training data"
    stem_data(train)
    print "Stemming test data"
    stem_data(test)

    print "Extracting training features"
    extract_features(train, stemmed = True)
    print "Extracting test features"
    extract_features(test, stemmed = True)

    #Extract features that can only be extracted on the training set
    print "Extracting training/test features"
    extract_training_features(train, test)

