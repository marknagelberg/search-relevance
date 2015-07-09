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
from sklearn.cross_validation import KFold, StratifiedKFold



def extract_features(data):
    '''
    Perform feature extraction for variables that can be extracted
    the same way for both training and test data sets. The input
    "data" is the pandas dataframe for the training or test sets.
    '''
    token_pattern = re.compile(r"(?u)\b\w+\b")
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

        two_grams_in_query = set(get_n_grams(row["query"], 2))
        two_grams_in_title = set(get_n_grams(row["product_title"], 2))
        two_grams_in_description = set(get_n_grams(row["product_description"], 2))

        data.set_value(i, "two_grams_in_q_and_t", len(two_grams_in_query.intersection(two_grams_in_title)))
        data.set_value(i, "two_grams_in_q_and_d", len(two_grams_in_query.intersection(two_grams_in_description)))

def get_string_similarity(s1, s2):
    token_pattern = re.compile(r"(?u)\b\w+\b")
    s1_tokens = set(x.lower() for x in token_pattern.findall(s1))
    s2_tokens = set(x.lower() for x in token_pattern.findall(s2))
    if len(s1_tokens.union(s2_tokens)) == 0:
        return 0
    else:
        return float(len(s1_tokens.intersection(s2_tokens)))/float(len(s1_tokens.union(s2_tokens)))

def get_n_gram_string_similarity(s1, s2, n):
    '''
    Helper function to get the n-gram "similarity" between two strings,
    where n-gram similarity is defined as the percentage of n-grams
    the two strings have in common out of all of the n-grams across the
    two strings.
    '''
    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):
    '''
    Helper function that takes in a string and the degree of n gram n and returns a list of all the
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

def calculate_nearby_relevance_tuple(group, row, col_name, ngrams):
    '''
    Takes the group of rows for a particular query ("group") and a row within that 
    group ("row") and returns a dictionary of "similarity"  calculations of row compared to the rest 
    of the rows in group. Returns a tuple of calculations that will be used to create similarity features for row.
    '''

    ngrams = range(1, ngrams + 1)
    #Weighted ratings takes the form
    #{median rating : {ngram : [number of comparisons with that rating/ngram, cumulative sum of similarity for that rating/ngram]}}
    weighted_ratings = {rating: {ngram: [0,0] for ngram in ngrams} for rating in range(1,5)}

    for i, group_row in group.iterrows():
        if group_row['id'] != row['id']:

            for ngram in ngrams:
                similarity = helper.get_n_gram_string_similarity(row[col_name], group_row[col_name], ngram)
                weighted_ratings[group_row['median_relevance']][ngram][1] += similarity
                weighted_ratings[group_row['median_relevance']][ngram][0] += 1

    return weighted_ratings

def extract_training_features(train, test):
    train_group = train.groupby('query')
    test["q_mean_of_training_relevance"] = 0.0
    test["q_median_of_training_relevance"] = 0.0
    test["avg_relevance_variance"] = 0
    for i, row in train.iterrows():
        #Move the two blocks below outside this loop - can make them run much faster.
        group = train_group.get_group(row["query"])
        q_mean = group["median_relevance"].mean()
        train.set_value(i, "q_mean_of_training_relevance", q_mean)
        test.loc[test["query"] == row["query"], "q_mean_of_training_relevance"] = q_mean

        q_median = group["median_relevance"].median()
        train.set_value(i, "q_median_of_training_relevance", q_median)
        test.loc[test["query"] == row["query"], "q_median_of_training_relevance"] = q_median

        avg_relevance_variance = group["relevance_variance"].mean()
        train.set_value(i, "avg_relevance_variance", avg_relevance_variance)
        test.loc[test["query"] == row["query"], "avg_relevance_variance"] = avg_relevance_variance

        weight_dict = calculate_nearby_relevance_tuple(group, row, 'product_title', 3)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, 'product_description', 3)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    train.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    train.set_value(i, variable_name, 0)


    for i, row in test.iterrows():
        group = train_group.get_group(row["query"])

        weight_dict = calculate_nearby_relevance_tuple(group, row, 'product_title', 3)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_title_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)

        weight_dict = calculate_nearby_relevance_tuple(group, row, 'product_description', 3)
        for rating in weight_dict:
            for ngram in weight_dict[rating]:
                variable_name = "average_description_" + str(ngram) + "gram_similarity_" + str(rating)
                if weight_dict[rating][ngram][0] != 0:
                    test.set_value(i, variable_name, float(weight_dict[rating][ngram][1])/float(weight_dict[rating][ngram][0]))
                else:
                    test.set_value(i, variable_name, 0)

def stem_data(data):
    '''
    Helper function to stem the raw training and test data.
    '''
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
    '''
    Helper function to remove stop words
    from the raw training and test data.
    '''
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

def extract_bow_v1_features(train, test):

    traindata = train['query'] + ' ' + train['product_title']
    y_train = train['median_relevance']
    testdata = test['query'] + ' ' + test['product_title']
    if 'median_relevance' in test.columns.values:
        y_test = test['median_relevance']
    else:
        y_test = []

    return (traindata, y_train, testdata, y_test)


def extract_bow_v2_features(train, test, test_contains_labels = False):
    s_data = []
    s_labels = []
    t_data = []
    t_labels = []
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()    
    
    for i, row in train.iterrows():
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i, row in test.iterrows():
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
        if test_contains_labels:
            t_labels.append(str(test["median_relevance"][i]))
            
    return (s_data, s_labels, t_data, t_labels)

def extract(train, test):

    print "Extracting training features"
    extract_features(train)
    print "Extracting test features"
    extract_features(test)

    #Extract features that can only be extracted on the training set
    print "Extracting training/test features"
    extract_training_features(train, test)

    y_train = train.loc[:,"median_relevance"]
    train.drop("median_relevance", 1)

    if 'median_relevance' in test.columns.values:
        y_test = test.loc[:, "median_relevance"]
        test.drop("median_relevance", 1)
    else:
        y_test = []

    return train, y_train, test, y_test

if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv('input/train.csv').fillna("")
    test = pd.read_csv('input/test.csv').fillna("")

    #Extract data for each StratifiedKFold fold
    #and add the resulting sets of training/test
    #data to a list a dump it to csv and pickle.
    #List takes form [(X1_train, y1_train, X1_test, y1_test), ..., (X5_train, y5_train, X5_test, y5_test)]
    
    kfold_train_test = []
    bow_v1_kfold_trian_test = []
    bow_v2_kfold_trian_test = []    
    kf = StratifiedKFold(train["query"], n_folds=5)

    #####LOOK BELOW FOR THE OUTPUT OF THE BAG OF WORDS MODELS - THINK YOU MIGHT BE USING WRONG DATA TO TEST ENSEMBLE####
    for train_index, test_index in kf:
        
        X_train = train.loc[train_index]
        y_train = train.loc[train_index,"median_relevance"]

        X_test = train.loc[test_index]
        y_test = train.loc[test_index, "median_relevance"]

        #Extract bag of words features and add them to lists
        
        bow_v1_features = extract_bow_v1_features(X_train, X_test)
        bow_v1_kfold_trian_test.append(bow_v1_features)
        
        bow_v2_features = extract_bow_v2_features(X_train, X_test, test_contains_labels = True)
        bow_v2_kfold_trian_test.append(bow_v2_features)
        
        #Add/extract new variables to train and test
        #extract(X_train, X_test)
        #Add them to the list
        kfold_train_test.append((X_train, y_train, X_test, y_test))


    cPickle.dump(kfold_train_test, open('kfold_train_test.pkl', 'w'))
    ##NOTE - NEED TO RUN THE TWO LINES BELOW TO UPDATE BOW_V1_KFOLD_TRAIN_TEST.PKL - THE 
    #CROSS VALIDATION SCORES COMING OUT OF MODELLING.PY ARE INACCURATE.
    #cPickle.dump(bow_v1_kfold_trian_test, open('bow_v1_kfold_trian_test.pkl', 'w'))
    
    #cPickle.dump(bow_v2_kfold_trian_test, open('bow_v2_kfold_trian_test.pkl', 'w'))

    
    print "Extracting bag of words v1 features"
    bow_v1_features = extract_bow_v1_features(train, test)
    cPickle.dump(bow_v1_features, open('bow_v1_features_full_dataset.pkl', 'w'))
    
    print "Extracting bag of words v2 features"
    bow_v2_features = extract_bow_v2_features(train, test)
    cPickle.dump(bow_v2_features, open('bow_v2_features_full_dataset.pkl', 'w'))
    
    '''
    #Extract variables for full train and test set
    extract(train, test)
    train.to_csv("Explore Train Set (With Transformations).csv", index=False)
    test.to_csv("Explore Test Set (With Transformations).csv", index=False)
    cPickle.dump(train, open('train_extracted_df.pkl', 'w'))
    cPickle.dump(test, open('test_extracted_df.pkl', 'w'))  
    '''