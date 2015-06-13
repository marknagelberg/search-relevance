import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold, StratifiedKFold
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
import benchmark

#The first version of this script was taken from
#https://www.kaggle.com/users/993/ben-hamner/crowdflower-search-relevance/python-benchmark


class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T


#Evaluates model on the training data
#and output a matrix that can be used to conduct
#error analysis.
def perform_cross_validation(pipeline, train):
    kf = StratifiedKFold(train["query"], n_folds=5)
    score_count = 0
    score_total = 0.0
    #frames = []
    for train_index, test_index in kf:
        X_train = train.loc[train_index]
        y_train = train.loc[train_index,"median_relevance"]

        X_test = train.loc[test_index]
        y_test = train.loc[test_index, "median_relevance"]
        y_test = y_test.loc[test_index]

        #Add/extract new variables to train and test
        benchmark.extract(X_train, X_test)

        cPickle.dump(X_train, open('train_extracted_df_StratifiedKFold.pkl', 'w'))
        cPickle.dump(X_test, open('test_extracted_df_StratifiedKFold.pkl', 'w'))

        #Drop variables in X_train that we don't want to use in training
        X_train.drop('median_relevance', 1)
        X_train.drop('id', 1)

        #Fit the model
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score_count += 1
        score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
        score_total += score
        print "Score " + str(score_count) + ": " + str(score)

        X_test["median_relevance_pred"] = predictions
        X_test["(i-j)^2"] = [(row["median_relevance"] - row["median_relevance_pred"])**2 for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        X_test["i-j"] = [row["median_relevance"] - row["median_relevance_pred"] for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        filename = "StratifiedKFold Test Set Iteration " + str(score_count) + ".csv"
        X_test.to_csv(filename, index=False)

        break
        
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score) 
    return (X_train, y_train)



def ouput_final_model(pipeline, train, test):
  y = train["median_relevance"]
  train.drop('median_relevance', 1)
  train.drop('id', 1)
  pipeline.fit(train, y)

  predictions = pipeline.predict(test)

  submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
  submission.to_csv("python_benchmark.csv", index=False)

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokensInDescription', 'query_tokens_in_description', SimpleTransform()),
                          ('QueryLength',              'query_length',                SimpleTransform()),
                          ('PQueryTokensInDescription','percent_query_tokens_in_description', SimpleTransform()),
                          ('PQueryTokensInTitle',      'percent_query_tokens_in_title', SimpleTransform()),
                          ('ExactQueryInTitle',        'exact_query_in_title',        SimpleTransform()),
                          ('ExactQueryInDescription',  'exact_query_in_description',  SimpleTransform()),
                          ('SpaceRemovedQinT',         'space_removed_q_in_t',        SimpleTransform()),
                          ('SpaceRemovedQinD',         'space_removed_q_in_d',        SimpleTransform()),
                          ('QMeanTrainingRelevance',   'q_mean_of_training_relevance',SimpleTransform()),
                          ('QMedianTrainingRelevance', 'q_median_of_training_relevance',SimpleTransform()),
                          ('ClosestTitleRelevance',    'closest_title_relevance',     SimpleTransform()),
                          ('ClosestDescriptionRelevance', 'closest_description_relevance', SimpleTransform()),
                          ('WeightedTitleRelevance',   'weighted_title_relevance',    SimpleTransform()),
                          ('WeightedDescriptionRelevance', 'weighted_description_relevance', SimpleTransform()),
                          ('WeightedDescriptionRelevanceTwo', 'weighted_description_relevance_two', SimpleTransform()),
                          ('WeightedTitleRelevanceTwo', 'weighted_title_relevance_two', SimpleTransform())])


# note - removed ('svd', TruncatedSVD(n_components=225, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)) from below
pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=200,
                                                         n_jobs=1,
                                                         min_samples_split=2,
                                                         random_state=1))])

#train = cPickle.load(open('train_extracted_df.pkl', 'r'))
#test = cPickle.load(open('test_extracted_df.pkl', 'r'))

train = pd.read_csv("input/train.csv").fillna("")
perform_cross_validation(pipeline, train)
#ouput_final_model(pipeline = pipeline, train = train, test = test)

#Need to develop an internal, quick cross validation framework for testing the models
