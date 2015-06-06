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


#The first version of this script was taken from
#https://www.kaggle.com/users/993/ben-hamner/crowdflower-search-relevance/python-benchmark

train = pd.read_csv("input/train.csv").fillna("")
test  = pd.read_csv("input/test.csv").fillna("")

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

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryBagOfWords',          'query',                       CountVectorizer(max_features=200)),
                          ('TitleBagOfWords',          'product_title',               CountVectorizer(max_features=200)),
                          ('DescriptionBagOfWords',    'product_description',         CountVectorizer(max_features=200)),
                          ('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryTokensInDescription', 'query_tokens_in_description', SimpleTransform()),
                          ('QueryLength',              'query_length',                SimpleTransform()),
                          ('PQueryTokensInDescription','percent_query_tokens_in_description', SimpleTransform()),
                          ('PQueryTokensInTitle',      'percent_query_tokens_in_title', SimpleTransform()),
                          ('ExactQueryInTitle',        'exact_query_in_title',        SimpleTransform()),
                          ('ExactQueryInDescription',  'exact_query_in_description',  SimpleTransform())])

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

        

        #Other feature ideas - number of words in query, popularity of query (% of searches)
        #Also try adding a feature that simply contains the median or mean rating for that particular query - 
        #this may be what the CountVectorizer is doing - figure out what CountVectorizer does

#Evaluates model on the training data
#and output a matrix that can be used to conduct
#error analysis.
def perform_cross_validation(pipeline, train):
    kf = KFold(len(train), n_folds=5)
    score_count = 0
    score_total = 0.0
    frames = []
    for train_index, test_index in kf:
        X_train = train.loc[train_index]
        y_train = train.loc[train_index,"median_relevance"]
        X_test = train.loc[test_index]
        y_test = train.loc[test_index, "median_relevance"]
        y_test = y_test.loc[test_index]
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score_count += 1
        score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
        score_total += score
        print "Score " + str(score_count) + ": " + str(score)

        X_test["median_relevance_pred"] = predictions
        X_test["(i-j)^2"] = [(row["median_relevance"] - row["median_relevance_pred"])**2 for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        X_test["i-j"] = [row["median_relevance"] - row["median_relevance_pred"] for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        
        filename = "Error Analysis Iteration " + str(score_count) + ".csv"

        X_test.to_csv(filename, index=False)
        frames.append(X_test)
    pd.concat(frames).to_csv("Master Error Analysis File.csv", index=False)
        
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score) 



def ouput_final_model(pipeline, train, test):
    pipeline.fit(train, train["median_relevance"])

    predictions = pipeline.predict(test)

    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
    submission.to_csv("python_benchmark.csv", index=False)


extract_features(train)
extract_features(test)

train.to_csv("Explore Training Set (With Transformations).csv", index=False)

pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=200,
                                                         n_jobs=1,
                                                         min_samples_split=2,
                                                         random_state=1))])
'''
pipeline = Pipeline([("extract_features", features),
                    ("classify", GaussianNB())])
'''
perform_cross_validation(pipeline, train)
#ouput_final_model(pipeline = pipeline, train = train, test = test)

#Need to develop an internal, quick cross validation framework for testing the models