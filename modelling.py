import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
import math
from sklearn import decomposition, pipeline, metrics, grid_search

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

    def remove(self, feature_list):
      new_features = [x for x in self.features if x[1] not in feature_list]
      return_features = FeatureMapper(new_features)
      return return_features

    def add(self, feature_name):
      self.features.append((feature_name, feature_name, SimpleTransform()))

    def get_column_names(self):
      column_names = []
      for feature_name, column_name, extractor in self.features:
        column_names.append(column_name)
      return column_names

    def get_params(self, deep = True):
      return {}

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
#and outputs test data that can be used to conduct
#error analysis. Takes in the model to evaluate and
#the list of train/test data from each of the K Folds.
def perform_cross_validation(pipeline, kfold_train_test):
    score_count = 0
    score_total = 0.0
    test_data = []
    for X_train, y_train, X_test, y_test in kfold_train_test:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score_count += 1
        score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
        score_total += score
        print "Score " + str(score_count) + ": " + str(score)

        X_test["median_relevance_pred"] = predictions
        X_test["(i-j)^2"] = [(row["median_relevance"] - row["median_relevance_pred"])**2 for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        X_test["i-j"] = [row["median_relevance"] - row["median_relevance_pred"] for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        filename = "stratifiedkfold_test_set_with_predictions_" + str(score_count) + ".csv"
        X_test.to_csv(filename, index=False)
        test_data.append(X_test)
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score)
    return test_data

def perform_tfidf_cross_validation(transform, pipeline, kfold_train_test):
    score_count = 0
    score_total = 0.0
    test_data = []
    for X_train, y_train, X_test, y_test in kfold_train_test:
        tfv.fit(X_train)
        X_train =  tfv.transform(X_train) 
        X_test = tfv.transform(X_test)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score_count += 1
        score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
        score_total += score
        print "Score " + str(score_count) + ": " + str(score)

        X_test["median_relevance_pred"] = predictions
        X_test["(i-j)^2"] = [(row["median_relevance"] - row["median_relevance_pred"])**2 for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        X_test["i-j"] = [row["median_relevance"] - row["median_relevance_pred"] for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        filename = "stratifiedkfold_tfidf_test_set_with_predictions_" + str(score_count) + ".csv"
        X_test.to_csv(filename, index=False)
        test_data.append(X_test)
        break
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score)
    return test_data


def ouput_final_model(pipeline, train, test, filename):

  y = train["median_relevance"]
  pipeline.fit(train, y)
  predictions = pipeline.predict(test)

  submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
  submission.to_csv(filename, index=False)
  return predictions

#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureMapper([('QueryTokensInTitle',       'query_tokens_in_title',       SimpleTransform()),
                          ('QueryLength',              'query_length',                SimpleTransform()),
                          ('PQueryTokensInTitle',      'percent_query_tokens_in_title', SimpleTransform()),
                          ('ExactQueryInTitle',        'exact_query_in_title',        SimpleTransform()),
                          ('ExactQueryInDescription',  'exact_query_in_description',  SimpleTransform()),
                          ('SpaceRemovedQinT',         'space_removed_q_in_t',        SimpleTransform()),
                          ('SpaceRemovedQinD',         'space_removed_q_in_d',        SimpleTransform()),
                          ('QMeanTrainingRelevance',   'q_mean_of_training_relevance',SimpleTransform()),
                          ('QMedianTrainingRelevance', 'q_median_of_training_relevance',SimpleTransform()),
                          ('ClosestTitleRelevance',    'closest_title_relevance',     SimpleTransform()),
                          ('Closest2GramTitleRelevance',    'closest_2gram_title_relevance',     SimpleTransform()),
                          ('ClosestDescriptionRelevance', 'closest_description_relevance', SimpleTransform()),
                          ('Closest2GramDescriptionRelevance', 'closest_2gram_description_relevance', SimpleTransform()),
                          ('WeightedTitleRelevance',   'weighted_title_relevance',    SimpleTransform()),
                          ('Weighted2GramTitleRelevance',   'weighted_2gram_title_relevance',    SimpleTransform()),
                          ('WeightedDescriptionRelevance', 'weighted_description_relevance', SimpleTransform()),
                          ('Weighted2GramDescriptionRelevance', 'weighted_2gram_description_relevance', SimpleTransform()),
                          ('WeightedDescriptionRelevanceTwo', 'weighted_description_relevance_two', SimpleTransform()),
                          ('Weighted2GramDescriptionRelevanceTwo', 'weighted_2gram_description_relevance_two', SimpleTransform()),
                          ('WeightedTitleRelevanceTwo', 'weighted_title_relevance_two', SimpleTransform()),
                          ('Weighted2GramTitleRelevanceTwo', 'weighted_2gram_title_relevance_two', SimpleTransform()),
                          ('TwoGramsInQandT',           'two_grams_in_q_and_t', SimpleTransform()),
                          ('TwoGramsInQandD',           'two_grams_in_q_and_d', SimpleTransform())])



#Load all of the extracted data, including the full train/test data 
#and the StratifiedKFold data
train = cPickle.load(open('train_extracted_df.pkl', 'r'))
test = cPickle.load(open('test_extracted_df.pkl', 'r'))
y_train = train["median_relevance"]
kfold_train_test = cPickle.load(open('kfold_train_test.pkl', 'r'))
bow_v1_features = cPickle.load(open('bow_v1_features_full_dataset.pkl', 'r'))
bow_v2_features = cPickle.load(open('bow_v2_features_full_dataset.pkl', 'r'))
bow_v1_kfold_trian_test = cPickle.load(open('bow_v1_kfold_trian_test.pkl', 'r'))
bow_v2_kfold_trian_test = cPickle.load(open('bow_v2_kfold_trian_test.pkl', 'r'))

# Kappa Scorer 
kappa_scorer = metrics.make_scorer(evaluation.quadratic_weighted_kappa, greater_is_better = True)

#Use variation of code below within each model to tweak parameters
'''
# Create a parameter grid to search for best parameters for everything in the pipeline
param_grid = {'svd__n_components' : [400],
              'svm__C': [10]}


# Initialize Grid Search Model
model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
'''


####Random forest model#####
pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=300,
                                                         n_jobs=1,
                                                         min_samples_split=10,
                                                         random_state=1,
                                                         class_weight='auto'))])

rf_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
rf_final_predictions = ouput_final_model(pipeline, train, test, "rf_final_predictions.csv")

'''
####SVC Model####
scl = StandardScaler()
svm_model = SVC(C=10.0)
pipeline = Pipeline([("extract_features", features), ('scl', scl), ('svm', svm_model)])
svc_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
svc_final_predictions = ouput_final_model(pipeline, train, test, "svc_final_predictions.csv")


####AdaBoost Model####
pipeline = Pipeline([("extract_features", features),
                     ("classify", AdaBoostClassifier(n_estimators=100))])
adaboost_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
adaboost_final_predictions = ouput_final_model(pipeline, train, test, "adaboost_final_predictions.csv")


####Model using bag of words TFIDF v1####
train = cPickle.load(open('bow_v1_train_full_dataset.pkl', 'r'))
test = cPickle.load(open('bow_v1_test_full_dataset.pkl', 'r')) 
bow_v1_kfold_trian_test = cPickle.load(open('bow_v1_kfold_trian_test.pkl', 'r'))

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

tfv.fit(train)
train =  tfv.transform(train) 
test = tfv.transform(test)
pipeline = Pipeline([('svd', TruncatedSVD()), ('scl', StandardScaler()), ('svm', SVC())])

tfidf_v1_test_data = perform_tfidf_cross_validation(tfv, pipeline, bow_v1_kfold_trian_test)
tfidf_v1_final_predictions = ouput_final_model(pipeline, train, test, "tfidf_v1_final_predictions.csv")

####Model using bag of words TFIDF v2####
data = cPickle.load(open('bow_v2_features_full_dataset.pkl', 'r'))
bow_v2_kfold_trian_test = cPickle.load(open('bow_v2_kfold_trian_test.pkl', 'r'))

#create sklearn pipeline, fit all, and predit test data
pipeline = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
pipeline.fit(s_data, s_labels)
t_labels = pipeline.predict(t_data)

tfidf_test_data = perform_tfidf_cross_validation(tfv, pipeline, bow_v1_kfold_trian_test)
tfidf_v2_final_predictions = ouput_final_model(pipeline, train, test, "tfidf_v2_final_predictions.csv")

#Develop a framework for testing different weightings of model results
predictions = (rf_predictions + svc_predictions + adaboost_predictions)/3.0

#Try rounding rather than floor function
predictions = [int(math.floor(p)) for p in predictions]
#y_test = test["median_relevance"]
#score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
#print "Score: " + str(score)

submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("python_benchmark.csv", index=False)

'''


###########################

'''
#Test removing each variable to see how score changes
for col_name in features.get_column_names():

  new_features = features.remove([col_name])

  #pipeline = Pipeline([("extract_features", new_features), ("classify", AdaBoostClassifier(n_estimators=100))])

  pipeline = Pipeline([("extract_features", new_features), ('scl', scl), ('svm', svm_model)])


  #Fit the model
  pipeline.fit(train, y_train)
  predictions = pipeline.predict(test)
  score = evaluation.quadratic_weighted_kappa(y = y_test, y_pred = predictions)
  print "Score with " + col_name + " removed: " + str(score)
'''