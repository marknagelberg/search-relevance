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

        #Add features to X_test and print it to a csv file
        X_test["median_relevance_pred"] = predictions
        X_test["(i-j)^2"] = [(row["median_relevance"] - row["median_relevance_pred"])**2 for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        X_test["i-j"] = [row["median_relevance"] - row["median_relevance_pred"] for idx, row in X_test.loc[:, ("median_relevance","median_relevance_pred")].iterrows()]
        filename = "stratifiedkfold_test_set_with_predictions_" + str(score_count) + ".csv"
        X_test.to_csv(filename, index=False)
        y_and_y_pred = pd.DataFrame({'y': y_test, 'y_pred': predictions})

        #Add y and y_pred to array to return for ensembling purposes
        test_data.append(y_and_y_pred)
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score)
    return test_data

def perform_tfidf_v1_cross_validation(tfv, pipeline, kfold_train_test):
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
      y_and_y_pred = pd.DataFrame({'y': y_test, 'y_pred': predictions})
      test_data.append(y_and_y_pred)
      
    average_score = score_total/float(score_count)
    print "Average score: " + str(average_score)
    return test_data


def ouput_final_model(pipeline, train, test, filename):

  y = train["median_relevance"]
  pipeline.fit(train, y)
  train_predictions = pipeline.predict(train)
  predictions = pipeline.predict(test)
  submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
  submission.to_csv(filename, index=False)
  train['prediction'] = train_predictions
  train.to_csv(filename + '.csv', index=False)
  return submission

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
                          ('TwoGramsInQandD',           'two_grams_in_q_and_d', SimpleTransform()),
                          ('AvgRelevanceVariance',      'avg_relevance_variance', SimpleTransform())])



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

#features = features.remove(['weighted_2gram_description_relevance'])

# Kappa Scorer 
kappa_scorer = metrics.make_scorer(evaluation.quadratic_weighted_kappa, greater_is_better = True)


####Random forest model#####
print "Begin random forest model"
pipeline = Pipeline([("extract_features", features),
                     ("classify", RandomForestClassifier(n_estimators=300,
                                                         n_jobs=1,
                                                         min_samples_split=10,
                                                         random_state=1,
                                                         class_weight='auto'))])

#Get importance of each variable from Random Forest
'''
clf = RandomForestClassifier(n_estimators=300,n_jobs=1,min_samples_split=10,random_state=1,class_weight='auto')
clf.fit(train[features.get_column_names()], y_train)
import csv
with open('importances.csv', 'wb') as csvfile:

  for i, name in enumerate(features.get_column_names()):
    print name + ', ' + str(clf.feature_importances_[i])
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([name + ',' + str(clf.feature_importances_[i])])
'''


rf_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
cPickle.dump(rf_cv_test_data, open('rf_cv_test_data.pkl', 'w'))
rf_final_predictions = ouput_final_model(pipeline, train, test, "rf_final_predictions.csv")
cPickle.dump(rf_final_predictions, open('rf_final_predictions.pkl', 'w'))

'''
for col_name in features.get_column_names():

  new_features = features.remove([col_name])
  pipeline = Pipeline([("extract_features", new_features),
                     ("classify", RandomForestClassifier(n_estimators=300,
                                                         n_jobs=1,
                                                         min_samples_split=10,
                                                         random_state=1,
                                                         class_weight='auto'))])
  print "Score with " + col_name + " removed"
  perform_cross_validation(pipeline, kfold_train_test)
'''




####SVC Model####
print "Begin SVC model"
scl = StandardScaler()
svm_model = SVC(C=10.0, random_state = 1, class_weight = {1:2, 2:1.5, 3:1, 4:1})
pipeline = Pipeline([("extract_features", features), ('scl', scl), ('svm', svm_model)])
svc_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
cPickle.dump(svc_cv_test_data, open('svc_cv_test_data.pkl', 'w'))
svc_final_predictions = ouput_final_model(pipeline, train, test, "svc_final_predictions.csv")
cPickle.dump(svc_final_predictions, open('svc_final_predictions.pkl', 'w'))



####AdaBoost Model####
print "Begin AdaBoost model"
pipeline = Pipeline([("extract_features", features),
                     ("classify", AdaBoostClassifier(n_estimators=200, random_state = 1, learning_rate = 0.25))])
adaboost_cv_test_data = perform_cross_validation(pipeline, kfold_train_test)
cPickle.dump(adaboost_cv_test_data, open('adaboost_cv_test_data.pkl', 'w'))
adaboost_final_predictions = ouput_final_model(pipeline, train, test, "adaboost_final_predictions.csv")
cPickle.dump(adaboost_final_predictions, open('adaboost_final_predictions.pkl', 'w'))


'''
####Model using bag of words TFIDF v1####
print "Begin TFIDF v1 model"
idx = test.id.values.astype(int)
train_v1, y_v1, test_v1 = bow_v1_features

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

pipeline = Pipeline([('svd', TruncatedSVD(n_components=400)), ('scl', StandardScaler()), ('svm', SVC(C=10))])

tfidf_v1_test_data = perform_tfidf_v1_cross_validation(tfv, pipeline, bow_v1_kfold_trian_test)
cPickle.dump(tfidf_v1_test_data, open('tfidf_v1_test_data.pkl', 'w'))

#Output final model for TFIDF v1

tfv.fit(train_v1)
X_train =  tfv.transform(train_v1) 
X_test = tfv.transform(test_v1)
pipeline.fit(X_train, y_v1)
predictions = pipeline.predict(X_test)

submission = pd.DataFrame({"id": idx, "prediction": predictions})
submission.to_csv('tfidf_v1_final_predictions.csv', index=False)
cPickle.dump(submission, open('tfidf_v1_final_predictions.pkl', 'w'))


####Model using bag of words TFIDF v2####
print "Begin TFIDF v2 model"
data = cPickle.load(open('bow_v2_features_full_dataset.pkl', 'r'))
bow_v2_kfold_trian_test = cPickle.load(open('bow_v2_kfold_trian_test.pkl', 'r'))
idx = test.id.values.astype(int)

tfv = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')
#create sklearn pipeline, fit all, and predit test data
pipeline = Pipeline([('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])

#Note that perform_tfidf_v1_cross_validation works for v2 tfidf as well - might want to change name of
#function to  perform_tfidf_cross_validation
tfidf_v2_test_data = perform_tfidf_v1_cross_validation(tfv, pipeline, bow_v2_kfold_trian_test)
cPickle.dump(tfidf_v2_test_data, open('tfidf_v2_test_data.pkl', 'w'))

#Output final model for TFIDF v2
train_v2, y_v2, test_v2, y_v2_empty = bow_v2_features
tfv.fit(train_v2)
X_train =  tfv.transform(train_v2) 
X_test = tfv.transform(test_v2)
pipeline.fit(X_train, y_v2)
predictions = pipeline.predict(X_test)

submission = pd.DataFrame({"id": idx, "prediction": predictions})
submission.to_csv('tfidf_v2_final_predictions.csv', index=False)
cPickle.dump(submission, open('tfidf_v2_final_predictions.pkl', 'w'))

'''


###########################

'''
for col_name in features.get_column_names():

  new_features = features.remove([col_name])
  pipeline = Pipeline([("extract_features", new_features),
                     ("classify", RandomForestClassifier(n_estimators=300,
                                                         n_jobs=1,
                                                         min_samples_split=10,
                                                         random_state=1,
                                                         class_weight='auto'))])
  print "Score with " + col_name + " removed"
  perform_cross_validation(pipeline, kfold_train_test)
'''


