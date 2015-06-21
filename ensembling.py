import evaluation
import cPickle
import math
import pandas as pd
import numpy as np

#Find the best ensemble model, trying every combination of the following weights - 0, .2, .4, .6, .8, 1

rf_cv_predictions = cPickle.load(open('rf_cv_test_data.pkl', 'r'))
svc_cv_predictions = cPickle.load(open('svc_cv_test_data.pkl', 'r'))
adaboost_cv_predictions = cPickle.load(open('adaboost_cv_test_data.pkl', 'r'))
tfidf_v1_cv_predictions = cPickle.load(open('tfidf_v1_test_data.pkl', 'r'))
tfidf_v2_cv_predictions = cPickle.load(open('tfidf_v2_test_data.pkl', 'r'))

preds = [rf_cv_predictions, svc_cv_predictions, adaboost_cv_predictions, tfidf_v1_cv_predictions, tfidf_v2_cv_predictions]

y_true = preds[0][0]['y']
y_pred = preds[0][0]['y_pred']
#preds = (rf_cv_predictions[0] + svc_cv_predictions[0] + adaboost_cv_predictions[0] + tfidf_v1_cv_predictions[0] + tfidf_v2_cv_predictions[0])/5


wt = [0.0, .1, .25, .5, .2, 1.0]
wt_list = [(a,b,c,d,e) for a in wt for b in wt for c in wt for d in wt for e in wt]
wt_final = []

#Built wt_final, which contains all lists of length 5 with values in wt which sum to 1
#These will be used as weights in the ensembling
for w in wt_list:
	if sum(w) == 1.0:
		wt_final.append(w)

max_average_score = 0
max_weights = None
for wt in wt_final:
	total_score = 0
	for i in range(5):
		y_true = preds[0][i]['y']
		weighted_prediction = sum([wt[x] * preds[x][i]['y_pred'].astype(int).reset_index() for x in range(5)])
		#weighted_prediction = [int(math.floor(p)) for p in weighted_prediction['y_pred']]
		weighted_prediction = [round(p) for p in weighted_prediction['y_pred']]
		total_score += evaluation.quadratic_weighted_kappa(y = y_true, y_pred = weighted_prediction)
	average_score = total_score/5.0
	if average_score > max_average_score:
		max_average_score = average_score
		max_weights = wt
print "Best set of weights: " + str(max_weights)
print "Corresponding score: " + str(max_average_score)


#Now perform the best ensembling on the full dataset and output the model to submit
rf_final_predictions = cPickle.load(open('rf_final_predictions.pkl', 'r'))
svc_final_predictions = cPickle.load(open('svc_final_predictions.pkl', 'r'))
adaboost_final_predictions = cPickle.load(open('adaboost_final_predictions.pkl', 'r'))
tfidf_v1_final_predictions = cPickle.load(open('tfidf_v1_final_predictions.pkl', 'r'))
tfidf_v2_final_predictions = cPickle.load(open('tfidf_v2_final_predictions.pkl', 'r'))

preds = [rf_final_predictions, svc_final_predictions, adaboost_final_predictions, tfidf_v1_final_predictions, tfidf_v2_final_predictions]

weighted_prediction = sum([max_weights[x] * preds[x]["prediction"].astype(int) for x in range(5)])
weighted_prediction = [int(round(p)) for p in weighted_prediction]

'''
#Different rounding technique that gets .002 increase in cv - NOTE - resulted in lower score on leaderboard.
new_weighted_prediction = []
for p in weighted_prediction:
	if p >= 2.0:
		new_weighted_prediction.append(int(round(p)))
	else:
		new_weighted_prediction.append(int(math.floor(p)))
'''

test = cPickle.load(open('test_extracted_df.pkl', 'r'))
submission = pd.DataFrame({"id": test["id"], "prediction": weighted_prediction})
submission.to_csv('ensembled_submission.csv', index=False)

'''
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