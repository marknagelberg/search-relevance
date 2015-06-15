Note - recommended using k fold cross validation when evaluating cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold

evaluation.py contains scripts for performing evaluation against the metric provided.

When conducting error analysis with KFold - instead of breaking it down into k files, stack the 
tables on top of one another to get one big file of the model trained and tested on different
data.

6/5/2015 - seem to be getting a lot of false positives - tend to overrate the query results.
This probably makes sense, because the overall rating of query results is high, so any mistake
is probably going to err on the high side. 

TFIDF vectorizer always crashes my computer and tends to fail. In fact, when I run the models without any
vectorizers (count of TFIDF), I get a better result. Perhaps down the road try TFIDF but running more random
forest trees through it.

TO DO:
Add and test features (see below)
Add comments to your code (so other can undersetand and also so you can understand - your
code borrows a lot from others)
Adjust cross validation so it always includes a proportionate number of each query
Look for algorithms for determining the similarity between two sentences.
Check out recommender system algorithms that may be of use http://en.wikipedia.org/wiki/Recommender_system
Perhaps consider ensembling with a model that uses bag of words technique. Look up ensembling methods - 
this will be a last step, but don't want to be rushed.

Separate work - code that extracts features and pickles the data - make it so it extracts full data set and also 
extracts for each of the 5 StratifiedKFolds. Then have code that runs the models on the data - code that runs on the full
data and also runs on each of the cv sets.

Feature ideas:
Variance rating in training for that particular query, across all titles and descriptions
When looking at the rating of the closest other query, there is always a chance that the closest actually
isn't all that similar at all. So, we might want to include the actual measure of closeness of the nearest title/description.
Or only include distance if some sort of threshold is met - notice this problem affects weighted relevance too.
At least do this so you can visualize the relationship beween closeness and accuracy of prediction.

Would definitely help to have a 2-gram similarity rating. Also, it may be good to have a feature indicating how many 2-grams
in the query appear in the title/description.

Another issue - probably want to remove stop words before doing the similarity calculations, at least with 
get_string_similarity. If you do a bigram comparison, then might not want to remove stopwords in advance.

I notice that, when we have false positives, they are often because the words appear in the title/description, 
and sometimes in the same order, but there is a noun after the query in the title/description that makes it more 
specific and usually wrong for the query. e.g. "road bicycle" rates a road bike bell highly when relevance is actually low.

Note that for the some of our features rely on the fact that every query in training occurs in testing.
Basically, to use these features, you would perform a preprocessing step to X_test to add these 
variables before you run the model. Also note that cross validation on our training set may become 
tricky when you use these variables since they assume you have at least one example of each query.

One of the queries is harleydavison. It is miscategorized by the algorithm - seems to underestimate 
the relevance of the results, because the titles and descriptions always say "Harley Davidson". Consider
making a feature that converts the title to lower case and removes all spaces - then provides an indicator
whether the query exists in the title. Also consider removing dashes '-' - for example, the "spiderman" query
has results with "Spider-man"

Are there repeated query/title combinations? If so, it seems that any time we encounter a known query/title combination
in test set, we should just assign it the rating seen already. Check to see whether this is the case and also check
the variation in ratings among common query/title combos.

Try to incorporate relevance variance - perhaps for each query, insert the average relevance variance. 

Got the following results to test what happens when leaving out variables:
Score: 0.643129109937
Score with query_tokens_in_title removed: 0.63171127765
Score with query_tokens_in_description removed: 0.637368418749
Score with query_length removed: 0.639903085722
Score with percent_query_tokens_in_description removed: 0.635497898503
Score with percent_query_tokens_in_title removed: 0.634468060064
Score with exact_query_in_title removed: 0.639227453057
Score with exact_query_in_description removed: 0.633615569655
Score with space_removed_q_in_t removed: 0.639006304509
Score with space_removed_q_in_d removed: 0.637813906525
Score with q_mean_of_training_relevance removed: 0.642376944202
Score with q_median_of_training_relevance removed: 0.639869449157
Score with closest_title_relevance removed: 0.642020054147
Score with closest_description_relevance removed: 0.638538444249
Score with weighted_title_relevance removed: 0.631477073537
Score with weighted_description_relevance removed: 0.64144060066
Score with weighted_description_relevance_two removed: 0.625971007429
Score with weighted_title_relevance_two removed: 0.612785990654
Score with two_grams_in_q_and_t removed: 0.636179408066
Score with two_grams_in_q_and_d removed: 0.639781626616

Ran it a get increasing the n_estimators from 200 to 300
Score: 0.64339148288
Score with query_tokens_in_title removed: 0.628861951478
Score with query_tokens_in_description removed: 0.644904717005
Score with query_length removed: 0.640712919403
Score with percent_query_tokens_in_description removed: 0.64365615009
Score with percent_query_tokens_in_title removed: 0.641225123491
Score with exact_query_in_title removed: 0.637028267544
Score with exact_query_in_description removed: 0.632414963298
Score with space_removed_q_in_t removed: 0.644115771826
Score with space_removed_q_in_d removed: 0.637551043855
Score with q_mean_of_training_relevance removed: 0.641494919057
Score with q_median_of_training_relevance removed: 0.636043876177
Score with closest_title_relevance removed: 0.642511308549
Score with closest_description_relevance removed: 0.638677893082
Score with weighted_title_relevance removed: 0.627494765937
Score with weighted_description_relevance removed: 0.639336361057
Score with weighted_description_relevance_two removed: 0.631739895519
Score with weighted_title_relevance_two removed: 0.608384307066
Score with two_grams_in_q_and_t removed: 0.642005666464
Score with two_grams_in_q_and_d removed: 0.64068374048

Ran it a get increasing the n_estimators from 300 to 600
Score: 0.643328348999
Score with query_tokens_in_title removed: 0.626235742233
Score with query_tokens_in_description removed: 0.645825578491
Score with query_length removed: 0.640429638798
Score with percent_query_tokens_in_description removed: 0.655858419571
Score with percent_query_tokens_in_title removed: 0.641996682498
Score with exact_query_in_title removed: 0.644583480825
Score with exact_query_in_description removed: 0.639295693177
Score with space_removed_q_in_t removed: 0.648030322296
Score with space_removed_q_in_d removed: 0.635626339377
Score with q_mean_of_training_relevance removed: 0.636089501646
Score with q_median_of_training_relevance removed: 0.635297694001
Score with closest_title_relevance removed: 0.641121446785
Score with closest_description_relevance removed: 0.640939072305
Score with weighted_title_relevance removed: 0.631125731045
Score with weighted_description_relevance removed: 0.636296730253
Score with weighted_description_relevance_two removed: 0.635983988741
Score with weighted_title_relevance_two removed: 0.617787075129
Score with two_grams_in_q_and_t removed: 0.644978566096
Score with two_grams_in_q_and_d removed: 0.642201981003

After removing query_tokens_in_description, percent_query_tokens_in_description
and adding 2gram similarity measures, got the following:

Score: 0.647889581076
Score with query_tokens_in_title removed: 0.645025037265
Score with query_length removed: 0.646184124407
Score with percent_query_tokens_in_title removed: 0.639125677917
Score with exact_query_in_title removed: 0.653587069391
Score with exact_query_in_description removed: 0.643095452699
Score with space_removed_q_in_t removed: 0.657651373962
Score with space_removed_q_in_d removed: 0.648367126302
Score with q_mean_of_training_relevance removed: 0.636816444027
Score with q_median_of_training_relevance removed: 0.647789248918
Score with closest_title_relevance removed: 0.655143239957
Score with closest_2gram_title_relevance removed: 0.643381494965
Score with closest_description_relevance removed: 0.645581181561
Score with closest_2gram_description_relevance removed: 0.646222266638
Score with weighted_title_relevance removed: 0.632236478738
Score with weighted_2gram_title_relevance removed: 0.65008212439
Score with weighted_description_relevance removed: 0.65184426758
Score with weighted_2gram_description_relevance removed: 0.643231180018
Score with weighted_description_relevance_two removed: 0.632722504888
Score with weighted_2gram_description_relevance_two removed: 0.64777286743
Score with weighted_title_relevance_two removed: 0.635804803525
Score with weighted_2gram_title_relevance_two removed: 0.643532450041
Score with two_grams_in_q_and_t removed: 0.646731827421
Score with two_grams_in_q_and_d removed: 0.64879745359

Got the following running the same as above using AdaBoost:
Score: 0.624407669816
Score with query_tokens_in_title removed: 0.624008310817
Score with query_length removed: 0.611598175913
Score with percent_query_tokens_in_title removed: 0.623004140411
Score with exact_query_in_title removed: 0.624407669816
Score with exact_query_in_description removed: 0.624407669816
Score with space_removed_q_in_t removed: 0.624407669816
Score with space_removed_q_in_d removed: 0.620414704021
Score with q_mean_of_training_relevance removed: 0.623470527086
Score with q_median_of_training_relevance removed: 0.622687478923
Score with closest_title_relevance removed: 0.626723458965
Score with closest_2gram_title_relevance removed: 0.610126837778
Score with closest_description_relevance removed: 0.618875244505
Score with closest_2gram_description_relevance removed: 0.621458727383
Score with weighted_title_relevance removed: 0.625559162612
Score with weighted_2gram_title_relevance removed: 0.624407669816
Score with weighted_description_relevance removed: 0.624407669816
Score with weighted_2gram_description_relevance removed: 0.624407669816
Score with weighted_description_relevance_two removed: 0.619711419199
Score with weighted_2gram_description_relevance_two removed: 0.624407669816
Score with weighted_title_relevance_two removed: 0.629721499609
Score with weighted_2gram_title_relevance_two removed: 0.624407669816
Score with two_grams_in_q_and_t removed: 0.624893067111
Score with two_grams_in_q_and_d removed: 0.619573224356

Following from same as above using SVC with scaling

Score: 0.636031415371
Score with query_tokens_in_title removed: 0.634789651152
Score with query_length removed: 0.637833433663
Score with percent_query_tokens_in_title removed: 0.63735259409
Score with exact_query_in_title removed: 0.63727176659
Score with exact_query_in_description removed: 0.628805037119
Score with space_removed_q_in_t removed: 0.634760807479
Score with space_removed_q_in_d removed: 0.631442294622
Score with q_mean_of_training_relevance removed: 0.617972257481
Score with q_median_of_training_relevance removed: 0.63851201616
Score with closest_title_relevance removed: 0.626325183563
Score with closest_2gram_title_relevance removed: 0.621909925799
Score with closest_description_relevance removed: 0.623897495514
Score with closest_2gram_description_relevance removed: 0.636440402046
Score with weighted_title_relevance removed: 0.614831050879
Score with weighted_2gram_title_relevance removed: 0.630909883639
Score with weighted_description_relevance removed: 0.629869466186
Score with weighted_2gram_description_relevance removed: 0.630909883639
Score with weighted_description_relevance_two removed: 0.622480254803
Score with weighted_2gram_description_relevance_two removed: 0.630909883639
Score with weighted_title_relevance_two removed: 0.63236990647
Score with weighted_2gram_title_relevance_two removed: 0.630909883639
Score with two_grams_in_q_and_t removed: 0.639234955175
Score with two_grams_in_q_and_d removed: 0.623194505726
[Finished in 204.9s]