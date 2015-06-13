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

Performed Stratified K-Fold cross validation, with strata determined by query. Interestingly, this takes almost
a half hour to run. It's running on 1/3 of the data of the full run so this doesn't really make much sense.

TO DO:
Add and test features (see below)
Add comments to your code (so other can undersetand and also so you can understand - your
code borrows a lot from others)
Adjust cross validation so it always includes a proportionate number of each query
Incorporate porter stemmer code https://www.kaggle.com/duttaroy/crowdflower-search-relevance/porter-stemmer/run/11533
Incorporate stop words below after getting basic stemming working.
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','a','the']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
Look for algorithms for determining the similarity between two sentences.
Check out recommender system algorithms that may be of use http://en.wikipedia.org/wiki/Recommender_system
Perhaps consider ensembling with a model that uses bag of words technique. Look up ensembling methods - 
this will be a last step, but don't want to be rushed.



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