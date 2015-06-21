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

Try ensembling models that perform badly with ones that perform well - e.g. RandomForest with class_weight = 'auto', along with
RandomForest with default class_weight = None. It seems like it may be good to throw in at least one model in the ensemble that
puts a lot of weight on median_rating = 1 or 2

Try using gradient boosting - an ensemble method that allows you to use a custom loss function - insert the kappa loss - http://scikit-learn.org/dev/modules/ensemble.html#gradient-tree-boosting

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

Try looking into the implications of having "0" as a default value for some features (particularly the 
k-nearest measures). See what happens when you impute the overall median relevance or something like that.

Important to consider that people search a particular query for a purpose that may vary by person. For example,
people looking for 

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

Following output running modelling.py on 6/17/2015

Begin random forest model
Score 1: 0.665458244217
Score 2: 0.685827198148
Score 3: 0.664815233248
Score 4: 0.679087900115
Score 5: 0.687667811831
Average score: 0.676571277512
Begin SVC model
Score 1: 0.650030531348
Score 2: 0.678222922425
Score 3: 0.654738474998
Score 4: 0.653206776354
Score 5: 0.665430738888
Average score: 0.660325888803
Begin AdaBoost model
Score 1: 0.62612860127
Score 2: 0.662701980327
Score 3: 0.618708504307
Score 4: 0.655131017659
Score 5: 0.649796914633
Average score: 0.642493403639
Begin TFIDF v1 model
Score 1: 0.567270816228
Score 2: 0.607283145266
Score 3: 0.604396831139
Score 4: 0.595896181024
Score 5: 0.600472858411
Average score: 0.595063966414
Begin TFIDF v2 model
Score 1: 0.549181036341
Score 2: 0.590604987578
Score 3: 0.5944862103
Score 4: 0.599271968334
Score 5: 0.588588306131
Average score: 0.584426501737
[Finished in 1242.8s]


Output from 6/20/2015 2:16am

Begin random forest model
Score 1: 0.661548611514
Score 2: 0.697730878818
Score 3: 0.67120255031
Score 4: 0.681020206393
Score 5: 0.694796050327
Average score: 0.681259659472
Score with query_tokens_in_title removed
Score 1: 0.668507245733
Score 2: 0.703017779436
Score 3: 0.669397616485
Score 4: 0.682530521779
Score 5: 0.692391894133
Average score: 0.683169011513
Score with query_length removed
Score 1: 0.669167497667
Score 2: 0.698858436491
Score 3: 0.667751671749
Score 4: 0.679918580243
Score 5: 0.68836682333
Average score: 0.680812601896
Score with percent_query_tokens_in_title removed
Score 1: 0.67499685331
Score 2: 0.70064248987
Score 3: 0.668152170943
Score 4: 0.676320727765
Score 5: 0.686402658118
Average score: 0.681302980001
Score with exact_query_in_title removed
Score 1: 0.668384228521
Score 2: 0.696538687537
Score 3: 0.669123204001
Score 4: 0.680461836666
Score 5: 0.687231435589
Average score: 0.680347878463
Score with exact_query_in_description removed
Score 1: 0.667745216642
Score 2: 0.698031518388
Score 3: 0.663664948163
Score 4: 0.681092067294
Score 5: 0.698567815856
Average score: 0.681820313268
Score with space_removed_q_in_t removed
Score 1: 0.667463386566
Score 2: 0.702582787743
Score 3: 0.662864801277
Score 4: 0.681600692386
Score 5: 0.687357921291
Average score: 0.680373917853
Score with space_removed_q_in_d removed
Score 1: 0.662605142479
Score 2: 0.70022924982
Score 3: 0.666282356045
Score 4: 0.681359699405
Score 5: 0.687230311327
Average score: 0.679541351815
Score with q_mean_of_training_relevance removed
Score 1: 0.678492227745
Score 2: 0.691813785614
Score 3: 0.665786817831
Score 4: 0.680859454456
Score 5: 0.689009643213
Average score: 0.681192385772
Score with q_median_of_training_relevance removed
Score 1: 0.672820814203
Score 2: 0.697400619942
Score 3: 0.670651778651
Score 4: 0.688134489993
Score 5: 0.68998537795
Average score: 0.683798616148
Score with closest_title_relevance removed
Score 1: 0.67179439342
Score 2: 0.695523890651
Score 3: 0.661074787376
Score 4: 0.684337809457
Score 5: 0.700824399389
Average score: 0.682711056058
Score with closest_2gram_title_relevance removed
Score 1: 0.671812287743
Score 2: 0.707816031976
Score 3: 0.659675799158
Score 4: 0.68541588891
Score 5: 0.686895209446
Average score: 0.682323043447
Score with closest_description_relevance removed
Score 1: 0.674687281891
Score 2: 0.700788516554
Score 3: 0.671302972305
Score 4: 0.677551172035
Score 5: 0.695985110956
Average score: 0.684063010748
Score with closest_2gram_description_relevance removed
Score 1: 0.679450169354
Score 2: 0.699025540826
Score 3: 0.677200973898
Score 4: 0.686728041351
Score 5: 0.695781198657
Average score: 0.687637184817
Score with weighted_title_relevance removed
Score 1: 0.663436548739
Score 2: 0.700788109851
Score 3: 0.656008316418
Score 4: 0.678020390305
Score 5: 0.692232440104
Average score: 0.678097161084
Score with weighted_2gram_title_relevance removed
Score 1: 0.665021429654
Score 2: 0.701349791754
Score 3: 0.669149323584
Score 4: 0.680335502174
Score 5: 0.69243112186
Average score: 0.681657433805
Score with weighted_description_relevance removed
Score 1: 0.667702544355
Score 2: 0.702454158522
Score 3: 0.666124320654
Score 4: 0.679953422723
Score 5: 0.691686007257
Average score: 0.681584090702
Score with weighted_2gram_description_relevance removed
Score 1: 0.671694800573
Score 2: 0.699764414967
Score 3: 0.668374879552
Score 4: 0.683485414294
Score 5: 0.693449653695
Average score: 0.683353832616
Score with weighted_description_relevance_two removed
Score 1: 0.661446636586
Score 2: 0.698285202119
Score 3: 0.669584806721
Score 4: 0.663825166532
Score 5: 0.688770901279
Average score: 0.676382542647
Score with weighted_2gram_description_relevance_two removed
Score 1: 0.666391441136
Score 2: 0.701099887704
Score 3: 0.670205814914
Score 4: 0.68050733228
Score 5: 0.692772529462
Average score: 0.682195401099
Score with weighted_title_relevance_two removed
Score 1: 0.654784700839
Score 2: 0.692723675553
Score 3: 0.655689787282
Score 4: 0.685460565328
Score 5: 0.677948798002
Average score: 0.673321505401
Score with weighted_2gram_title_relevance_two removed
Score 1: 0.66402314213
Score 2: 0.697629890832
Score 3: 0.669892072125
Score 4: 0.669807691713
Score 5: 0.703127867129
Average score: 0.680896132786
Score with two_grams_in_q_and_t removed
Score 1: 0.67411189638
Score 2: 0.697174629839
Score 3: 0.675432274095
Score 4: 0.682974656837
Score 5: 0.690471634839
Average score: 0.684033018398
Score with two_grams_in_q_and_d removed
Score 1: 0.676064208909
Score 2: 0.702844752694
Score 3: 0.665658076039
Score 4: 0.679986786551
Score 5: 0.689401141435
Average score: 0.682790993126
Score with avg_relevance_variance removed
Score 1: 0.665458244217
Score 2: 0.685827198148
Score 3: 0.664815233248
Score 4: 0.679087900115
Score 5: 0.687667811831
Average score: 0.676571277512
[Finished in 702.9s]

Random forest feature_importances_ (the higher, the more important the feature)
query_tokens_in_title 0.0493135460242
query_length 0.016727936699
percent_query_tokens_in_title 0.0445457094559
exact_query_in_title 0.00808707798978
exact_query_in_description 0.00409817747511
space_removed_q_in_t 0.00819820508917
space_removed_q_in_d 0.00448674494775
q_mean_of_training_relevance 0.101262287132
q_median_of_training_relevance 0.0335079180183
closest_title_relevance 0.0714336656926
closest_2gram_title_relevance 0.063314279969
closest_description_relevance 0.0236297579873
closest_2gram_description_relevance 0.0247557774187
weighted_title_relevance 0.0385371869373
weighted_2gram_title_relevance 0.0633025243363
weighted_description_relevance 0.0219616189154
weighted_2gram_description_relevance 0.0466448156278
weighted_description_relevance_two 0.0548357652644
weighted_2gram_description_relevance_two 0.0465966246839
weighted_title_relevance_two 0.109485362169
weighted_2gram_title_relevance_two 0.0709558503641
two_grams_in_q_and_t 0.0127882021587
two_grams_in_q_and_d 0.0080009282503
avg_relevance_variance 0.0735300373939
