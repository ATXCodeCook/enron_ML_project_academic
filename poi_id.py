#!/usr/bin/env python3
# coding: utf-8

import sys
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
# sys.path.append("../tools/")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from tester import dump_classifier_and_data
import tester

from time import time
import warnings
warnings.filterwarnings('ignore')

## Task 1: Select what features you'll use

# ---------------- BEGIN "Task 1: Select what features you'll use  ----------------

# get f_classif kbest by running StratifiedShuffleSplits (100) and taking
# means of scores. Selection based on scores that are at least 50% of maximum score
# Max score: 17.2   Cutscore: 17.2 * 0.5 = 8.6
# Select features with scores 8.6 and above
# Will be further adjusted during classifier selection (Task 4) as needed.

# df to hold scores (for taking mean for final kbest list)
kbest_scores_df = pd.DataFrame(columns = features_list[1:])

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Changing to np.array for StratifiedShuffleSplit indices
labels = np.array(labels)
features = np.array(features)

sss = StratifiedShuffleSplit(n_splits = 100, test_size = 0.30, random_state = 42)
sss.get_n_splits(features, labels)

for train_index, test_index in sss.split(features, labels):
    features_train, _ = features[train_index], features[test_index]
    labels_train, _ = labels[train_index], labels[test_index]

    # Perform feature selection
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(features_train, labels_train)

    scores_data = selector.scores_
    kbest_scores_df.loc[len(kbest_scores_df)] = scores_data

kbest_scores_df = kbest_scores_df   
kbest_mean_scores = pd.DataFrame(kbest_scores_df.mean().sort_values(ascending = False), columns = ['mean_score'])

plt.figure(figsize = (6, 4), dpi = 150)

sns.barplot(x = kbest_mean_scores.index, y = kbest_mean_scores['mean_score'])
plt.axhline(10, color = '#00daff', linestyle = '--');
plt.xticks(rotation = 90);
kbest_mean_scores
features_sel_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 
                    'bonus_totalpay_ratio', 'salary', 'from_poi_total_from_ratio', 
                    'deferred_income']

# best features for knn verified by scores for KNeighborsClassifier 
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
enron_df = pd.DataFrame.from_dict(data_dict, orient = 'index')

# convert "NaN" stringtype to np.nan type
enron_df.replace('NaN', np.nan, inplace = True)

# ---------------- END "Task 1: Select what features you'll use  ----------------



## Task 2: Remove outliers

# Drop outliers and low features observations (analysis in jupter notebook)
drop_observations_list = ['LOCKHART EUGENE E', 'SCRIMSHAW MATTHEW', 'WHALEY DAVID A', 
                          'CLINE KENNETH W', 'CHAN RONNIE', 'WROBEL BRUCE', 
                          'LOWRY CHARLES P', 'CHRISTODOULOU DIOMEDES', 'GATHMANN WILLIAM D',
                          'GILLIS JOHN', 'THE TRAVEL AGENCY IN THE PARK', "TOTAL"]

try:
    enron_df = enron_df.drop(index = drop_observations_list)
    print('\n', len(drop_observations_list), 'rows dropped. Shape is now:', enron_df.shape)
except:
    print('\nAn error occurred. Rows may have already been dropped. Verify rows are 134.')
    print('Current shape is:', enron_df.shape)
    print('Full code is available in the jupyter notebook.')





## Task 3: Create new feature(s)
# Added 5 engineered features. Two colums dropped (to_messages and from_messages) 
# due to using ratio to combine. 
          
try:
    enron_df['bonus_totalpay_ratio'] = enron_df['bonus'].divide(
                                                        enron_df['total_payments'], 
                                                        fill_value = 0.0 )
          
    enron_df['total_minus_sal'] = enron_df['total_payments'].sub(
                                                            enron_df['salary'], 
                                                            fill_value = 0.0 )
          
    enron_df['exec_stock_tot_ratio'] = enron_df['exercised_stock_options'].divide(
                                                                          enron_df['total_stock_value'], 
                                                                          fill_value = 0.0 )
    enron_df['from_poi_total_from_ratio'] = enron_df['from_this_person_to_poi'].divide(
                                                                                enron_df['from_messages'], 
                                                                                fill_value = 0.0)
    enron_df['shared_rec_to_mes_ratio'] = enron_df['shared_receipt_with_poi'].divide(
                                                                              enron_df['to_messages'], 
                                                                              fill_value = 0.0)
    enron_df.drop(['to_messages', 'from_messages'], axis = 1, inplace=True)
except:
    pass

# Change inf, created due to ratios, to zero
enron_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert nan datatypes to string 'NaN' for tools scripts processing
enron_df.fillna("NaN", inplace = True)
          
# Store to my_dataset for easy export below.
my_dataset = enron_df.to_dict(orient = 'index')
          




## Task 4: Try a varity of classifiers

# ---------------- BEGIN "Task 4: Try a varity of classifiers" Base classifier score comparisions ----------------
          
# Create dict of scores to compare base models with stratified shuffle split and Standard scaler

# check models using cross_val_score

scale_clf_results = {}

n_splits=1 # 50

cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
n_iter = 100

clf_list = [LogisticRegression(max_iter = n_iter),
            GaussianNB(), 
            LinearSVC(max_iter = n_iter * 15), 
            KNeighborsClassifier(n_neighbors = 5), 
            RandomForestClassifier(n_estimators = 30), 
            GradientBoostingClassifier(n_estimators = 150)]

# iteratively checking reduced top features for each classifier
feat_dict ={'features_sel_list_k7': features_sel_list,
            'features_sel_list_k1': features_sel_list[:2],
            'features_sel_list_k2': features_sel_list[:3],
            'features_sel_list_k3': features_sel_list[:4],
            'features_sel_list_k4': features_sel_list[:5],
            'features_sel_list_k5': features_sel_list[:6]}

for clf_type in clf_list:
    for name in feat_dict:
        data = featureFormat(my_dataset, feat_dict[name], sort_keys = True)
        labels, features = targetFeatureSplit(data)
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        clf = clf_type
        
        a_cvs = cross_val_score(clf, features, labels, cv = cv, scoring = 'accuracy', verbose = 0)
        p_cvs = cross_val_score(clf, features, labels, cv = cv, scoring = 'precision', verbose = 0)
        r_cvs = cross_val_score(clf, features, labels, cv = cv, scoring = 'recall', verbose = 0)
        f1_cvs = cross_val_score(clf, features, labels, cv = cv, scoring = 'f1', verbose = 0)
        total_records = len(features)

        means_cvs = [round(a_cvs.mean(), 2),
                     round(p_cvs.mean(), 2),
                     round(r_cvs.mean(),2),
                     round(f1_cvs.mean(),2)]
        
        dict_key = str(clf_type) + '_' + str(name) + '_scaled'
        scale_clf_results[dict_key] = {'accuracy': means_cvs[0],
                                       'precision': means_cvs[1],
                                       'recall': means_cvs[2],
                                       'f1': means_cvs[3],
                                       'total_record': total_records,
                                       'features_count': len(feat_dict[name]) - 1,
                                       'features': feat_dict[name]}
# Compares scores based on classifier, features and scaling (ordered by classifier)
scale_clf_results_df = pd.DataFrame(scale_clf_results).transpose()

# print('Base classifier results with scaling sorted by classifier.')
scale_clf_results_df.sort_index()

# ---------------- END "Task 4: Try a varity of classifiers" Base classifier score comparisions ----------------




# ## Task 5: Tune your classifier to achieve better than .3 precision and recall
 
# The following classifiers were hypertuned and the best classifier chosen based on 
# the order f1, recall and precision values (higher recall preferred over higher precision). 
# 
# Classifiers with tuned parameters:
# Note on KNeighborsClassifier from sklearn docs:  "When p = 1, this is equivalent to using [metric] manhattan_distance (l1)"
# 
# KNeighborsClassifier(metric = 'minkowski', leaf_size=2, n_neighbors=4, p=1, weights='distance')
# RandomForestClassifier(max_depth=100, max_features=None, min_samples_split=3, n_estimators=30)
# GradientBoostingClassifier(learning_rate=0.04, max_depth=9, min_samples_split=15, n_estimators=154)
# GaussianNB(var_smoothing=0.015199110829529336)
#
# Classifier	                  Total Predictions	    Accuracy	Precision	Recall	    F1	    F2
# Best--> KNeighborsClassifier	        13,000	         0.873	       0.640	0.397	  0.490	  0.430
# RandomForestClassifier	            13,000	         0.841	       0.479	0.384	  0.426	  0.400
# GradientBoostingClassifier	        13,000	         0.833	       0.447	0.375	  0.408	  0.387
# GaussianNB	                        13,000	         0.852	       0.531	0.342	  0.416	  0.368
# 
# Scores_variance Testing--> KNeighborsClassifier	   13,000	0.873	0.640	0.397	0.490	0.430
# Scores_variance Testing--> KNeighborsClassifier	  130,000	0.871	0.628	0.392	0.483	0.424
# Scores_variance Testing--> KNeighborsClassifier	1,300,000	0.872	0.637	0.397	0.489	0.430



# ---------------- Begin "Task 5: Tune your classifier to achieve better than .3 precision and recall ----------------
# Grid search cv KNeighborsClassifier

# Base conditions:
# RandomForestClassifier(n_estimators=30)_features_sel_list_k3	0.86	0.46	0.32	0.40	121.0
# GaussianNB()_features_sel_list_k6	0.85	0.47	0.37	0.39	123.0
# GradientBoostingClassifier(n_estimators=150)_features_sel_list_k3	0.84	0.4 	0.34	0.34	121.0
# KNeighborsClassifier()_features_sel_list_k3	0.89	0.55	0.36	0.43	121

n_splits = 1 # 100

features_list = features_sel_list[0:4]
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)

print('\nFeatures used are:\n', features_list, '\n')

# separate the data into features and target (featureFormat converts "NaN" to zeros and has option to drop rows)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

knnc_clf = Pipeline([('scaler',StandardScaler()), 
                    ('knnc', KNeighborsClassifier())])

param_grid = {
              'knnc__metric': ['minkowski'],                 # ['minkowski', 'chebyshev']  --> 'minkowski'
              'knnc__p': [1] ,                                # [1, 2, 3]  --> 1
              'knnc__n_neighbors': [4],                      # np.arange(1, 12, 1)  --> 4
              'knnc__weights': ['distance'],                 # ['uniform', 'distance']  --> 'distance'
              'knnc__leaf_size': [2]                        # np.arange(2, 15, 1)  --> 2
             }

knnc_grid_model = GridSearchCV(estimator = knnc_clf,
                              param_grid = param_grid,
                              scoring = 'f1',
                              cv = cv,
                              verbose = 1)

knnc_grid_model.fit(features, labels)
knnc_labels_predict = knnc_grid_model.predict(features)

# # Final model for gbc
long_run_time = False
if long_run_time:
    for folds in [1000, 10000, 100000]:
        clf = knnc_grid_model.best_estimator_
        clf_scores_dict = tester.test_classifier(clf, my_dataset, features_list, folds = folds)  # 10000, 100000
        clf_tuning_scores_df = clf_tuning_scores_df.append(clf_scores_dict, ignore_index = True)
else:
    clf = knnc_grid_model.best_estimator_
    clf_scores_dict= tester.test_classifier(clf, my_dataset, features_list)
    clf_tuning_scores_df = pd.DataFrame.from_dict(clf_scores_dict, orient = 'index').transpose()
    
    
    
# [output]
# Features used are:
#  ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus'] 

# Fitting 1 folds for each of 1 candidates, totalling 1 fits
# Pipeline(steps=[('scaler', StandardScaler()),
#                 ('knnc',
#                  KNeighborsClassifier(leaf_size=2, n_neighbors=4, p=1,
#                                       weights='distance'))])
# 	Accuracy: 0.87285	Precision: 0.63981	Recall: 0.39700	F1: 0.48997	F2: 0.42961
# 	Total predictions: 13000	True positives:  794	False positives:  447	False negatives: 1206	True negatives: 10553





# ---------------- END "Task 5: Tune your classifier to achieve better than .3 precision and recall ----------------

# ## Task 6: Dump your classifier, dataset, and `features_list` so anyone can

# Check your results. You do not need to change anything below, but make sure
# that the version of `poi_id.py` that you submit can be run on its own and
# generates the necessary `.pkl` files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)