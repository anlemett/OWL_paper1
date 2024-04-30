import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import ShuffleSplit
from scipy.stats import randint
from sklearn import preprocessing

#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = True
EQUAL_PERCENTILES = False

#MODEL = "LR"
#MODEL = "DT"
MODEL = "RF"
#MODEL = "SVC"
#MODEL = "HGBC"

LABEL = "Workload"
#LABEL = "Vigilance"
#LABEL = "Stress"

TIME_INTERVAL_DURATION = 60

np.random.seed(RANDOM_STATE)

def weight_classes(scores):
    
    vals_dict = {}
    for i in scores:
        if i in vals_dict.keys():
            vals_dict[i] += 1
        else:
            vals_dict[i] = 1
    total = sum(vals_dict.values())

    # Formula used:
    # weight = 1 - (no. of samples present / total no. of samples)
    # So more the samples, lower the weight

    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
    print(weight_dict)
        
    return weight_dict

def getEEGThreshold(scores):
    #Split into 2 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        th = eeg_series.quantile(.5)
    else:
        if LABEL == "Workload":
            th = eeg_series.quantile(.93)
        elif LABEL == "Vigilance":
            th = eeg_series.quantile(.1)
        else: #Stress
            th = eeg_series.quantile(.1)
    return th

def getEEGThresholds(scores):
    #Split into 3 bins by percentile
    eeg_series = pd.Series(scores)
    if EQUAL_PERCENTILES:
        (th1, th2) = eeg_series.quantile([.33, .66])
    else:
        (th1, th2) = eeg_series.quantile([.52, .93])
    return (th1, th2)


def main():
    
    filename = "ML_features_1min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    features_np = data_df.to_numpy()

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")
    

    ###########################################################################
    #Shuffle data

    print(features_np.shape)
    print(scores_np.shape)
    
    if LABEL == "Workload":
        scores_np = scores_np[0,:] # WL
    elif LABEL == "Vigilance":
        scores_np = scores_np[1,:] # Vigilance
    else:
        scores_np = scores_np[2,:] # Stress

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    features_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    '''
    if BINARY:
        #Split into 2 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            th = eeg_series.quantile(.5)
        else:
            if LABEL == "Workload":
                th = eeg_series.quantile(.93)
            elif LABEL == "Vigilance":
                th = eeg_series.quantile(.1)
            else:
                th = eeg_series.quantile(.9)
        scores = [1 if score < th else 2 for score in scores]

    else:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            (th1, th2) = eeg_series.quantile([.33, .66])
        else:
            (th1, th2) = eeg_series.quantile([.52, .93])
            #(th1, th2) = eeg_series.quantile([.7, .48])
        scores = [1 if score < th1 else 3 if score > th2 else 2 for score in scores]

    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
    '''
    
    print(type(features_np))
    features_np = np.array(features_np)
    #print(type(TS_np))
    #X = featurize_data(TS_np)
    
    #X_df = pd.DataFrame(X, columns = features)
    
    # Spit the data into train and test
    '''
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, scores, test_size=0.1, shuffle=True
        )
    '''
    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=0)
    
    for i, (train_idx, test_idx) in enumerate(rs.split(features_np)):
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
    
    #normalize train set
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
       
    if  BINARY:
        th = getEEGThreshold(y_train)
        y_train = [1 if score < th else 2 for score in y_train]
    else:
        (th1, th2) = getEEGThresholds(y_train)
        y_train = [1 if score < th1 else 3 if score > th2 else 2 for score in y_train]
    #number_of_classes = len(set(y_train))
    
    weight_dict = weight_classes(y_train)
    
    #normalize test set
    X_test = scaler.transform(X_test)
    
    if  BINARY:
        y_test = [1 if score < th else 2 for score in y_test]
    else:
        y_test = [1 if score < th1 else 3 if score > th2 else 2 for score in y_test]

    ################################# Fit #####################################

    if MODEL == "LR":
        clf = LogisticRegression(class_weight=weight_dict)
        #clf.fit(X_train_df, y_train)
        clf.fit(X_train, y_train)
                
    elif MODEL == "SVC":
        clf = SVC(class_weight=weight_dict)
        #clf.fit(X_train_df, y_train)
        clf.fit(X_train, y_train)
        
    elif  MODEL == "DT":
        clf = DecisionTreeClassifier(class_weight=weight_dict)
        #clf.fit(X_train_df, y_train)
        clf.fit(X_train, y_train)
        
    elif  MODEL == "RF":
        clf = RandomForestClassifier(class_weight=weight_dict,
                                     #bootstrap=False,
                                     max_features=None,
                                     random_state=RANDOM_STATE)
        
        # Use random search to find the best hyperparameters
        param_dist = {'n_estimators': randint(50,500),
             'max_depth': randint(1,79),
             }
        
        search = RandomizedSearchCV(clf, 
                                param_distributions = param_dist,
                                #scoring = 'f1_macro',
                                n_iter=10, 
                                cv=10,
                                random_state=RANDOM_STATE)
        '''
        param_grid = {'n_estimators': np.arange(100, 150, dtype=int),
             'max_depth': np.arange(1, 79, dtype=int),
             }
        search = GridSearchCV(clf, param_grid=param_grid, cv=10)
        '''
        # Fit the search object to the data
        #search.fit(X_train_df, y_train)
        print("Before fit")
        search.fit(X_train, y_train)
        print("After fit")
 
        # Create a variable for the best model
        best_rf = search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  search.best_params_)
        #WL, n_iter=10: {'max_depth': , 'n_estimators': }
        
    elif  MODEL == "HGBC":
        clf = HistGradientBoostingClassifier(class_weight='balanced')
        #clf.fit(X_train_df, y_train)
        clf.fit(X_train, y_train)
    
    #importances = clf.feature_importances_
    #print(type(importances)) # class 'numpy.ndarray' 1x79
    
    ############################## Predict ####################################
    
    if  MODEL == "RF":
        #y_pred = best_rf.predict(X_test_df)
        y_pred = best_rf.predict(X_test)
    else:
        #y_pred = clf.predict(X_test_df)
        y_pred = clf.predict(X_test)

    print("Shape at output after classification:", y_pred.shape)

    ############################ Evaluate #####################################
    
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    
    if BINARY:
        precision = precision_score(y_pred=y_pred, y_true=y_test, average='binary')
        recall = recall_score(y_pred=y_pred, y_true=y_test, average='binary')
        f1 = f1_score(y_pred=y_pred, y_true=y_test, average='binary')
    else:
        f1 = f1_score(y_pred=y_pred, y_true=y_test, average='micro')
        recall = recall_score(y_pred=y_pred, y_true=y_test, average='micro')
        precision = precision_score(y_pred=y_pred, y_true=y_test, average='micro')
        
    f1_macro = f1_score(y_pred=y_pred, y_true=y_test, average='macro')
    
    print("Accuracy:", accuracy)
    #print("Precision: ", precision)
    #print("Recall: ", recall)
    #print("F1-score:", f1)
    print("Macro F1-score:", f1_macro)
    
    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    