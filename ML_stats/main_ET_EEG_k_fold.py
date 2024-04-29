import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
from statistics import mean
#import sys

from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.inspection import permutation_importance

from sklearn.model_selection import RandomizedSearchCV#, GridSearchCV

from scipy.stats import randint
#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True
EQUAL_PERCENTILES = False
RANDOM_SEARCH = False

LABEL = "Workload"
#LABEL = "Vigilance"
#LABEL = "Stress"

#MODEL = "LR"
#MODEL = "DT"
MODEL = "RF"
#MODEL = "SVC"
#MODEL = "HGBC"

FEATURE_IMPORTANCE = False
TIME_INTERVAL_DURATION = 60


np.random.seed(0)

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
    
    if TIME_INTERVAL_DURATION == 60:
        filename = "ML_features_1min.csv"
    else:
        filename = "ML_features_3min.csv"
    
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
            else: #Stress
                th = eeg_series.quantile(.1)
        scores = [1 if score < th else 2 for score in scores]

    else:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            (th1, th2) = eeg_series.quantile([.33, .66])
        else:
            (th1, th2) = eeg_series.quantile([.52, .93])
        scores = [1 if score < th1 else 3 if score > th2 else 2 for score in scores]
    
    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
    '''
    
    # Define the K-fold Cross Validator
    num_folds = 10

    kfold = model_selection.KFold(n_splits=num_folds, shuffle=True)
    #kfold = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=False)
    
    # K-fold Cross Validation model evaluation
    
    # Define per-fold score containers
    acc_per_fold = []
    prec_per_fold = []
    rec_per_fold = []
    f1_per_fold = []
    f1_macro_per_fold = []
    
    if FEATURE_IMPORTANCE:
        gini_kfold_importances = np.empty(shape=[10, 79])
        perm_kfold_importances = np.empty(shape=[10, 79])
    
    #occular:
    #gini_kfold_importances = np.empty(shape=[10, 64])
    #perm_kfold_importances = np.empty(shape=[10, 64])

    #saccade_fixation:
    #gini_kfold_importances = np.empty(shape=[10, 14])
    #perm_kfold_importances = np.empty(shape=[10, 14])
    
    fold_no = 1
    for train_idx, test_idx in kfold.split(features_np):
    
        X_train = np.array(features_np)[train_idx.astype(int)]
        
        #normalize
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        y_train = np.array(scores)[train_idx.astype(int)]
        
        if  BINARY:
            th = getEEGThreshold(y_train)
            y_train = [1 if score < th else 2 for score in y_train]
        else:
            (th1, th2) = getEEGThresholds(y_train)
            y_train = [1 if score < th1 else 3 if score > th2 else 2 for score in y_train]
        #number_of_classes = len(set(y_train))
         
        weight_dict = weight_classes(y_train)
        
        
        X_test = np.array(features_np)[test_idx.astype(int)]
        
        #normalize
        X_test = scaler.transform(X_test)
        
        y_test = np.array(scores)[test_idx.astype(int)]
        
        if  BINARY:
            y_test = [1 if score < th else 2 for score in y_test]
        else:
            y_test = [1 if score < th1 else 3 if score > th2 else 2 for score in y_test]

        
        ################################# Fit #####################################
        if MODEL == "LR":
            clf = LogisticRegression(class_weight=weight_dict)
            clf.fit(X_train, y_train)
            
        elif  MODEL == "DT":
            clf = DecisionTreeClassifier(class_weight=weight_dict,
                                         random_state=0)
            clf.fit(X_train, y_train)
            
        elif  MODEL == "RF":
            clf = RandomForestClassifier(#class_weight=weight_dict,
                             class_weight='balanced',
                             #bootstrap=False,
                             max_features=None,
                             random_state=0)

            if RANDOM_SEARCH:
                
                # Use random search to find the best hyperparameters
                param_dist = {'n_estimators': randint(50,500),
                              'max_depth': randint(1,79),
                              }
                
                search = RandomizedSearchCV(clf,
                                        param_distributions = param_dist,
                                        n_iter=5,
                                        cv=10)
                
                '''
                param_grid = {'n_estimators': np.arange(100, 150, dtype=int),
                              'max_depth': np.arange(1, 79, dtype=int),
                              }
                search = GridSearchCV(clf, param_grid=param_grid, cv=10)
                '''
                
                # Fit the search object to the data
                search.fit(X_train, y_train)
                
                # Create a variable for the best model
                clf = search.best_estimator_

                # Print the best hyperparameters
                #print('Best hyperparameters:',  search.best_params_)
            else:
                
                clf.fit(X_train, y_train)
        
        elif MODEL == "SVC":
            clf = SVC(class_weight=weight_dict)
            clf.fit(X_train, y_train)
            
        elif  MODEL == "HGBC":
            clf = HistGradientBoostingClassifier(class_weight='balanced',
                                                 random_state=0)
            clf.fit(X_train, y_train)
            
        ############################## Predict ####################################
        
        y_pred = clf.predict(X_test)
        print("Shape at output after classification:", y_pred.shape)
    
    
        if MODEL == "RF":
            if FEATURE_IMPORTANCE:
                gini_fold_importances = clf.feature_importances_
                gini_kfold_importances[fold_no-1, :] = gini_fold_importances
                
                perm_fold_importances = permutation_importance(
                    clf, X_test, y_test, n_repeats=10, random_state=0, n_jobs=2
                    )
                perm_kfold_importances[fold_no-1, :] = perm_fold_importances.importances_mean
                
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
        
        #recall_macro = recall_score(y_pred=y_pred, y_true=y_test, average='macro')
        #precision_macro = precision_score(y_pred=y_pred, y_true=y_test, average='macro')
            
        print("Accuracy:", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-score:", f1)
        print("F1-score macro:", f1_macro)
        
        
        #print(f"Train accuracy: {clf.score(X_train_df, y_train):.3f}")
        #print(f"Test accuracy: {clf.score(X_test_df, y_test):.3f}")
        
        acc_per_fold.append(accuracy)
        prec_per_fold.append(precision)
        rec_per_fold.append(recall)
        f1_per_fold.append(f1)
        f1_macro_per_fold.append(f1_macro)
                
        # Increase fold number
        fold_no = fold_no + 1

    print(acc_per_fold)
    print(prec_per_fold)
    print(rec_per_fold)
    print(f1_per_fold)
    print(f1_macro_per_fold)
    
    print(mean(acc_per_fold))
    print(mean(f1_per_fold))
    print(mean(f1_macro_per_fold))

    '''    
    if FEATURE_IMPORTANCE:
        importances_mean = np.mean(gini_kfold_importances, axis=0)
        
        importances_df = pd.DataFrame()
        importances_df['feature'] = features
        #importances_df['feature'] = occular_features
        #importances_df['feature'] = saccade_fixation
        importances_df['importance'] = importances_mean
    
        importances_df.sort_values(by=['importance'], ascending=False,inplace=True)
        importances_df.to_csv("forest_importances_gini.csv", sep=',', header=True, index=False)
        
        importances_mean = np.mean(perm_kfold_importances, axis=0)
        
        importances_df = pd.DataFrame()
        importances_df['feature'] = features
        #importances_df['feature'] = occular_features
        #importances_df['feature'] = saccade_fixation
        importances_df['importance'] = importances_mean
    
        importances_df.sort_values(by=['importance'], ascending=False,inplace=True)
        importances_df.to_csv("forest_importances_perm.csv", sep=',', header=True, index=False)
      '''  

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    