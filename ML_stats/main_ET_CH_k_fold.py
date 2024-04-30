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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV#, GridSearchCV

from scipy.stats import randint


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0

BINARY = True
RANDOM_SEARCH = False

#MODEL = "LR"
MODEL = "SVC"
#MODEL = "DT"
#MODEL = "RF"
#MODEL = "HGBC"


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


def main():
    
    filename = "ML_features_3min.csv"
    
    full_filename = os.path.join(ML_DIR, filename)
    
    data_df = pd.read_csv(full_filename, sep=' ')
    
    features_np = data_df.to_numpy()


    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")


    ###########################################################################
    #Shuffle data

    print(features_np.shape)
    print(scores_np.shape)

    zipped = list(zip(features_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    
    if BINARY:
        scores = [0 if score < 4 else 1 for score in scores]
    else:
        scores = [1 if score < 2 else 3 if score > 3 else 2 for score in scores]

    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
    
    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = model_selection.KFold(n_splits=num_folds, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    
    # Define per-fold score containers
    acc_per_fold = []
    prec_per_fold = []
    rec_per_fold = []
    f1_per_fold = []
    f1_macro_per_fold = []
    
    fold_no = 1
    for train_idx, test_idx in kfold.split(scores):
    
        X_train = np.array(features_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(features_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
        
        #normalize train set
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        #normalize test set
        X_test = scaler.transform(X_test)
        
        
        ################################# Fit #####################################

        if MODEL == "LR":
            clf = LogisticRegression(class_weight=weight_dict)
            clf.fit(X_train, y_train)
            
        elif MODEL == "SVC":
            clf = SVC(class_weight=weight_dict)
            clf.fit(X_train, y_train)

        elif  MODEL == "DT":
            clf = DecisionTreeClassifier(class_weight=weight_dict,
                                         random_state=0)
            clf.fit(X_train, y_train)
        elif  MODEL == "RF":
            clf = RandomForestClassifier(
                                         class_weight='balanced',
                                         #max_depth=md,
                                         #n_estimators=ne,
                                         random_state=0
                                         )

            if RANDOM_SEARCH:
                
                # Use random search to find the best hyperparameters
                param_dist = {'n_estimators': randint(50,500),
                              'max_depth': randint(1,79),
                              }
                
                search = RandomizedSearchCV(clf,
                                        param_distributions = param_dist,
                                        scoring = 'f1_macro',
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

        elif  MODEL == "HGBC":
            clf = HistGradientBoostingClassifier(class_weight='balanced',
                                                 random_state=0)
            clf.fit(X_train, y_train)
            
    
        ############################## Predict ####################################
                
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
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-score:", f1)
        print("F1-score macro:", f1_macro)
        
        print(classification_report(y_test, y_pred, digits=4))
        
        matrix = confusion_matrix(y_test, y_pred)
        print(matrix)
        
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

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    