import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
#import sys

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True

#SELECTED_FEATURES = "ALL"
#SELECTED_FEATURES = "OCULAR"
#SELECTED_FEATURES = "HEAD"
#SELECTED_FEATURES = "SACCADE"
#SELECTED_FEATURES = "FIXATION"
#SELECTED_FEATURES = "DIAMETER"
#SELECTED_FEATURES = "BLINK"
SELECTED_FEATURES = "DIAMETER_BLINK"


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

    # Formula used - Naive method where
    # weight = 1 - (no. of samples present / total no. of samples)
    # So more the samples, lower the weight

    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
    print(weight_dict)
        
    return weight_dict


def featurize_data(x_data):
    """
    :param x_data: numpy array of shape
    (number_of_timeintervals, number_of_timestamps, number_of_features)
    where number_of_timestamps == TIME_INTERVAL_DURATION*250

    :return: featurized numpy array of shape
    (number_of_timeintervals, number_of_new_features)
    where number_of_new_features = 5*number_of_features
    """
    print("Input shape before feature union:", x_data.shape)

    mean = np.mean(x_data, axis=-2)
    std = np.std(x_data, axis=-2)
    median = np.median(x_data, axis=-2)
    min = np.min(x_data, axis=-2)
    max = np.max(x_data, axis=-2)

    featurized_data = np.concatenate([
        mean,
        std,
        median,
        min,
        max,
    ], axis=-1)

    print("Shape after feature union, before classification:", featurized_data.shape)
    return featurized_data


def main():
    
    full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")

    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, 180*250, number_of_features)
    # (667, 45000, 17)
    TS_np = TS_np.reshape((667, 45000, 17))

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    if SELECTED_FEATURES == "OCULAR":
        TS_np = TS_np [:,:,0:14]
    elif SELECTED_FEATURES == "HEAD":
        TS_np = TS_np [:,:,14:17]
    elif SELECTED_FEATURES == "SACCADE":
        TS_np = TS_np [:,:,0:2]
    elif SELECTED_FEATURES == "FIXATION":
        TS_np = TS_np [:,:,2:4]
    elif SELECTED_FEATURES == "DIAMETER":
        TS_np = TS_np [:,:,4:6]
    elif SELECTED_FEATURES == "BLINK":
        TS_np = TS_np [:,:,6:14]
    elif SELECTED_FEATURES == "DIAMETER_BLINK":
        TS_np = TS_np [:,:,4:14]

    ###########################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    
    if BINARY:
        scores = [1 if score < 4 else 2 for score in scores]
    else:
        scores = [1 if score < 2 else 3 if score > 3 else 2 for score in scores]

    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
        
    # Spit the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        TS_np, scores, test_size=0.1, random_state=0, shuffle=False
        )
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(
        f"Length of train  X : {len(X_train)}\nLength of test X : {len(X_test)}\nLength of train Y : {len(y_train)}\nLength of test Y : {len(y_test)}"
        )

    ################################# Fit #####################################
    X_train_featurized = featurize_data(X_train)

    classifier = RandomForestClassifier(class_weight=weight_dict)

    #TODO: add second parameter
    param_dist = {'max_depth': randint(1,20)}
    
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(classifier, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

    # Fit the random search object to the data
    rand_search.fit(X_train_featurized, y_train)   

    #classifier.fit(X_train_featurized, y_train)
    
    # Create a variable for the best model
    best_dt = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    #Binary: {'max_depth': 19}
    #3 classes: {'max_depth': 11}
    
    ############################## Predict ####################################

    X_test_featurized = featurize_data(X_test)

    #y_pred = classifier.predict(X_test_featurized)
    y_pred = best_dt.predict(X_test_featurized)
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
    print("Accuracy:", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score:", f1)

    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    