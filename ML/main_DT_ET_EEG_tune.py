import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True
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


def main():
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    ###########################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    print(scores)
    
    if BINARY:
        #Split into 2 bins by percentile
        eeg_series = pd.Series(scores)
        th = eeg_series.quantile(.93)
        #th = eeg_series.quantile(.5)
        scores = [1 if score < th else 2 for score in scores]

    else:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        (th1, th2) = eeg_series.quantile([.52, .93])
        #(th1, th2) = eeg_series.quantile([.33, .66])
        scores = [1 if score < th1 else 3 if score > th2 else 2 for score in scores]

    print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
        
    # Spit the data into train and test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        TS_np, scores, test_size=0.1, random_state=0, shuffle=False
        )
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(
        f"Length of train  X : {len(X_train)}\nLength of test X : {len(X_test)}\nLength of train Y : {len(y_train)}\nLength of test Y : {len(y_test)}"
        )

    ################################# Fit #####################################

    classifier = DecisionTreeClassifier(class_weight=weight_dict)

    classifier.fit(X_train, y_train)
    
    ############################## Predict ####################################

    y_pred = classifier.predict(X_test)
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
    