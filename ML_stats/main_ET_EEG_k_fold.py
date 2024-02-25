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


#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True
EQUAL_PERCENTILES = False

#LABEL = "Workload"
LABEL = "Vigilance"
#LABEL = "Stress"

#MODEL = "LR"
#MODEL = "DT"
MODEL = "RF"
#MODEL = "SVC"
#MODEL = "HGBC"

FEATURE_IMPORTANCE = False
TIME_INTERVAL_DURATION = 60

#saccade_fixation = ['SaccadesNumber', 'SaccadesTotalDuration',
features = ['SaccadesNumber', 'SaccadesTotalDuration',
            'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
            'SaccadesDurationMin', 'SaccadesDurationMax',
            'FixationNumber', 'FixationTotalDuration',
            'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
            'FixationDurationMin', 'FixationDurationMax',
            ]

old_features = [
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll']

statistics = ['mean', 'std', 'min', 'max', 'median']
'''
features = []

for feature in saccade_fixation:
    features.append(feature)
'''
for feature in old_features:
    for stat in statistics:
        new_feature = feature + '_' + stat
        features.append(new_feature)


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
    :param x_data: ngumpy array of shape
    (number_of_timeintervals, number_of_timestamps, number_of_features)
    where number_of_timestamps == TIME_INTERVAL_DURATION*250

    :return: featurized numpy array of shape
    (number_of_timeintervals, number_of_new_features)
    where number_of_new_features = 5*number_of_features
    """
    print("Input shape before feature union:", x_data.shape)

    new_data = x_data[:,0,:14]
    feature_to_featurize = x_data[:,:,14:]
    #feature_to_featurize = x_data[:,:,16:] #exclude pupil diameter

    mean = np.mean(feature_to_featurize, axis=-2)
    std = np.std(feature_to_featurize, axis=-2)
    median = np.median(feature_to_featurize, axis=-2)
    min = np.min(feature_to_featurize, axis=-2)
    max = np.max(feature_to_featurize, axis=-2)

    featurized_data = np.concatenate([
        mean,    
        std,     
        min,     
        max, 
        median
    ], axis=-1)

    new_data = np.concatenate((new_data, featurized_data), axis=1)
    print("Shape after feature union, before classification:", new_data.shape)
    return new_data


def main():
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")
    
    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # 180 -> (631, 45000, 15), 60 -> (1768, 15000, 15)
    if TIME_INTERVAL_DURATION == 180: 
        TS_np = TS_np.reshape((631, 45000, 15)) #old
    else: # 60
        TS_np = TS_np.reshape((1731, 15000, 27))

    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    ###########################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)
    
    if LABEL == "Workload":
        scores_np = scores_np[0,:] # WL
    elif LABEL == "Vigilance":
        scores_np = scores_np[1,:] # Vigilance
    else:
        scores_np = scores_np[2,:] # Stress
    

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    #print(scores)
    
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
    
    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = model_selection.KFold(n_splits=num_folds, shuffle=False)
    
    # K-fold Cross Validation model evaluation
    
    # Define per-fold score containers
    acc_per_fold = []
    prec_per_fold = []
    rec_per_fold = []
    f1_per_fold = []
    
    kfold_importances = np.empty(shape=[10, 79])
    
    fold_no = 1
    for train_idx, test_idx in kfold.split(scores):
    
        X_train = np.array(TS_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(TS_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
        
        X_train_featurized = featurize_data(X_train)
        
        X_train_df = pd.DataFrame(X_train_featurized, columns = features)

        ################################# Fit #####################################
        if MODEL == "LR":
            clf = LogisticRegression(class_weight=weight_dict)
            clf.fit(X_train_df, y_train)
        elif  MODEL == "DT":
            clf = DecisionTreeClassifier(class_weight=weight_dict,
                                         random_state=0)
            clf.fit(X_train_df, y_train)
        elif  MODEL == "RF":
            if FEATURE_IMPORTANCE:
                clf = RandomForestClassifier(class_weight=weight_dict,
                                         #bootstrap=False,
                                         max_features=79,
                                         max_depth=79,
                                         n_estimators=102,
                                         random_state=0
                                         )
            else:
                clf = RandomForestClassifier(#class_weight=weight_dict,
                                         class_weight='balanced',
                                         max_depth=39,
                                         n_estimators=102,
                                         random_state=0
                                         )

            clf.fit(X_train_df, y_train)
        
            if FEATURE_IMPORTANCE:
                fold_importances = clf.feature_importances_
                kfold_importances[fold_no-1, :] = fold_importances
        
        elif MODEL == "SVC":
            clf = SVC(class_weight=weight_dict)
            clf.fit(X_train_df, y_train)
        elif  MODEL == "HGBC":
            clf = HistGradientBoostingClassifier(class_weight='balanced',
                                                 random_state=0)
            clf.fit(X_train_df, y_train)
        ############################## Predict ####################################
        
        X_test_featurized = featurize_data(X_test)
        
        X_test_df = pd.DataFrame(X_test_featurized, columns = features)

        y_pred = clf.predict(X_test_df)
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
        
        acc_per_fold.append(accuracy)
        prec_per_fold.append(precision)
        rec_per_fold.append(recall)
        f1_per_fold.append(f1)
                
        # Increase fold number
        fold_no = fold_no + 1

    print(acc_per_fold)
    print(prec_per_fold)
    print(rec_per_fold)
    print(f1_per_fold)
    
    print(mean(acc_per_fold))
    print(mean(f1_per_fold))
    
    if FEATURE_IMPORTANCE:
        importances_mean = np.mean(kfold_importances, axis=0)
        
        importances_df = pd.DataFrame()
        importances_df['feature'] = features
        importances_df['importance'] = importances_mean
    
        importances_df.sort_values(by=['importance'], ascending=False,inplace=True)
        importances_df.to_csv("forest_importances.csv", sep=',', header=True, index=False)

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    