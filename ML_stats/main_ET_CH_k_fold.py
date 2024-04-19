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


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = False

#MODEL = "LR"
#MODEL = "DT"
#MODEL = "RF"
#MODEL = "SVC"
MODEL = "HGBC"

saccade_fixation = [
            'SaccadesNumber', 'SaccadesTotalDuration',
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

features = []
for feature in saccade_fixation:
    features.append(feature)
for stat in statistics:
    for feature in old_features:
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

    # Formula used:
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
    """
    print("Input shape before feature union:", x_data.shape)
    
    new_data = x_data[:,0,:14]
    print(new_data.shape)
    
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
    
    full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")

    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, 180*250, number_of_features)
    # (667, 45000, 27)
    TS_np = TS_np.reshape((667, 45000, 27))
    print(TS_np[1,1,:])

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")


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
    
    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = model_selection.KFold(n_splits=num_folds, shuffle=False)
    
    # K-fold Cross Validation model evaluation
    
    # Define per-fold score containers
    acc_per_fold = []
    prec_per_fold = []
    rec_per_fold = []
    f1_per_fold = []
    
    fold_no = 1
    for train_idx, test_idx in kfold.split(scores):
    
        X_train = np.array(TS_np)[train_idx.astype(int)]
        y_train = np.array(scores)[train_idx.astype(int)]
        X_test = np.array(TS_np)[test_idx.astype(int)]
        y_test = np.array(scores)[test_idx.astype(int)]
        
        #normalize train set
        scaler = preprocessing.MinMaxScaler()
        X_train_shape = X_train.shape    # save the shape
        X_train = X_train.reshape(-1, X_train_shape[2])
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = X_train.reshape(X_train_shape)    # restore the shape
        
        #normalize test set
        X_test_shape = X_test.shape    # save the shape
        X_test = X_test.reshape(-1, X_test_shape[2])
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape(X_test_shape)    # restore the shape
        
        
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
            clf = RandomForestClassifier(
                                         class_weight='balanced',
                                         #max_depth=md,
                                         #n_estimators=ne,
                                         random_state=0
                                         )

            clf.fit(X_train_df, y_train)
        
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

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    