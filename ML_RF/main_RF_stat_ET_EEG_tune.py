import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

#import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True
EQUAL_PERCENTILES = False

SELECTED_FEATURES = "ALL"
#SELECTED_FEATURES = "OCULAR"
#SELECTED_FEATURES = "HEAD"
#SELECTED_FEATURES = "SACCADE"
#SELECTED_FEATURES = "FIXATION"
#SELECTED_FEATURES = "DIAMETER"
#SELECTED_FEATURES = "BLINK"
#SELECTED_FEATURES = "DIAMETER_BLINK"


TIME_INTERVAL_DURATION = 60

features = ['SaccadesNumber', 'SaccadesDuration',
            'FixationNumber', 'FixationDuration']

old_features = [
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll']

statistics = ['mean', 'std', 'min', 'max', 'median']

for feature in old_features:
    for stat in statistics:
        new_feature = feature + '_' + stat
        features.append(new_feature)

np.random.seed(0)

# features: ['SaccadesNumber', 'SaccadesDuration',
#            'FixationNumber', 'FixationDuration',
#            'LeftPupilDiameter', 'RightPupilDiameter',
#            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
#            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
#            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
#            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
#            'HeadHeading', 'HeadPitch', 'HeadRoll']

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
        min,     
        max, 
        median
    ], axis=-1)

    saccades_data = featurized_data[:,4:6]
    fixation_data = featurized_data[:,14:16]
    rest_data = featurized_data[:,20:]
    new_featurized_data = np.concatenate((saccades_data, fixation_data, rest_data), axis=1)
    print("Shape after feature union, before classification:", new_featurized_data.shape)
    return new_featurized_data


def main():
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")
    
    #print(np.isnan(TS_np).any())
    #nan_count = np.count_nonzero(np.isnan(TS_np))
    #print(nan_count)
    
    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # 180 -> (631, 45000, 15), 60 -> (1768, 15000, 15)
    if TIME_INTERVAL_DURATION == 180: 
        TS_np = TS_np.reshape((631, 45000, 15)) # old
    else: # 60
        TS_np = TS_np.reshape((1731, 15000, 17)) #(1731, 15000, 17)

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

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
        #Split into 2 bins by percentile
        eeg_series = pd.Series(scores)
        if EQUAL_PERCENTILES:
            th = eeg_series.quantile(.5)
        else:
            th = eeg_series.quantile(.93)
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
    X_train_featurized = featurize_data(X_train)
    
    X_train_df = pd.DataFrame(X_train_featurized, columns = features)
    
    features3 = ['RightBlinkClosingSpeed_mean',
                 'RightBlinkClosingSpeed_max',
                 'LeftBlinkClosingSpeed_std']
    
    X_train_df3 = X_train_df[features3]
    
    rf = RandomForestClassifier(class_weight=weight_dict, max_depth=3)

    rf.fit(X_train_df3, y_train)
    
    import pickle
    # save the model to disk
    filename = 'rf_binary.sav'
    pickle.dump(rf, open(filename, 'wb')) 

    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
        
    ############################## Predict ####################################

    X_test_featurized = featurize_data(X_test)
    
    X_test_df = pd.DataFrame(X_test_featurized, columns = features)
    
    X_test_df3 = X_test_df[features3]

    y_pred = rf.predict(X_test_df3)
    print("Shape at output after classification:", y_pred.shape)
    
    #print(y_pred)
    #print(y_test)
    
    for i in range(0, len(y_pred)):
        if y_pred[i] == 1:
            if y_pred[i] == y_test[i]:
                print("1")
                print(i)
                break # 0
    class_low_sample = X_test_df3.iloc[i:i+1]
    print(class_low_sample)
    class_low_sample.to_csv("class_low_sample.csv", sep = ' ', header=True, index=False)
    for i in range(0, len(y_pred)):
        if y_pred[i] == 2:
            if y_pred[i] == y_test[i]:
                print("2") #56
                print(i)
                break
    class_high_sample = X_test_df3.iloc[i:i+1]
    class_high_sample.to_csv("class_high_sample.csv", sep = ' ', header=True, index=False)     
    
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
    