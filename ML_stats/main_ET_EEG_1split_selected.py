import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
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

LABEL = "Workload"
#LABEL = "Vigilance"

#MODEL = "LR"
#MODEL = "DT"
MODEL = "RF"
#MODEL = "SVC"
#MODEL = "HGBC"

VISUALIZE = True

TIME_INTERVAL_DURATION = 60

features = ['SaccadesNumber', 'SaccadesTotalDuration',
            'SaccadesDurationMean', 'SaccadesDurationStd', 'SaccadesDurationMedian',
            'SaccadesDurationMin', 'SaccadesDurationMax',
            'FixationNumber', 'FixationTotalDuration',
            'FixationDurationMean', 'FixationDurationStd', 'FixationDurationMedian',
            'FixationDurationMin', 'FixationDurationMax',
            ]
saccade_fixation = features

old_features = [
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll']

statistics = ['mean', 'std', 'min', 'max', 'median']

for stat in statistics:
    for feature in old_features:
        new_feature = feature + '_' + stat
        features.append(new_feature)


blinks_head = []
blinks_head_old = old_features[2:]
for stat in statistics: 
    for feature in blinks_head_old:
        new_feature = feature + '_' + stat
        blinks_head.append(new_feature)

blinks = []
blinks_old  = old_features[2:10]
for stat in statistics:
    for feature in blinks_old:
        new_feature = feature + '_' + stat
        blinks.append(new_feature)

head = []
head_old = old_features[10:]
for stat in statistics:
    for feature in head_old:
        new_feature = feature + '_' + stat
        head.append(new_feature)

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
                #th = eeg_series.quantile(.33)
                #th = eeg_series.quantile(.9)
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
    
    selected_features = [
                         'RightBlinkOpeningAmplitude_mean',
                         'HeadRoll_median',
                         'HeadPitch_min',
                         ]

    #selected_features = saccade_fixation
    #selected_features = blinks_head
    #selected_features = blinks
    #selected_features = head
    X_train_df = X_train_df[selected_features]
    
    if VISUALIZE:
        rf = RandomForestClassifier(class_weight=weight_dict, max_depth=3)
    else:
        rf = RandomForestClassifier(class_weight=weight_dict)

    rf.fit(X_train_df, y_train)
    
    import pickle
    # save the model to disk
    if VISUALIZE:
        filename = 'rf_binary.sav'
        pickle.dump(rf, open(filename, 'wb')) 

    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    
    ############################## Predict ####################################

    X_test_featurized = featurize_data(X_test)
    
    X_test_df = pd.DataFrame(X_test_featurized, columns = features)
    
    
    X_test_df = X_test_df[selected_features]
    y_pred = rf.predict(X_test_df)
    
    for i in range(0, len(y_pred)):
        if y_pred[i] == 1:
            if y_pred[i] == y_test[i]:
                print("1")
                print(i)
                break # 0
    class_low_sample = X_test_df.iloc[i:i+1]
    #print(class_low_sample)
    class_low_sample.to_csv("class_low_sample.csv", sep = ' ', header=True, index=False)
    for i in range(0, len(y_pred)):
        if y_pred[i] == 2:
            if y_pred[i] == y_test[i]:
                print("2") #56
                print(i)
                break
    class_high_sample = X_test_df.iloc[i:i+1]
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
