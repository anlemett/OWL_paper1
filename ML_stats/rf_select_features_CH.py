#pip install selective
#https://github.com/fidelity/selective

from feature.selector import SelectionMethod, Selective, benchmark, calculate_statistics

import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import pandas as pd
#import sys

from sklearn.ensemble import RandomForestClassifier


DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

BINARY = True

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
        #scores = [1 if score < 2 else 2 for score in scores]
    else:
        scores = [1 if score < 2 else 3 if score > 3 else 2 for score in scores]

    #print(scores)
       
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    weight_dict = weight_classes(scores)
        
    TS_np = np.array(TS_np)
    X = featurize_data(TS_np)
    X_df = pd.DataFrame(X, columns = features)  
    
    y = np.array(scores)
    y = pd.Series(y)


    ########################### Select features ###############################

    selectors = {

        # Non-linear tree-based methods
        "random_forest5": SelectionMethod.TreeBased(num_features=5),

    }
    # Benchmark (sequential)
    score_df, selected_df, runtime_df = benchmark(selectors, X_df, y,
                                                  cv=5)
    #print(score_df, "\n\n", selected_df, "\n\n", runtime_df)
   
    #selected_df = selected_df[selected_df['random_forest5']==1]
    #print(selected_df)

    # Get benchmark statistics by feature
    stats_df = calculate_statistics(score_df, selected_df)
    pd.set_option('display.max_columns', None)
    print(stats_df)
    
    
start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")