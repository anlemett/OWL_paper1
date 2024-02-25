import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

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



full_filename = os.path.join(ML_DIR, "ML_ET_EEG_60__ET.csv")
print("reading data")

# Load the 2D array from the CSV file
TS_np = np.loadtxt(full_filename, delimiter=" ")
    
# Reshape the 2D array back to its original 3D shape
# (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
TS_np = TS_np.reshape((1731, 15000, 17)) #(1731, 15000, 17)


all_np = featurize_data(TS_np)

full_filename = os.path.join(ML_DIR, "ML_ET_EEG_60__EEG.csv")

scores_np = np.loadtxt(full_filename, delimiter=" ")

scores = list(scores_np)
eeg_series = pd.Series(scores)

th = eeg_series.quantile(.93)
scores = [1 if score < th else 2 for score in scores]

df = pd.DataFrame(all_np, columns = features)
df['score'] = scores

df = df [['RightBlinkClosingSpeed_mean', 
          'RightBlinkClosingSpeed_max',
          'LeftBlinkClosingSpeed_std',
          'score']]

df1 = df[df['score']==1]
df2 = df[df['score']==2]

pd.pandas.set_option('display.max_columns', None)
print(df1.head(1))

data1 = df1["RightBlinkClosingSpeed_mean"].tolist()
data2 = df2["RightBlinkClosingSpeed_mean"].tolist()

data3 = df1["RightBlinkClosingSpeed_max"].tolist()
data4 = df2["RightBlinkClosingSpeed_max"].tolist()

data5 = df1["LeftBlinkClosingSpeed_std"].tolist()
data6 = df2["LeftBlinkClosingSpeed_std"].tolist()


arr1 = np.array(data1)
arr1 = arr1[~np.isnan(arr1)]

arr2 = np.array(data2)
arr2 = arr2[~np.isnan(arr2)]

arr3 = np.array(data3)
arr3 = arr3[~np.isnan(arr3)]

arr4 = np.array(data4)
arr4 = arr4[~np.isnan(arr4)]

arr5 = np.array(data5)
arr5 = arr5[~np.isnan(arr5)]

arr6 = np.array(data6)
arr6 = arr6[~np.isnan(arr6)]

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot((arr1, arr2, arr3, arr4, arr5, arr6))
 
# show plot
plt.show()