import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from statistics import median
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

def transform_to_median_np(x_data):
    """
    :param x_data: numpy array of shape
    (number_of_timeintervals, number_of_timestamps, number_of_features)
    where number_of_timestamps == TIME_INTERVAL_DURATION*250

    :return: numpy array of shape
    (number_of_timeintervals, number_features)
    """
    print("Input shape before transformation:", x_data.shape)

    median_np = np.median(x_data, axis=-2)

    print("Shape after transformation:", median_np.shape)
    return median_np

if not os.path.exists(ML_DIR):
    print("no folder")
full_filename = os.path.join(ML_DIR, "ML_ET_EEG_60__ET.csv")
if not os.path.exists(full_filename):
    print("no file")


print("reading data")

# Load the 2D array from the CSV file
et_np = np.loadtxt(full_filename, delimiter=" ")
    
# Reshape the 2D array back to its original 3D shape
# (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
# 60 -> (1731, 15000, 17)
et_np = et_np.reshape((1731, 15000, 17))

median_et_np = transform_to_median_np(et_np)

full_filename = os.path.join(ML_DIR, "ML_ET_EEG_60__EEG.csv")

eeg_np = np.loadtxt(full_filename, delimiter=" ")

features = [
            'SaccadesNumber', 'SaccadesDuration',
            'FixationNumber', 'FixationDuration',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch', 'HeadRoll'
            ]

et_df = pd.DataFrame(data=median_et_np, columns=features)

eeg_df = pd.DataFrame(data=eeg_np, columns=['EEGMean'])

#et_df = et_df[['SaccadesNumber']]

df = pd.concat([et_df, eeg_df], axis=1)

eeg_series = pd.Series(df['EEGMean'].tolist())
#Split into 5 bins by percentile
#thresholds = eeg_series.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])
thresholds = eeg_series.quantile([.1, .2, .3, .4])
thresholds_lst = thresholds.to_list()


def getBin(value):
    for i in range(0, 4):
      if value < thresholds_lst[i]:
          return i+1
    return 5

metric_medians = []
df['Bin'] = df.apply(lambda row: getBin(row['EEGMean']), axis=1)

dict = {}
for feature in features:
    dict[feature] = []

for i in range(1,6):
    bin_df = df[df['Bin']==i]
    
    for feature in features:
        dict[feature].append(median(bin_df[feature]))
    
for feature in features:
    print(dict[feature])
    



