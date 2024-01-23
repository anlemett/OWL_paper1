import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np
import math
#from itertools import groupby

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

from sklearn import preprocessing, model_selection

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EyeTracking")

features = ['Saccade', 'Fixation',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch',	'HeadRoll']
# for testing:
#features = ['Fixation', 'LeftPupilDiameter']
#features = ['Fixation']
features = ['LeftPupilDiameter']

columns = ['UnixTimestamp'] + ['ValuesPerSecond'] + features

def getTimeInterval(timestamp, first_timestamp, timeIntervalDuration):

    return math.trunc((timestamp - first_timestamp)/timeIntervalDuration) + 1


def createTimeSeriesDf(df, score):
    
    #fill the null rows with the mean of respective columns
    df = df.fillna(df.mean())

    first_timestamp = df['UnixTimestamp'].loc[0]
    time_interval_duration = 1 #sec
    #timeIntervalDuration = 60 #sec

    df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                              first_timestamp,
                                                              time_interval_duration),
                                  axis=1)

    new_columns = ['timeInterval'] + columns
    df = df[new_columns]

    last_time_interval = list(df['timeInterval'])[-1]


    #####################################
    intervals = []
    window_size = 249 * time_interval_duration

    timeseries_df = pd.DataFrame(columns = [features])

    for ti in range (1, last_time_interval + 1):
        ti_df = df[df['timeInterval']==ti]
        
        if ti_df.empty:
            continue
        if len(ti_df.index) < window_size:
            continue
        
        row_lst = []

        for feature in features:
            feature_lst = ti_df[feature].tolist()
            row_lst.append(feature_lst[0:window_size])
            
        timeseries_df.loc[ti-1] = row_lst
        intervals.append(ti)

    number_of_rows = len(timeseries_df.index)
    scores = [score] * number_of_rows
    
    timeseries_df['timeInterval'] = timeseries_df.index
    timeseries_df = timeseries_df.reset_index(drop=True)
    
    return (timeseries_df, scores)


###############################################################################
full_filename = os.path.join(DATA_DIR, "ET_D5r2_RI.csv")
df_low = pd.read_csv(full_filename, sep=' ', low_memory=False)
(TS_df, all_scores) = createTimeSeriesDf(df_low, 1)

full_filename = os.path.join(DATA_DIR, "ET_D5r3_RI.csv")
df_high = pd.read_csv(full_filename, sep=' ', low_memory=False)
(temp_df, temp_scores) = createTimeSeriesDf(df_low, 3)
TS_df = pd.concat([TS_df, temp_df]).reset_index(drop=True)
all_scores = all_scores + temp_scores

full_filename = os.path.join(DATA_DIR, "ET_D5r1_RI.csv")
df_medium = pd.read_csv(full_filename, sep=' ', low_memory=False) 
 
(temp_df, temp_scores) = createTimeSeriesDf(df_low, 2)
TS_df = pd.concat([TS_df, temp_df]).reset_index(drop=True)
all_scores = all_scores + temp_scores

###############################################################################
scaler = preprocessing.MinMaxScaler()
#scale the values

for index, row in TS_df.iterrows():
    for feature in features:
        current_lst = row[feature]
        TS_df.at[index, feature] = scaler.fit_transform(np.asarray(current_lst).reshape(-1, 1))
        
le = preprocessing.LabelEncoder()  # Generates a look-up table
le.fit(all_scores)
all_scores = le.transform(all_scores)

###############################################################################