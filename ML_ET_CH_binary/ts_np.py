import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import math
import sys

from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking3")
CH_DIR = os.path.join(DATA_DIR, "CH")


# "D6r3_AE", 
FILENAMES_LOW =    ["D1r1_MO", "D1r6_EI",            "D2r4_UO", "D3r1_KB", "D3r5_PF", 
                    "D4r3_AL", "D4r4_IH", "D5r2_RI", "D5r5_JO",            "D6r5_HC",
                    #"D4r3_AL", "D4r4_IH", "D5r2_RI", "D5r5_JO", "D6r3_AE", "D6r5_HC",
                    "D7r3_LS", "D7r4_ML", "D8r1_AP", "D8r6_AK", "D9r2_RE", "D9r6_SV"
                   ]
# "D6r2_AE", 
FILENAMES_HIGH =   ["D1r3_MO", "D1r4_EI", "D2r1_KV", "D2r6_UO", "D3r2_KB", "D3r4_PF",
                    "D4r2_AL", "D4r5_IH", "D5r3_RI", "D5r4_JO",            "D6r6_HC",
                    #"D4r2_AL", "D4r5_IH", "D5r3_RI", "D5r4_JO", "D6r2_AE", "D6r6_HC",
                    "D7r2_LS", "D7r5_ML", "D8r2_AP", "D8r5_AK", "D9r1_RE", "D9r4_SV"
                   ]
# "D6r1_AE", 
FILENAMES_MEDIUM = ["D1r2_MO", "D1r5_EI", "D2r2_KV", "D2r5_UO", "D3r3_KB", "D3r6_PF",
                    "D4r1_AL", "D4r6_IH", "D5r1_RI", "D5r6_JO",            "D6r4_HC",
                    #"D4r1_AL", "D4r6_IH", "D5r1_RI", "D5r6_JO", "D6r1_AE", "D6r4_HC",
                    "D7r1_LS", "D7r6_ML", "D8r3_AP", "D8r4_AK", "D9r3_RE", "D9r5_SV"
                   ]

#FILENAMES_LOW =    ["D1r1_MO"]
#FILENAMES_HIGH =   ["D1r3_MO"]
#FILENAMES_MEDIUM = ["D1r2_MO"]


def getTimeInterval(timestamp, ch_first_timestamp, time_interval_duration):

    if timestamp < ch_first_timestamp:
        return 0
    return math.trunc((timestamp - ch_first_timestamp)/time_interval_duration) + 1


def create_TS_np(df, features, scores_df):

    columns = ['UnixTimestamp'] + ['SamplePerSecond'] + features
    df = df[columns]
          
    #####################################
    #scale the values
    scaler = preprocessing.MinMaxScaler()

    for feature in features:
        feature_lst = df[feature].tolist()
        scaled_feature_lst = scaler.fit_transform(np.asarray(feature_lst).reshape(-1, 1))
        df = df.drop(feature, axis = 1)
        df[feature] = scaled_feature_lst
    #####################################
    number_of_features = len(features)
    
    
    # ch_first_timestamp - first timestmap from CH file
    # CH score time interval: [score timestamp - 180 s, score timestamp]
    
    ch_first_timestamp = scores_df['timestamp'].loc[0]
    
    df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                              ch_first_timestamp,
                                                              180 # 3min
                                                              ),
                                  axis=1) 

    new_columns = ['timeInterval'] + columns
    df = df[new_columns]
  
    scores = scores_df['score'].tolist()
    print(scores)
    del scores[0]
    print(scores)
    scores = [2 if score>2 else 0 if score == 0 else 1 for score in scores]
    print(scores)
    
    number_of_time_intervals = len(scores)    
    
    #####################################
    window_size = 250 * 180 #sample per second * number of seconds
    
    timeseries_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
       
    dim1_idx = 0
    for ti in range (1, number_of_time_intervals + 1):
        ti_df = df[df['timeInterval']==ti]
        
        if scores[ti-1]==0: #ATCO missed sound signal
            continue
        
        dim2_idx = 0
        print(len(ti_df.index))
        for index, row in ti_df.iterrows():
            #exclude timeInterval, UnixTimestamp, SamplePerSecond
            lst_of_features = row.values.tolist()[3:]
            #print(lst_of_features)
            #print(row_num)
            timeseries_np[dim1_idx, dim2_idx] = lst_of_features
            dim2_idx = dim2_idx + 1
        dim1_idx = dim1_idx + 1

    if 0 in scores:
        scores.remove(0)
    
    return (timeseries_np, scores)


def get_TS_np(features, time_interval_duration):
    
    window_size = 250 * time_interval_duration
    number_of_features = len(features)
    
    # TS_np shape (a,b,c):
    # a - number of time periods, b - number of measures per run (WINDOW_SIZE), c - number of features
    
    # we squeeze to 0 the dimension which we do not know and
    # to which we want to append
    TS_np = np.zeros(shape=(0, window_size, number_of_features))

    all_scores = []

    for filename in FILENAMES_LOW:
        print(filename)
        full_filename = os.path.join(ET_DIR, "ET_" + filename + ".csv")
        df_low = pd.read_csv(full_filename, sep=' ', low_memory=False)
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df_low = pd.read_csv(full_filename, sep=' ', low_memory=False)

        (temp_TS_np, temp_scores_lst) = create_TS_np(df_low, features, scores_df_low)

        TS_np = np.append(TS_np, temp_TS_np, axis=0)
        all_scores.extend(temp_scores_lst)
    
    for filename in FILENAMES_HIGH:
        print(filename)
        full_filename = os.path.join(ET_DIR, "ET_" + filename + ".csv")
        df_high = pd.read_csv(full_filename, sep=' ', low_memory=False)
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df_high = pd.read_csv(full_filename, sep=' ', low_memory=False)

        (temp_TS_np, temp_scores_lst) = create_TS_np(df_high, features, scores_df_high)

        TS_np = np.append(TS_np, temp_TS_np, axis=0)
        all_scores.extend(temp_scores_lst)

    
    for filename in FILENAMES_MEDIUM:
        print(filename)
        full_filename = os.path.join(ET_DIR, "ET_" + filename + ".csv")
        df_medium = pd.read_csv(full_filename, sep=' ', low_memory=False)
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df_medium = pd.read_csv(full_filename, sep=' ', low_memory=False)

        (temp_TS_np, temp_scores_lst) = create_TS_np(df_medium, features, scores_df_medium)

        TS_np = np.append(TS_np, temp_TS_np, axis=0)
        all_scores.extend(temp_scores_lst)
    
    return (TS_np, all_scores)

