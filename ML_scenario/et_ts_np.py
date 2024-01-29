import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
#import math

from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EyeTracking")
'''
FILENAMES_LOW =    ["D1r1_MO", "D1r6_EI", "D2r3_KV", "D2r4_UO", "D3r1_KB", "D3r5_PF", 
                    "D4r3_AL", "D4r4_IH", "D5r2_RI", "D5r5_JO", "D6r3_AE", "D6r5_HC",
                    "D7r3_LS", "D7r4_ML", "D8r1_AP", "D8r6_AK", "D9r2_RE", "D9r6_SV"
                   ]

FILENAMES_HIGH =   ["D1r3_MO", "D1r4_EI", "D2r1_KV", "D2r6_UO", "D3r2_KB", "D3r4_PF",
                    "D4r2_AL", "D4r5_IH", "D5r3_RI", "D5r4_JO", "D6r2_AE", "D6r6_HC",
                    "D7r2_LS", "D7r5_ML", "D8r2_AP", "D8r5_AK", "D9r1_RE", "D9r4_SV"
                   ]

FILENAMES_MEDIUM = ["D1r2_MO", "D1r5_EI",            "D2r5_UO", "D3r3_KB", "D3r6_PF",
                    "D4r1_AL", "D4r6_IH", "D5r1_RI", "D5r6_JO", "D6r1_AE", "D6r4_HC",
                    "D7r1_LS", "D7r6_ML", "D8r3_AP", "D8r4_AK", "D9r3_RE", "D9r5_SV"
                   ]
'''
FILENAMES_LOW =    ["D1r1_MO", "D2r4_UO"]
FILENAMES_HIGH =   ["D1r3_MO", "D2r6_UO"]
FILENAMES_MEDIUM = ["D1r2_MO", "D2r5_UO"]


def create_TS_np(df, features):

    df = df[features]
      
    #fill the null rows with the mean of respective columns
    df = df.fillna(df.mean())
    
    #####################################
    window_size = 500000    #should be the amount of rows in the smallest data file
    df = df[df.index<window_size]
    
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
    
    timeseries_np = np.zeros(shape=(window_size, number_of_features))
    
    for index, row in df.iterrows():
        lst_of_features = row.values.tolist()
        timeseries_np[index] = lst_of_features

    return timeseries_np


def get_TS_np(features):
    
    number_of_runs = len(FILENAMES_LOW) + len(FILENAMES_HIGH) + len(FILENAMES_MEDIUM)
    window_size = 500000
    number_of_features = len(features)
    
    # TS_np shape (a,b,c):
    # a - number of runs, b - number of measures per run (WINDOW_SIZE), c - number of features
    TS_np = np.zeros(shape=(number_of_runs, window_size, number_of_features))
    all_scores = []

    row_idx = 0
    for filename in FILENAMES_LOW:
        full_filename = os.path.join(DATA_DIR, "ET_" + filename + ".csv")
        df_low = pd.read_csv(full_filename, sep=' ', low_memory=False)
        temp_np = create_TS_np(df_low, features)
        TS_np[row_idx] = temp_np
        row_idx = row_idx + 1
        
        all_scores.append(1)
    
    for filename in FILENAMES_HIGH:
        full_filename = os.path.join(DATA_DIR, "ET_" + filename + ".csv")
        df_high = pd.read_csv(full_filename, sep=' ', low_memory=False)
        temp_np = create_TS_np(df_high, features)
        TS_np[row_idx] = temp_np
        row_idx = row_idx + 1
        
        all_scores.append(3)
    
    for filename in FILENAMES_MEDIUM:
        full_filename = os.path.join(DATA_DIR, "ET_" + filename + ".csv")
        df_medium = pd.read_csv(full_filename, sep=' ', low_memory=False) 
        temp_np = create_TS_np(df_medium, features)

        TS_np[row_idx] = temp_np
        row_idx = row_idx + 1

        all_scores.append(2)
    
    return (TS_np, all_scores)

