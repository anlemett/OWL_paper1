import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np

import sys

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

ATCOs = ['MO', 'EI', 'KV', 'UO', 'KB', 'PF', 'AL', 'IH', 'RI',
         'JO', 'AE', 'HC', 'LS', 'ML', 'AP', 'AK', 'RE', 'SV']


SPLIT_BY_QUANTILE = False

def get_TS_np(features, time_interval_duration):
    
    window_size = 250 * time_interval_duration
    number_of_features = len(features)
    
    # TS_np shape (a,b,c):
    # a - number of time intervals, b - number of measures per time interval (WINDOW_SIZE),
    # c - number of features
    
    # we squeeze to 0 the dimension which we do not know and
    # to which we want to append
    TS_np = np.zeros(shape=(0, window_size, number_of_features))
    all_scores = []
    
    #**************************************
    print("Reading Eye Tracking data")
    full_filename = os.path.join(ML_DIR, "ML_ET_" + str(time_interval_duration) + ".csv")
    et_df = pd.read_csv(full_filename, sep=' ', low_memory=False)

    print("Reading EEG data")
    full_filename = os.path.join(ML_DIR, "ML_EEG_" + str(time_interval_duration) + ".csv")
    eeg_df = pd.read_csv(full_filename, sep=' ', low_memory=False)
     
       
    dim1_idx = 0

    for atco in ATCOs:
        print(atco)
        et_atco_df = et_df[et_df['ATCO']==atco]
        eeg_atco_df = eeg_df[eeg_df['ATCO']==atco]
        
        if et_atco_df.empty or eeg_atco_df.empty:
            continue
        
        for run in range(1,4):
            et_run_df = et_atco_df[et_atco_df['Run']==run]
            eeg_run_df = eeg_atco_df[eeg_atco_df['Run']==run]
            
            if et_run_df.empty or eeg_run_df.empty:
                continue
        
            number_of_time_intervals = len(eeg_run_df['timeInterval'].tolist())
        
            run_TS_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
            run_scores = []
            
            print(number_of_time_intervals)
            dim1_idx = 0
            for ti in range(1, number_of_time_intervals+1):
                et_ti_df = et_run_df[et_run_df['timeInterval']==ti]
                eeg_ti_df = eeg_run_df[eeg_run_df['timeInterval']==ti]
                                
                ti_score_lst = eeg_ti_df['WorkloadMean'].tolist()
                
                if et_ti_df.empty or not ti_score_lst:
                    continue
                
                ti_score = ti_score_lst[0]
                
                dim2_idx = 0
                for index, row in et_ti_df.iterrows():
                    #exclude ATCO, Run, timeInterval, UnixTimestamp, SamplePerSecond
                    lst_of_features = row.values.tolist()[5:]
                    run_TS_np[dim1_idx, dim2_idx] = lst_of_features
                    dim2_idx = dim2_idx + 1
                    
                if SPLIT_BY_QUANTILE:
                    run_scores.append(ti_score)
                else:
                    score = 1 if ti_score < 0.33 else 3 if ti_score > 0.66 else 2
                    run_scores.append(score)
                        
                dim1_idx = dim1_idx + 1
                
            if dim1_idx < number_of_time_intervals:
                run_TS_np = run_TS_np[:dim1_idx]
                
            TS_np = np.append(TS_np, run_TS_np, axis=0)
            all_scores.extend(run_scores)

    if SPLIT_BY_QUANTILE:
        #Split into 3 bins with ap. equal amount of values
        eeg_series = pd.Series(all_scores)
        (th1, th2) = eeg_series.quantile([.33, .66])
        all_scores = [1 if score < th1 else 3 if score > th2 else 2 for score in all_scores]

    return (TS_np, all_scores)

