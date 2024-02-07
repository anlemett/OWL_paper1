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

def get_TS_np(features):
    
    window_size = 250 * 180
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
    full_filename = os.path.join(ML_DIR, "ML_ET_180.csv")
    et_df = pd.read_csv(full_filename, sep=' ', low_memory=False)

    print("Reading CH data")
    full_filename = os.path.join(ML_DIR, "ML_CH.csv")
    ch_df = pd.read_csv(full_filename, sep=' ', low_memory=False)
     
       
    dim1_idx = 0

    for atco in ATCOs:
        print(atco)
        et_atco_df = et_df[et_df['ATCO']==atco]
        ch_atco_df = ch_df[ch_df['ATCO']==atco]
                
        if et_atco_df.empty:
            continue
        
        for run in range(1,4):
            et_run_df = et_atco_df[et_atco_df['Run']==run]
            ch_run_df = ch_atco_df[ch_atco_df['Run']==run]
            
            if et_run_df.empty:
                continue
        
            number_of_time_intervals = len(ch_run_df.index)
        
            run_TS_np = np.zeros(shape=(number_of_time_intervals, window_size, number_of_features))
            print(run_TS_np.shape)
            all_run_scores = ch_run_df['score'].tolist() 
            run_scores = []
                        
            print(number_of_time_intervals)
            dim1_idx = 0
            for ti in range(1, number_of_time_intervals+1):
                et_ti_df = et_run_df[et_run_df['timeInterval']==ti]
                                               
                if et_ti_df.empty or all_run_scores[ti-1]==0:
                    continue
                       
                dim2_idx = 0
                for index, row in et_ti_df.iterrows():
                    #exclude ATCO, Run, timeInterval, UnixTimestamp, SamplePerSecond
                    lst_of_features = row.values.tolist()[5:]
                    run_TS_np[dim1_idx, dim2_idx] = lst_of_features
                    dim2_idx = dim2_idx + 1
                 
                dim1_idx = dim1_idx + 1
                run_score = all_run_scores[ti-1]
                #score = all_run_scores[ti-1]
                #run_score = 1 if score<2 else 3 if score>2 else 2
                run_scores.extend([run_score])
                
            if dim1_idx < number_of_time_intervals:
                run_TS_np = run_TS_np[:dim1_idx]
                
            TS_np = np.append(TS_np, run_TS_np, axis=0)
            all_scores.extend(run_scores)

    return (TS_np, all_scores)

