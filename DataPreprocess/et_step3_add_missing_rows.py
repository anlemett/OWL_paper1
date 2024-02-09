import warnings
warnings.filterwarnings('ignore')

import os
#import sys

import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
INPUT_DIR = os.path.join(DATA_DIR, "EyeTracking2")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking3")

metrics_list = ['Saccade', 'Fixation',
                'LeftPupilDiameter', 'RightPupilDiameter',
                'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                'HeadHeading', 'HeadPitch',	'HeadRoll']

column_names = ['UnixTimestamp'] + ['SamplePerSecond'] + metrics_list

filenames = ["D1r1_MO", "D1r2_MO", "D1r3_MO",
             "D1r4_EI", "D1r5_EI", "D1r6_EI",
             "D2r1_KV", "D2r3_KV",
             "D2r4_UO", "D2r5_UO", "D2r6_UO",
             "D3r1_KB", "D3r2_KB", "D3r3_KB",
             "D3r4_PF", "D3r5_PF", "D3r6_PF",
             "D4r1_AL", "D4r2_AL", "D4r3_AL",
             "D4r4_IH", "D4r5_IH", "D4r6_IH",
             "D5r1_RI", "D5r2_RI", "D5r3_RI",
             "D5r4_JO", "D5r5_JO", "D5r6_JO",
             "D6r1_AE", "D6r2_AE", "D6r3_AE",
             "D6r4_HC", "D6r5_HC", "D6r6_HC",
             "D7r1_LS", "D7r2_LS", "D7r3_LS",
             "D7r4_ML", "D7r5_ML", "D7r6_ML",
             "D8r1_AP", "D8r2_AP", "D8r3_AP",
             "D8r4_AK", "D8r5_AK", "D8r6_AK",
             "D9r1_RE", "D9r2_RE", "D9r3_RE",
             "D9r4_SV", "D9r5_SV", "D9r6_SV"
             ]

# for testing
#filenames = ["D2r3_KV"]

for filename in filenames:
    full_filename = os.path.join(INPUT_DIR, "ET_" + filename +  ".csv")
    df = pd.read_csv(full_filename, sep=' ')
        
    df = df[df['SamplePerSecond']<=250]
    
    first_timestamp = df['UnixTimestamp'].loc[0]
    last_timestamp = df['UnixTimestamp'].loc[len(df.index)-1]
    
    
    new_df = df.copy()
    
    for ts in range(first_timestamp, last_timestamp + 1):
        
        ts_df = df[df['UnixTimestamp']==ts]
        
        if ts_df.empty:
            print(filename + ": empty second")

            # add 250 rows            
            timestamp_lst = [ts]*250
            sample_per_second_lst = range(1,251)
            metric_values_lst = [None]*250
            
            df_to_add = pd.DataFrame()
            
            df_to_add['UnixTimestamp'] = timestamp_lst
            df_to_add['SamplePerSecond'] = sample_per_second_lst
            for metric in metrics_list:
                df_to_add[metric] = metric_values_lst
            
            new_df = pd.concat([new_df, df_to_add])
            continue
            
        number_of_samples = len(ts_df.index)
        
        if number_of_samples < 250:
            print(filename + ": adding rows") 
            
            num_to_add = 250 - number_of_samples
            # add num_to_add rows
            timestamp_lst = [ts]*num_to_add
            sample_per_second_lst = range(number_of_samples + 1,251)
            metric_values_lst = [None]*num_to_add
            
            df_to_add = pd.DataFrame()
            
            df_to_add['UnixTimestamp'] = timestamp_lst
            df_to_add['SamplePerSecond'] = sample_per_second_lst
            for metric in metrics_list:
                df_to_add[metric] = metric_values_lst
            
            new_df = pd.concat([new_df, df_to_add])
            

    #fill the null rows with the median of respective columns
    new_df = new_df.fillna(new_df.median())
    
    new_df.sort_values(['UnixTimestamp', 'SamplePerSecond'], ascending=[True, True], inplace=True)
    
    number_of_timestamps = last_timestamp - first_timestamp + 1
    number_of_rows1 = number_of_timestamps*250
    number_of_rows2 = len(new_df.index)
    
    print(number_of_timestamps)
    print(number_of_rows1)
    print(number_of_rows2)
    
    full_filename = os.path.join(OUTPUT_DIR, "ET_" + filename +  ".csv")
    new_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)