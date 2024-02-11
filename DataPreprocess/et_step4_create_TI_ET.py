import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import math

from sklearn import preprocessing

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ET_DIR = os.path.join(DATA_DIR, "EyeTracking3")
CH_DIR = os.path.join(DATA_DIR, "CH1")
OUTPUT_DIR = os.path.join(DATA_DIR, "EyeTracking4")

TIME_INTERVAL_DURATION = 180  #sec

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV"           ],
             ["D2r4_UO", "D2r5_UO", "D2r6_UO"],
             ["D3r1_KB", "D3r2_KB", "D3r3_KB"],
             ["D3r4_PF", "D3r5_PF", "D3r6_PF"],
             ["D4r1_AL", "D4r2_AL", "D4r3_AL"],
             ["D4r4_IH", "D4r5_IH", "D4r6_IH"],
             ["D5r1_RI", "D5r2_RI", "D5r3_RI"],
             ["D5r4_JO", "D5r5_JO", "D5r6_JO"],
             ["D6r1_AE", "D6r2_AE", "D6r3_AE"],
             ["D6r4_HC", "D6r5_HC", "D6r6_HC"],
             ["D7r1_LS", "D7r2_LS", "D7r3_LS"],
             ["D7r4_ML", "D7r5_ML", "D7r6_ML"],
             ["D8r1_AP", "D8r2_AP", "D8r3_AP"],
             ["D8r4_AK", "D8r5_AK", "D8r6_AK"],
             ["D9r1_RE", "D9r2_RE", "D9r3_RE"],
             ["D9r4_SV", "D9r5_SV", "D9r6_SV"]
             ]

#filenames = [["D3r1_KB", "D3r2_KB", "D3r3_KB"]]

features = ['Saccade', 'Fixation',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch',	'HeadRoll']

def getTimeInterval(timestamp, ch_first_timestamp, ch_last_timestamp):

    if timestamp < ch_first_timestamp:
        return 0
    if timestamp > ch_last_timestamp:
        return 0
    return math.trunc((timestamp - ch_first_timestamp)/TIME_INTERVAL_DURATION) + 1


TI_df = pd.DataFrame()

for atco in filenames:
    atco_df = pd.DataFrame()
    run = 1
    for filename in atco:
        print(filename)
        full_filename = os.path.join(ET_DIR, 'ET_' + filename +  ".csv")
        df = pd.read_csv(full_filename, sep=' ')
        
        #nan_count = df.isna().sum()
        #print(nan_count)
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        ch_first_timestamp = scores_df['timestamp'].loc[0]
        ch_last_timestamp = scores_df['timestamp'].tolist()[-1]

        df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                                  ch_first_timestamp,
                                                                  ch_last_timestamp
                                                                  ),
                                      axis=1) 

        df = df[df['timeInterval']!=0]
        
        row_num = len(df.index)
        df['ATCO'] = [filename[-2:]] * row_num
        df['Run'] = [run] * row_num
        run = run + 1    

        columns = ['ATCO'] + ['Run'] + ['timeInterval'] + ['UnixTimestamp'] + ['SamplePerSecond'] + features
        df = df[columns]
        
        atco_df = pd.concat([atco_df, df], ignore_index=True)
        
    #####################################
    #scale the values
    scaler = preprocessing.MinMaxScaler()

    for feature in features:
        feature_lst = atco_df[feature].tolist()
        scaled_feature_lst = scaler.fit_transform(np.asarray(feature_lst).reshape(-1, 1))
        atco_df = atco_df.drop(feature, axis = 1)
        atco_df[feature] = scaled_feature_lst
    #####################################
    
    TI_df = pd.concat([TI_df, atco_df], ignore_index=True)

#print(TI_df.isnull().any().any())
#nan_count = TI_df.isna().sum()
#print(nan_count)

full_filename = os.path.join(OUTPUT_DIR, "ET_all_" + str(TIME_INTERVAL_DURATION) + ".csv")
TI_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)

