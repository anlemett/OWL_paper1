import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
CH_DIR = os.path.join(DATA_DIR, "CH")
OUTPUT_DIR = os.path.join(DATA_DIR, "MLInput")

TIME_INTERVAL_DURATION = 180  #sec

filenames = [["D1r1_MO", "D1r2_MO", "D1r3_MO"],
             ["D1r4_EI", "D1r5_EI", "D1r6_EI"],
             ["D2r1_KV", "D2r2_KV", "D2r2_KV"],
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

#list index - number_of_run-1, value - 1(low), 2(medium), 3(high)
scenario_dict = {
    'MO':[1, 2, 3],
    'EI':[3, 2, 1],
    'KV':[3, 2, 1],
    'UO':[1, 2, 3],
    'KB':[1, 3, 2],
    'PF':[3, 1, 2],
    'AL':[2, 3, 1],
    'IH':[1, 3, 2],
    'RI':[2, 1, 3],
    'JO':[3, 1, 2],
    'AE':[2, 3, 1],
    'HC':[2, 1, 3],
    'LS':[2, 3, 1],
    'ML':[1, 3, 2],
    'AP':[1, 3, 2],
    'AK':[2, 3, 1],
    'RE':[3, 1, 2],
    'SV':[3, 2, 1]
    }


ML_df = pd.DataFrame()

for atco in filenames:

    run = 1
    for filename in atco:
        print(filename)
        
        full_filename = os.path.join(CH_DIR, filename + ".csv")
        scores_df = pd.read_csv(full_filename, sep=' ')
        
        timeintervals = scores_df['timestamp'].tolist()
        number_of_time_intervals = len(timeintervals)-1
        
        score = scenario_dict[filename[-2:]][run-1]
        for ti in range (1, number_of_time_intervals + 1):
            new_row = {'ATCO': filename[-2:], 'Run': run, 'timeInterval': ti,
                       'score': score}

            ML_df = pd.concat([ML_df, pd.DataFrame([new_row])], ignore_index=True)
                
        run = run + 1
        
full_filename = os.path.join(OUTPUT_DIR, "ML_Scenario.csv")
ML_df.to_csv(full_filename, sep=' ', encoding='utf-8', index = False, header = True)
