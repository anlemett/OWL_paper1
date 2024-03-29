import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from statistics import mean, median, stdev

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EEG")
stats_filename = "EEG_stats.xlsx"

metrics_list = ['workload', 'stress', 'vigilance']

# Create sheets and save headers
with pd.ExcelWriter(path = stats_filename, engine = 'openpyxl') as writer:
    
    for metric in metrics_list:
        df = pd.DataFrame(columns=['atco_id', 'scenario',
                               'average', 'median', 'sd', 'min', 'max'])
        df.to_excel(writer, index = False, sheet_name = metric)

###############################################################################
def metric_stat(metric, participant_id, df_low, df_medium, df_high):
    low_lst = list(df_low[metric].dropna()) if not df_low.empty else []
    low_av = mean(low_lst) if low_lst else 0
    low_median = median(low_lst) if low_lst else 0
    low_sd = stdev(low_lst) if low_lst else 0
    low_min = min(low_lst) if low_lst else 0
    low_max = max(low_lst) if low_lst else 0
    
    high_lst = list(df_high[metric].dropna()) if not df_high.empty else []
    high_av = mean(high_lst) if high_lst else 0
    high_median = median(high_lst) if high_lst else 0
    high_sd = stdev(high_lst) if high_lst else 0
    high_min = min(high_lst) if high_lst else 0
    high_max = max(high_lst) if high_lst else 0
    
    medium_lst = list(df_medium[metric].dropna()) if not df_medium.empty else []
    medium_av = mean(medium_lst) if medium_lst else 0
    medium_median = median(medium_lst) if medium_lst else 0
    medium_sd = stdev(medium_lst) if medium_lst else 0
    medium_min = min(medium_lst) if medium_lst else 0
    medium_max = max(medium_lst) if medium_lst else 0
    
    data = {'atco_id': [participant_id, participant_id, participant_id],
        'scenario': ['low', 'medium', 'high'],
        'average': [low_av, medium_av, high_av],
        'median': [low_median, medium_median, high_median],
        'sd': [low_sd, medium_sd, high_sd],
        'min': [low_min, medium_min, high_min],
        'max': [low_max, medium_max, high_max]
       }
    
    return data

###############################################################################
def calculate_stat(participant_id, df_low, df_medium, df_high):
    
    for metric in metrics_list:
        
        data =  metric_stat(metric, participant_id, df_low, df_medium, df_high)
        
        new_rows_df = pd.DataFrame(data)
    
        sheetname = metric
        workbook = openpyxl.load_workbook(stats_filename)  # load workbook if already exists
        sheet = workbook[sheetname]  # declare the active sheet 

        # append the dataframe results to the current excel file
        for row in dataframe_to_rows(new_rows_df, header = False, index = False):
            sheet.append(row)
        workbook.save(stats_filename)  # save workbook
        workbook.close()  # close workbook

###############################################################################
filename = os.path.join(DATA_DIR, "D1r1_MO.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D1r3_MO.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D1r2_MO.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('MO', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D1r6_EI.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D1r4_EI.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D1r5_EI.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('EI', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D2r3_KV.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D2r1_KV.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D2r2_KV.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('KV', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D2r4_UO.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D2r6_UO.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D2r5_UO.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('UO', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D3r1_KB.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D3r3_KB.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D3r2_KB.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('KB', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D3r5_PF.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D3r4_PF.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D3r6_PF.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('PF', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D4r3_AL.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D4r2_AL.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D4r1_AL.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('AL', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D4r4_IH.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D4r5_IH.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D4r6_IH.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('IH', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D5r2_RI.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D5r3_RI.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D5r1_RI.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('RI', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D5r5_JO.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D5r4_JO.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D5r6_JO.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('JO', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D6r3_AE.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D6r2_AE.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D6r1_AE.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('AE', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D6r5_HC.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D6r6_HC.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D6r4_HC.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('HC', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D7r3_LS.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D7r2_LS.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D7r1_LS.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('LS', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D7r4_ML.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D7r5_ML.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D7r6_ML.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('ML', df_low, df_medium, df_high)

###############################################################################
df_low = pd.DataFrame()
df_high = pd.DataFrame()
df_medium = pd.DataFrame()

calculate_stat('AP', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D8r6_AK.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D8r5_AK.csv")
df_high = pd.read_csv(filename, sep=';')

df_medium = pd.DataFrame()

calculate_stat('AK', df_low, df_medium, df_high)

###############################################################################
filename = os.path.join(DATA_DIR, "D9r2_RE.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D9r1_RE.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D9r3_RE.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('RE', df_low, df_medium, df_high)

###############################################################################    
filename = os.path.join(DATA_DIR, "D9r6_SV.csv")
df_low = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D9r4_SV.csv")
df_high = pd.read_csv(filename, sep=';')

filename = os.path.join(DATA_DIR, "D9r5_SV.csv")
df_medium = pd.read_csv(filename, sep=';')

calculate_stat('SV', df_low, df_medium, df_high)
