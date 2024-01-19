import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EyeTracking")
anova_filename = "ET_anova_input.csv"
#anova_filename = "ET_anova_input_temp.csv"

metrics_list = ['Saccade', 'PupilDiameter', 'LeftPupilDiameter', 'RightPupilDiameter',
                    'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
                    'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
                    'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
                    'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
                    'HeadHeading', 'HeadPitch',	'HeadRoll']

#metrics_list = ['Saccade', 'PupilDiameter']

columns_list = ['atco_id','scenario'] + metrics_list
anova_df = pd.DataFrame(columns=columns_list)


def create_anova_input(df, atco_id, scenario):
    metric_lst = df['Saccade'].to_list() #any metric to get the number of values
    num = len(metric_lst)
 
    atco_id_lst = [atco_id] * num
    scenario_lst = [scenario] * num

    data = {'atco_id': atco_id_lst,
            'scenario': scenario_lst
           }
    
    for metric in metrics_list:
        metric_lst = df[metric].to_list()
        data[metric] = metric_lst        
 
    new_df = pd.DataFrame(data)

    return pd.concat([anova_df, new_df])

###############################################################################    
filename = os.path.join(DATA_DIR, "D1r1_MO.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'MO', 'low')

filename = os.path.join(DATA_DIR, "D1r3_MO.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'MO', 'high')

filename = os.path.join(DATA_DIR, "D1r2_MO.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'MO', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D1r6_EI.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'EI', 'low')

filename = os.path.join(DATA_DIR, "D1r4_EI.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'EI', 'high')

filename = os.path.join(DATA_DIR, "D1r5_EI.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'EI', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D2r3_KV.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'KV', 'low')

filename = os.path.join(DATA_DIR, "D2r1_KV.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'KV', 'high')

###############################################################################
filename = os.path.join(DATA_DIR, "D2r4_UO.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'UO', 'low')

filename = os.path.join(DATA_DIR, "D2r6_UO.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'UO', 'high')

filename = os.path.join(DATA_DIR, "D2r5_UO.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'UO', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D3r1_KB.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'KB', 'low')

filename = os.path.join(DATA_DIR, "D3r3_KB.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'KB', 'high')

filename = os.path.join(DATA_DIR, "D3r2_KB.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'KB', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D3r5_PF.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'PF', 'low')

filename = os.path.join(DATA_DIR, "D3r4_PF.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'PF', 'high')

filename = os.path.join(DATA_DIR, "D3r6_PF.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'PF', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D4r3_AL.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'AL', 'low')

filename = os.path.join(DATA_DIR, "D4r2_AL.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'AL', 'high')

filename = os.path.join(DATA_DIR, "D4r1_AL.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'AL', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D4r4_IH.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'IH', 'low')

filename = os.path.join(DATA_DIR, "D4r5_IH.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'IH', 'high')

filename = os.path.join(DATA_DIR, "D4r6_IH.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'IH', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D5r2_RI.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'RI', 'low')

filename = os.path.join(DATA_DIR, "D5r3_RI.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'RI', 'high')

filename = os.path.join(DATA_DIR, "D5r1_RI.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'RI', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D5r6_JO.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'JO', 'low')

filename = os.path.join(DATA_DIR, "D5r4_JO.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'JO', 'high')

filename = os.path.join(DATA_DIR, "D5r6_JO.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'JO', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D5r6_JO.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'JO', 'low')

filename = os.path.join(DATA_DIR, "D5r4_JO.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'JO', 'high')

filename = os.path.join(DATA_DIR, "D5r6_JO.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'JO', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D6r3_AE.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'AE', 'low')

filename = os.path.join(DATA_DIR, "D6r2_AE.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'AE', 'high')

filename = os.path.join(DATA_DIR, "D6r1_AE.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'AE', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D6r5_HC.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'HC', 'low')

filename = os.path.join(DATA_DIR, "D6r6_HC.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'HC', 'high')

filename = os.path.join(DATA_DIR, "D6r4_HC.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'HC', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D7r3_LS.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'LS', 'low')

filename = os.path.join(DATA_DIR, "D7r2_LS.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'LS', 'high')

filename = os.path.join(DATA_DIR, "D7r1_LS.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'LS', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D7r4_ML.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'ML', 'low')

filename = os.path.join(DATA_DIR, "D7r5_ML.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'ML', 'high')

filename = os.path.join(DATA_DIR, "D7r6_ML.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'ML', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D8r1_AP.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'AP', 'low')

filename = os.path.join(DATA_DIR, "D8r2_AP.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'AP', 'high')

filename = os.path.join(DATA_DIR, "D8r3_AP.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'AP', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D8r6_AK.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'AK', 'low')

filename = os.path.join(DATA_DIR, "D8r5_AK.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'AK', 'high')

filename = os.path.join(DATA_DIR, "D8r4_AK.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'AK', 'medium')

###############################################################################
filename = os.path.join(DATA_DIR, "D9r2_RE.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'RE', 'low')

filename = os.path.join(DATA_DIR, "D9r1_RE.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'RE', 'high')

filename = os.path.join(DATA_DIR, "D9r3_RE.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'RE', 'medium')

###############################################################################    
filename = os.path.join(DATA_DIR, "D9r6_SV.log")
df_low = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_low, 'SV', 'low')

filename = os.path.join(DATA_DIR, "D9r4_SV.log")
df_high = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_high, 'SV', 'high')

filename = os.path.join(DATA_DIR, "D9r5_SV.log")
df_medium = pd.read_csv(filename, sep='\t', low_memory=False)
anova_df = create_anova_input(df_medium, 'SV', 'medium')

anova_df.to_csv(anova_filename, sep = ' ', index = False, header = True)
