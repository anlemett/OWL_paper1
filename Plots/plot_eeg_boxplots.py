import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
EEG_DIR = os.path.join(DATA_DIR, "EEG4")
CH_DIR = os.path.join(DATA_DIR, "CH2")

print("Reading CH data")
full_filename = os.path.join(CH_DIR, "CH_all.csv")
ch_df = pd.read_csv(full_filename, sep=' ')
ch_df.set_index(['ATCO', 'Run', 'timeInterval'], inplace=True)

print("Reading EEG data")
full_filename = os.path.join(EEG_DIR, "EEG_all_180.csv")
eeg_df = pd.read_csv(full_filename, sep=' ')
eeg_df.set_index(['ATCO', 'Run', 'timeInterval'], inplace=True)

df = pd.concat([ch_df, eeg_df],  join="inner", axis=1)

df1 = df[df['score']<=3]
df2 = df[df['score']>3]

data1 = df1["WorkloadMean"].tolist()
data2 = df2["WorkloadMean"].tolist()

arr1 = np.array(data1)
arr1 = arr1[~np.isnan(arr1)]

arr2 = np.array(data2)
arr2 = arr2[~np.isnan(arr2)]

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot((arr1, arr2))
 
# show plot
plt.show()