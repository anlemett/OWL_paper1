import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EyeTracking")

filename = os.path.join(DATA_DIR, "D3r2r3.log")
df = pd.read_csv(filename, sep='\t', low_memory=False)

print(df.head(1))

# Convert LDAP/Win32 FILETIME to Unix Timestamp
def getUnixTimestamp(file_time):
    winSecs       = int(file_time / 10000000); # divide by 10 000 000 to get seconds
    unixTimestamp = (winSecs - 11644473600); # 1.1.1600 -> 1.1.1970 difference in seconds
    return unixTimestamp

df['UnixTimestamp'] = df.apply(lambda row: getUnixTimestamp(row['RealTimeClock']), axis=1)

timestamps = df['UnixTimestamp'].to_list()

print(timestamps[0])
print(timestamps[-1])

# 1701156895 -> Tue Nov 28 2023 08:34:55 GMT+0100 (Central European Standard Time)
# 1701163368 -> Tue Nov 28 2023 10:22:48 GMT+0100 (Central European Standard Time)

# Run 3 starts:
# Tue Nov 28 2023 09:42:00 GMT+0100 (Central European Standard Time) -> 1701160920

r3_start_timestamp = 1701160920

r3_indices = df['UnixTimestamp'].loc[lambda x: x>r3_start_timestamp].index
r3_start_index = r3_indices[0]
print(r3_start_index) # -> 656748

df = df.drop(columns=['UnixTimestamp'])

df1 = df.iloc[:r3_start_index,:]
df2 = df.iloc[r3_start_index:,:]

print(df.shape[0])
print(df1.shape[0])
print(df2.shape[0])

df1.to_csv("D3r2_.log", sep='\t', encoding='utf-8', index = False, header = True)
df2.to_csv("D3r3_.log", sep='\t', encoding='utf-8', index = False, header = True)