import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import csv

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EEG")

###############################################################################
filename = os.path.join(DATA_DIR, "D3r3_1_KB.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D3r3_2_KB.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")
filename = os.path.join(DATA_DIR, "D3r3_3_KB.csv")
df3 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2, df3])

filename = os.path.join(DATA_DIR, "D3r3_KB.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D6r2_1_AE.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D6r2_2_AE.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D6r2_AE.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D6r5_1_HC.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D6r5_2_HC.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D6r5_HC.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D6r6_1_HC.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D6r6_2_HC.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D6r6_HC.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D7r4_1_ML.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D7r4_2_ML.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")
filename = os.path.join(DATA_DIR, "D7r4_3_ML.csv")
df3 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D7r4_4_ML.csv")
df4 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2, df3, df4])

filename = os.path.join(DATA_DIR, "D7r4_ML.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D7r6_1_ML.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D7r6_2_ML.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D7r6_ML.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D9r3_1_RE.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D9r3_2_RE.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D9r3_RE.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)

###############################################################################
filename = os.path.join(DATA_DIR, "D9r4_1_SV.csv")
df1 = pd.read_csv(filename, sep=';', encoding="utf-8")
filename = os.path.join(DATA_DIR, "D9r4_2_SV.csv")
df2 = pd.read_csv(filename, sep=';',  encoding="utf-8")

df = pd.concat([df1, df2])

filename = os.path.join(DATA_DIR, "D9r4_SV.csv")

df[:0].to_csv(filename, sep=';', encoding='utf-8', index=False, header = True)
df.to_csv(filename, mode="a", sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index = False, header = False)
