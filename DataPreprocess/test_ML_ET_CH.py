import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
print("reading ET data")

# Load the 2D array from the CSV file
temp_et_np = np.loadtxt(full_filename, delimiter=" ")

# Reshape the 2D array back to its original 3D shape
# (667, 45000, 15)
temp_et_np = temp_et_np.reshape((667, 45000, 15))

print(temp_et_np.shape)

full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
print("reading CH data")

temp_scores_np = np.loadtxt(full_filename, delimiter=" ")
temp_scores = temp_scores_np.tolist()

print(len(temp_scores))
