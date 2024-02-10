import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")
print("reading CH data")

temp_scores_np = np.loadtxt(full_filename, delimiter=" ")
temp_scores = temp_scores_np.tolist()

if 0 in temp_scores:
    temp_scores = temp_scores.remove(0)


equal_to_1 = [x for x in temp_scores if x == 1]
equal_to_2 = [x for x in temp_scores if x == 2]
equal_to_3 = [x for x in temp_scores if x == 3]
more_than_3 = [x for x in temp_scores if x >= 3]
less_or_3 = [x for x in temp_scores if x < 3]

equal_to_1_num = len(equal_to_1)
equal_to_2_num = len(equal_to_2)
equal_to_3_num = len(equal_to_3)
more_than_3_num = len(more_than_3)
less_or_3_num = len(less_or_3)
scores_num = len(temp_scores)

print(equal_to_1_num)
print(equal_to_2_num)
print(more_than_3_num)
print(less_or_3_num)
print(scores_num)

print(more_than_3_num/scores_num)
print(less_or_3_num/scores_num)

more_than_4 = [x for x in temp_scores if x >= 4]
more_than_4_num = len(more_than_4)
print(more_than_4_num)
print(more_than_4_num/scores_num)

print(equal_to_1_num/scores_num) #52%
print(equal_to_2_num/scores_num) # 23%
print(equal_to_3_num/scores_num) # 18%


