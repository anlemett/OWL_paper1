import warnings
warnings.filterwarnings('ignore')

#import sys #exit
import os
import numpy as np
import pandas as pd

from sklearn import model_selection

from rnn_model import weight_classes, train_and_evaluate, BINARY, TIME_INTERVAL_DURATION

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

###############################################################################

def main():
    
    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")
    
    #print(np.isnan(TS_np).any())
    #nan_count = np.count_nonzero(np.isnan(TS_np))
    #print(nan_count)
    
    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # 180 -> (631, 45000, 15), 60 -> (1768, 15000, 15)
    if TIME_INTERVAL_DURATION == 180: 
        TS_np = TS_np.reshape((631, 45000, 15))
    else: # 60
        TS_np = TS_np.reshape((1768, 15000, 15))

    full_filename = os.path.join(ML_DIR, "ML_ET_EEG_" + str(TIME_INTERVAL_DURATION) + "__EEG.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    ###########################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    print(scores)
    
    if BINARY:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        #th = eeg_series.quantile(.93)
        th = eeg_series.quantile(.5)
        scores = [1 if score < th else 2 for score in scores]

    else:
        #Split into 3 bins by percentile
        eeg_series = pd.Series(scores)
        (th1, th2) = eeg_series.quantile([.52, .93])
        scores = [1 if score < th1 else 3 if score > th2 else 2 for score in scores]

    print(scores)
    
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
 
    ###########################################################################
    scores, weight_dict = weight_classes(scores)

    ###########################################################################
  
    # Spit the data into train and test
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        TS_np, scores, test_size=0.1, random_state=0, shuffle=False
        )

    #print(np.any(np.isnan(test_X)))
    #print(np.any(np.isnan(test_Y)))
    
    print(
        f"Length of train_X : {len(train_X)}\nLength of test_X : {len(test_X)}\nLength of train_Y : {len(train_Y)}\nLength of test_Y : {len(test_Y)}"
        )

    ###########################################################################
    batch_size = 8
    epoch_num = 30

    (ac, f1, conv_model_history) = train_and_evaluate(train_X, train_Y, test_X, test_Y, weight_dict, batch_size, epoch_num) 
    return conv_model_history

conv_model_history = main()
        

###############################################################################
import matplotlib.pyplot as plt
from tensorflow import keras

def plot_history_metrics(history: keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    
    full_filename = os.path.join(FIG_DIR, "ET_EEG.png")
    plt.savefig(full_filename)
    plt.show()

plot_history_metrics(conv_model_history)
