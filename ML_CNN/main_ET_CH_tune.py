import warnings
warnings.filterwarnings('ignore')

#import sys #exit
import time
import os
import numpy as np

from sklearn import model_selection

from rnn_model import weight_classes, train_and_evaluate, BINARY

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

###############################################################################

def main():

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")

    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # (667, 45000, 15)
    TS_np = TS_np.reshape((667, 45000, 15))

    full_filename = os.path.join(ML_DIR, "ML_ET_CH__CH.csv")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

###############################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)
    
    if BINARY:
        scores = [1 if score < 3 else 2 for score in scores]
    else:
        scores = [1 if score < 2 else 3 if score > 2 else 2 for score in scores]

    print(scores)
    
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    #sys.exit(0)

    ###########################################################################
    scores, weight_dict = weight_classes(scores)
    
    ###########################################################################
    
    # Spit the data into train and test
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        TS_np, scores, test_size=0.1, random_state=0, shuffle=False
        )

    print(
        f"Length of train_X : {len(train_X)}\nLength of test_X : {len(test_X)}\nLength of train_Y : {len(train_Y)}\nLength of test_Y : {len(test_Y)}"
        )
    
    ###########################################################################
    batch_size = 8
    epoch_num = 15
    cat_accuracy, f1_score, conv_model_history = train_and_evaluate(train_X, train_Y, test_X, test_Y, weight_dict, batch_size, epoch_num)
    
    return conv_model_history

start_time = time.time()

conv_model_history = main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")

 
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
    
    full_filename = os.path.join(FIG_DIR, "ET_CH.png")
    plt.savefig(full_filename)
    plt.show()

plot_history_metrics(conv_model_history)
