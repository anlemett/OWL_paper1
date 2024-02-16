import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
from statistics import mean

from sklearn import model_selection

from rnn_model import weight_classes, train_and_evaluate, BINARY

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

###############################################################################

def main():

    full_filename = os.path.join(ML_DIR, "ML_ET_Scenario__ET.csv")
    print("reading ET data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")

    # Reshape the 2D array back to its original 3D shape
    # (number_of_timeintervals, TIME_INTERVAL_DURATION*250, number_of_features)
    # (673, 45000, 15)
    TS_np = TS_np.reshape((673, 45000, 15))

    full_filename = os.path.join(ML_DIR, "ML_ET_Scenario__Scenario.csv")
    print("reading Scenario data")

    scores_np = np.loadtxt(full_filename, delimiter=" ")

    ###########################################################################
    #Shuffle data

    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)

    if BINARY:
        scores = [1 if score < 2 else 2 for score in scores]
        #scores = [1 if score < 3 else 2 for score in scores]

    print(scores)
    
    number_of_classes = len(set(scores))
    print(f"Number of classes : {number_of_classes}")
    
    ###########################################################################
    scores, weight_dict = weight_classes(scores)
    
    ###########################################################################

    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = model_selection.KFold(n_splits=num_folds, shuffle=True)
    
    # K-fold Cross Validation model evaluation
    
    # Define per-fold score containers
    acc_per_fold = []
    f1_per_fold = []
    
    fold_no = 1
    for train_idx, test_idx in kfold.split(scores):
    
        train_X = np.array(TS_np)[train_idx.astype(int)]
        train_Y = scores[train_idx.astype(int)]
        test_X = np.array(TS_np)[test_idx.astype(int)]
        test_Y = scores[test_idx.astype(int)]
        #######################################################################
        batch_size = 8
        epoch_num = 10

        (cat_accuracy, f1_score, history) = train_and_evaluate(train_X, train_Y, test_X, test_Y, weight_dict, batch_size, epoch_num)   
        
        acc_per_fold.append(cat_accuracy)
        f1_per_fold.append(f1_score)
        
        # Increase fold number
        fold_no = fold_no + 1

    print(acc_per_fold)
    print(f1_per_fold)
    
    print(mean(acc_per_fold))
    print(mean(f1_per_fold))


start_time = time.time()

main()   

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
