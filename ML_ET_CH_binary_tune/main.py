import warnings
warnings.filterwarnings('ignore')

import sys #exit

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing, model_selection

from rnn_model import create_model
from ts_np import get_TS_np

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")

WINDOW_SIZE = 250 * 180

all_features = ['Saccade', 'Fixation',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch',	'HeadRoll']

#to make the result reproducable
keras.utils.set_random_seed(0)

###############################################################################
# Test defferent features
###############################################################################

def test_different_features(features):
###############################################################################
    full_filename = os.path.join(ML_DIR, "ML_ET_CH__ET.csv")
    print("reading data")

    # Load the 2D array from the CSV file
    TS_np = np.loadtxt(full_filename, delimiter=" ")

    # Reshape the 2D array back to its original 3D shape
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
    
    scores = [1 if score <3 else 2 for score in scores]

    print(scores)
    
    max_score = max(scores)
    print(f"Max score : {max_score}")
    print(set(scores))
    #sys.exit(0)


###############################################################################
    # conv_model.fit requires `class_weight` to be a dict with keys from 0 to 
    # one less than the number of classes
    le = preprocessing.LabelEncoder()  # Generates a look-up table
    le.fit(scores)
    scores = le.transform(scores)

    vals_dict = {}
    for i in scores:
        if i in vals_dict.keys():
            vals_dict[i] += 1
        else:
            vals_dict[i] = 1
    total = sum(vals_dict.values())

    # Formula used - Naive method where
    # weight = 1 - (no. of samples present / total no. of samples)
    # So more the samples, lower the weight

    weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
    print(weight_dict)

###############################################################################
    
    # Spit the data into train and test
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        TS_np, scores, test_size=0.1, random_state=0, shuffle=False
        )

    #print(
    #    f"Length of train_X : {len(train_X)}\nLength of test_X : {len(test_X)}\nLength of train_Y : {len(train_Y)}\nLength of test_Y : {len(test_Y)}"
    #    )
    
###############################################################################
    # Reshape the data
    
    x_train = np.asarray(train_X).astype(np.float32).reshape(-1, WINDOW_SIZE, len(features))
    y_train = np.asarray(train_Y).astype(np.float32).reshape(-1, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes=2) # transform to one-hot label

    x_test = np.asarray(test_X).astype(np.float32).reshape(-1, WINDOW_SIZE, len(features))
    y_test = np.asarray(test_Y).astype(np.float32).reshape(-1, 1)
    y_test = keras.utils.to_categorical(y_test, num_classes=2) # transform to one-hot label

    print(x_train.shape)
    print(y_train.shape)

###############################################################################
    #BATCH_SIZE = 1
    BATCH_SIZE = 16


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


###############################################################################
    callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="loss"
                ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_categorical_accuracy",
                factor=0.2,
                patience=2,
                min_lr=0.000001,
                ),
            #keras.callbacks.EarlyStopping(monitor='loss',
            #                              patience=2,
            #                              mode='min')
            ]

###############################################################################
    conv_model = create_model(len(features), WINDOW_SIZE, len(set(scores)))
    #print(conv_model.summary())

###############################################################################

    #epochs = 10
    epochs = 30

    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.0001)
    loss = keras.losses.BinaryCrossentropy()


###############################################################################
    conv_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            ],
        )


    conv_model_history = conv_model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
        class_weight=weight_dict,
        #validation_split = 0
        )


###############################################################################
    loss, cat_accuracy, auc, precision, recall = conv_model.evaluate(test_dataset)
    
    if not ((precision == 0) and ((recall == 0))):
        f1_score = 2 *(precision*recall)/(precision+recall)
    else:
        f1_score = 0

    print(f"Loss : {loss}")
    print(f"Categorical Accuracy : {cat_accuracy}")
    print(f"Area under the Curve (ROC) : {auc}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    print(f"F1-score : {f1_score}")
    
    return conv_model_history


conv_model_history = test_different_features(all_features)
 
###############################################################################
###############################################################################
import matplotlib.pyplot as plt

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
    plt.show()

plot_history_metrics(conv_model_history)
