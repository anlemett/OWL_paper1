import warnings
warnings.filterwarnings('ignore')

#import sys #exit

import numpy as np
import itertools

import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing, model_selection

from et_rnn_model import create_model
from et_ts_np import get_TS_np

TIME_INTERVAL_DURATION = 180  #sec
WINDOW_SIZE = 249 * TIME_INTERVAL_DURATION

all_features = ['Saccade', 'Fixation',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch',	'HeadRoll']
# for testing:
all_features = ['LeftBlinkOpeningAmplitude', 'LeftBlinkClosingSpeed', 'RightBlinkClosingAmplitude']

###############################################################################
# Test defferent features
###############################################################################

def test_different_features(features):
###############################################################################
    (TS_np, scores) = get_TS_np(features, TIME_INTERVAL_DURATION)

###############################################################################
    #Shuffle data

    scores_np = np.array(scores)
    print(TS_np.shape)
    print(scores_np.shape)

    zipped = list(zip(TS_np, scores_np))

    np.random.shuffle(zipped)

    TS_np, scores_np = zip(*zipped)

    scores = list(scores_np)

    print(scores)
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
        TS_np, scores, test_size=0.15, random_state=42, shuffle=True
        )

    print(
        f"Length of train_X : {len(train_X)}\nLength of test_X : {len(test_X)}\nLength of train_Y : {len(train_Y)}\nLength of test_Y : {len(test_Y)}"
        )


###############################################################################
# Reshape the data
    x_train = np.asarray(train_X).astype(np.float32).reshape(-1, len(features)*WINDOW_SIZE, 1)

    y_train = np.asarray(train_Y).astype(np.float32).reshape(-1, 1)
    y_train = keras.utils.to_categorical(y_train) # transform to one-hot label

    x_test = np.asarray(test_X).astype(np.float32).reshape(-1, len(features)*WINDOW_SIZE, 1)
    y_test = np.asarray(test_Y).astype(np.float32).reshape(-1, 1)
    y_test = keras.utils.to_categorical(y_test) # transform to one-hot label

    print(x_train.shape)
    print(y_train.shape)

###############################################################################
    #BATCH_SIZE = 2
    BATCH_SIZE = 1

    #to make the result reproducable
    keras.utils.set_random_seed(1)


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
            monitor="val_top_k_categorical_accuracy",
            factor=0.2,
            patience=2,
            min_lr=0.000001,
            ),
        #keras.callbacks.EarlyStopping(monitor='loss',
        #                              patience=2,
        #                              mode='min')
        ]

###############################################################################
    #precisions = []
    #recalls = []
    conv_model = create_model(len(features), WINDOW_SIZE, len(set(scores)))

    #print(conv_model.summary())


###############################################################################

    epochs = 5

    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
    loss = keras.losses.CategoricalCrossentropy()


###############################################################################
    conv_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            #keras.metrics.TopKCategoricalAccuracy(k=3),
            keras.metrics.Accuracy(),
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
    )


###############################################################################
    loss, accuracy, auc, precision, recall = conv_model.evaluate(test_dataset)
    print(f"Loss : {loss}")
    print(f"Accuracy : {accuracy}")
    print(f"Area under the Curve (ROC) : {auc}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    
    #precisions.append(precision)
    #recalls.append(recall)
    return (precision, recall)


precisions = []
recalls = []

precision_best_combination = []
recall_best_combination = []
f1_score_best_combination = []
best_precision = 0
best_recall = 0
best_f1_score = 0

#for i in range (1, 16):
#for i in range (1, 4):
for i in range (1, 2):
    '''    
    combinations = list(itertools.combinations(all_features, i))
    
    number_of_combinations = len(combinations)

    for j in range(0, number_of_combinations):
        f = open('results.txt', 'a')
        print("Number of features, number of combinations, combination")
        print(i, number_of_combinations, j)
        
        (precision, recall) = test_different_features(list(combinations[j]))
    '''    
    for j in range(0, 1):
        (precision, recall) = test_different_features(all_features)
        
        if not ((precision == 0) and ((recall == 0))):
            f1_score = (precision*recall)/(precision+recall)
        else:
            f1_score = 0
        print(precision)
        print(recall)
        print(f1_score)
        '''        
        if precision > best_precision:
            best_precision = precision
            precision_best_combination = combinations[j]
        if recall > best_recall:
            best_recall = recall
            recall_best_combination = combinations[j]
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            f1_score_best_combination = combinations[j]
        
        f.write("Number of features\n")
        f.write(str(i)+'\n')
        f.write("number of combinations\n")
        f.write(str(number_of_combinations)+'\n')
        f.write("combination\n")
        f.write(str(j)+'\n')
        f.write("Best precision\n")
        f.write(str(best_precision)+'\n')
        f.write(str(precision_best_combination)+'\n')
        f.write("Best recall\n")
        f.write(str(best_recall)+'\n')
        f.write(str(recall_best_combination)+'\n')
        f.write("Best f1-score\n")
        f.write(str(best_f1_score)+'\n')
        f.write(str(f1_score_best_combination)+'\n')
        f.close()
        
    
print(best_precision)
print(precision_best_combination)
print(best_recall)
print(recall_best_combination)
print(best_f1_score)
print(f1_score_best_combination)
'''
 
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

#plot_history_metrics(conv_model_history)

