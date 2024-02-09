import warnings
warnings.filterwarnings('ignore')

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from sklearn import preprocessing

#to make the result reproducable
keras.utils.set_random_seed(0)

BINARY = True

TIME_INTERVAL_DURATION = 60  #sec
NUMBER_OF_FEATURES = 15
WINDOW_SIZE = 250 * TIME_INTERVAL_DURATION


def weight_classes(scores):
    
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
    
    return scores, weight_dict


def create_model(number_of_features, number_of_classes):
    
    #input_layer = keras.Input(shape=(,,))
    a = number_of_features
    b = 2*a
    c = 2*b
    d = 2*c
    
    if BINARY:
        last_layer = layers.Dense(number_of_classes, activation="sigmoid")
    else:
        last_layer = layers.Dense(number_of_classes, activation="softmax")
    
    model = models.Sequential([
        layers.Conv1D(
            filters=a, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        
        layers.Conv1D(
            filters=b, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        
        layers.Conv1D(
            filters=c, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),

        #layers.Conv1D(
        #    filters=d, kernel_size=3, strides=2, activation="relu", padding="same"
        #),
        #layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Flatten(),
        
        last_layer
        ])

    return model


def train_and_evaluate(train_X, train_Y, test_X, test_Y, weight_dict, batch_size, epoch_num):
    # lists -> numpy arrays
    x_train = np.asarray(train_X).astype(np.float32).reshape(-1, WINDOW_SIZE, NUMBER_OF_FEATURES)
    y_train = np.asarray(train_Y).astype(np.float32).reshape(-1, 1)
    x_test = np.asarray(test_X).astype(np.float32).reshape(-1, WINDOW_SIZE, NUMBER_OF_FEATURES)
    y_test = np.asarray(test_Y).astype(np.float32).reshape(-1, 1)

    # transform to one-hot label
    number_of_classes = len(weight_dict)
    y_train = keras.utils.to_categorical(y_train, num_classes=number_of_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=number_of_classes)

    print(x_train.shape)
    print(y_train.shape)

    ###############################################################################
    BATCH_SIZE = batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    ###############################################################################
    callbacks = [
        #keras.callbacks.ModelCheckpoint(
        #    "best_model.h5", save_best_only=True, monitor="loss"
        #    ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_categorical_accuracy",
            factor=0.2,
            patience=2,
            min_lr=0.000001,
            ),
        #keras.callbacks.EarlyStopping(
        #    monitor='loss',
        #    patience=2,
        #    mode='min'
        #    )
        ]

    #######################################################################
    conv_model = create_model(NUMBER_OF_FEATURES, number_of_classes)

    #print(conv_model.summary())

    #######################################################################

    epochs = epoch_num

    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.0001)
        
    if BINARY:
        loss = keras.losses.BinaryCrossentropy()
    else:
        loss = keras.losses.CategoricalCrossentropy()

    #######################################################################
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
        )

    #######################################################################
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

    return cat_accuracy, f1_score, conv_model_history