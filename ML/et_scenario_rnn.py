import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing, model_selection

from et_rnn_model import create_model

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
DATA_DIR = os.path.join(DATA_DIR, "EyeTracking")

#TIME_INTERVAL_DURATION = 60 #sec
TIME_INTERVAL_DURATION = 60 #sec
WINDOW_SIZE = 249 * TIME_INTERVAL_DURATION


features = ['Saccade', 'Fixation',
            'LeftPupilDiameter', 'RightPupilDiameter',
            'LeftBlinkClosingAmplitude', 'LeftBlinkOpeningAmplitude',
            'LeftBlinkClosingSpeed', 'LeftBlinkOpeningSpeed',
            'RightBlinkClosingAmplitude', 'RightBlinkOpeningAmplitude',
            'RightBlinkClosingSpeed', 'RightBlinkOpeningSpeed',
            'HeadHeading', 'HeadPitch',	'HeadRoll']
# for testing:
features = ['Saccade', 'Fixation',
            'LeftBlinkClosingAmplitude', 'LeftBlinkClosingSpeed']
features = ['LeftBlinkClosingAmplitude', 'LeftBlinkClosingSpeed']

columns = ['UnixTimestamp'] + ['ValuesPerSecond'] + features


def getTimeInterval(timestamp, first_timestamp):

    return math.trunc((timestamp - first_timestamp)/TIME_INTERVAL_DURATION) + 1


def createTimeSeriesDf(df, score):
    
    #fill the null rows with the mean of respective columns
    df = df.fillna(df.mean())

    first_timestamp = df['UnixTimestamp'].loc[0]

    df['timeInterval'] = df.apply(lambda row: getTimeInterval(row['UnixTimestamp'],
                                                              first_timestamp
                                                              ),
                                  axis=1)

    new_columns = ['timeInterval'] + columns
    df = df[new_columns]

    last_time_interval = list(df['timeInterval'])[-1]
    
    #####################################
    intervals = []

    timeseries_df = pd.DataFrame(columns = [features])

    for ti in range (1, last_time_interval + 1):
        ti_df = df[df['timeInterval']==ti]
        
        if ti_df.empty:
            continue
        if len(ti_df.index) < WINDOW_SIZE:
            continue
        
        row_lst = []

        for feature in features:
            feature_lst = ti_df[feature].tolist()
            row_lst.append(feature_lst[0:WINDOW_SIZE])
            
        timeseries_df.loc[ti-1] = row_lst
        intervals.append(ti)
    
    timeseries_df['timeInterval'] = timeseries_df.index
    timeseries_df = timeseries_df.reset_index(drop=True)
    
    number_of_rows = len(timeseries_df.index)
    scores = [score] * number_of_rows
    
    # number_of_rows <= last_time_interval, because some time intervals might
    # miss data (or have insufficient amount)
    
    return (timeseries_df, scores, number_of_rows)


###############################################################################
full_filename = os.path.join(DATA_DIR, "ET_D3r1_KB.csv")
df_low = pd.read_csv(full_filename, sep=' ', low_memory=False)
(TS_df, all_scores, number_of_time_intervals) = createTimeSeriesDf(df_low, 1)

full_filename = os.path.join(DATA_DIR, "ET_D3r2_KB.csv")
df_high = pd.read_csv(full_filename, sep=' ', low_memory=False)
(temp_df, temp_scores, temp_number) = createTimeSeriesDf(df_high, 3)
TS_df = pd.concat([TS_df, temp_df]).reset_index(drop=True)
all_scores = all_scores + temp_scores
number_of_time_intervals = number_of_time_intervals + temp_number

full_filename = os.path.join(DATA_DIR, "ET_D3r3_KB.csv")
df_medium = pd.read_csv(full_filename, sep=' ', low_memory=False) 
 
(temp_df, temp_scores, temp_number) = createTimeSeriesDf(df_medium, 2)
TS_df = pd.concat([TS_df, temp_df]).reset_index(drop=True)
all_scores = all_scores + temp_scores
number_of_time_intervals = number_of_time_intervals + temp_number


###############################################################################
scaler = preprocessing.MinMaxScaler()
#scale the values

for index, row in TS_df.iterrows():
    for feature in features:
        current_lst = row[feature]
        TS_df.at[index, feature] = scaler.fit_transform(np.asarray(current_lst).reshape(-1, 1))
        
le = preprocessing.LabelEncoder()  # Generates a look-up table
le.fit(all_scores)
all_scores = le.transform(all_scores)


###############################################################################
# create numpy array for X
# timeseries_np shape (a,b,c):
# a - number of time periods, b - number of features, c - number of measures per time period
timeseries_np = np.empty(0)

number_of_rows = len(TS_df.index)
print(number_of_rows)
print(number_of_time_intervals)

for index in range(0, number_of_rows):
    for feature in features:
        #print(TS_df.loc[index].at[feature])
        new_np = np.array(TS_df.loc[index].at[feature])
        timeseries_np = np.append(timeseries_np, new_np)
        
timeseries_np = timeseries_np.reshape([number_of_rows, len(features), WINDOW_SIZE])
print(timeseries_np.shape)
print(len(all_scores))


###############################################################################
# Spit the data into train and test
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    timeseries_np, all_scores, test_size=0.15, random_state=42, shuffle=True
)

#np.split(timeseries_np, [])

print(
    f"Length of train_X : {len(train_X)}\nLength of test_X : {len(test_X)}\nLength of train_Y : {len(train_Y)}\nLength of test_Y : {len(test_Y)}"
)


###############################################################################
# Reshape the data
x_train = np.asarray(train_X).astype(np.float32).reshape(-1, len(features)*WINDOW_SIZE, 1)

y_train = np.asarray(train_Y).astype(np.float32).reshape(-1, 1)
y_train = keras.utils.to_categorical(y_train)

x_test = np.asarray(test_X).astype(np.float32).reshape(-1, len(features)*WINDOW_SIZE, 1)
y_test = np.asarray(test_Y).astype(np.float32).reshape(-1, 1)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)

###############################################################################
#BATCH_SIZE = 64
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

###############################################################################
vals_dict = {}
for i in all_scores:
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
conv_model = create_model(len(features), WINDOW_SIZE, len(set(all_scores)))

print(conv_model.summary())


###############################################################################
#epochs = 5
epochs = 20

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
]

optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
loss = keras.losses.CategoricalCrossentropy()


###############################################################################
conv_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=3),
        keras.metrics.AUC(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
    ],
)

print(train_dataset)


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
print(f"Top 3 Categorical Accuracy : {accuracy}")
print(f"Area under the Curve (ROC) : {auc}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")


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

