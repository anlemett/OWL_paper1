from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow import keras

def create_model(number_of_features, window_size, number_of_classes):
    
    #hidden_nodes = 10
    
    model = models.Sequential([
        layers.Conv1D(
            filters=number_of_features, kernel_size=3, strides=2, activation="relu", padding="same"
        ),        
        #layers.LSTM(hidden_nodes, return_sequences=False, input_shape=(number_of_features*window_size, 1)),
        layers.Dropout(0.2),
        #layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(3)
        ])

    return model
