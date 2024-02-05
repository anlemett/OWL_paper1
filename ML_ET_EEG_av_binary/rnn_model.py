from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow import keras

def create_model(number_of_features, window_size, number_of_classes):
    
    #input_layer = keras.Input(shape=(window_size, number_of_features, 1))
    a = number_of_features
    b = 2*a
    c = 2*b
    d = 2*c
    
    e = 10*number_of_features
    f = 2*e
    g = 2*f
    
    model = models.Sequential([
        #layers.Conv1D(
        #    filters=a, kernel_size=3, strides=2, activation="relu", padding="same"
        #),
        #layers.BatchNormalization(),
        
        layers.Conv1D(
            filters=e, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),
        
        layers.Conv1D(
            filters=f, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),

        layers.Conv1D(
            filters=g, kernel_size=3, strides=2, activation="relu", padding="same"
        ),
        layers.BatchNormalization(),

        layers.Dropout(0.2),

        layers.Flatten(),
        
        layers.Dense(number_of_classes, activation="sigmoid")
        ])

    return model
