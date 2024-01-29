from tensorflow import keras
from tensorflow.keras import layers

def create_model(number_of_features, window_size, number_of_classes, filter):
    
    input_layer = keras.Input(shape=(number_of_features*window_size, 1))

    x = layers.Conv1D(
        filters=filter, kernel_size=3, strides=2, activation="relu", padding="same"
    )(input_layer)

    x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(input_layer)
    
    output_layer = layers.Dense(number_of_classes, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)