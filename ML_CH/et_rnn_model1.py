from tensorflow import keras
from tensorflow.keras import layers

def create_model(number_of_features, window_size, number_of_classes):
    input_layer = keras.Input(shape=(number_of_features*window_size, 1))

    x = layers.Conv1D(
        filters=32, kernel_size=3, strides=2, activation="relu", padding="same"
    )(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    output_layer = layers.Dense(number_of_classes, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)