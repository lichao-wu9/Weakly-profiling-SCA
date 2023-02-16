from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *

def CNN(input_size, classes=256, desync_level=0):
    batch_size = 800
    model = Sequential(name="best_cnn_id_rpoi_7500_chesctf")
    model.add(RandomTranslation(input_shape=(input_size, 1), width_factor=desync_level/input_size, height_factor=0, fill_mode='wrap'))
    model.add(Conv1D(kernel_size=40, strides=20, filters=4, activation="selu", padding="same"))
    model.add(AveragePooling1D(pool_size=2, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(400, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(400, activation="selu", kernel_initializer="random_uniform"))
    model.add(Dense(classes, activation="softmax"))
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, batch_size

