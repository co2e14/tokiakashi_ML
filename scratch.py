from gc import callbacks
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print("Using TensorFlow v%s" % tf.__version__)
acc_str = "accuracy" if tf.__version__[:2] == "2." else "acc"

data_dir = pathlib.Path("C:/Users/ULTMT/Documents/code/tokiakashi_ML/newstyle")
batch_size = 64
img_height = 24
img_width = 32
seed = 32985476

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

normalization_layer = keras.layers.Rescaling(
    1.0 / 255, input_shape=(img_height, img_width, 3)
)


model = Sequential()
model.add(normalization_layer)
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10, activation="sigmoid"))

model.summary()

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

cp_callback = keras.callbacks.ModelCheckpoint(filepath="checkpoints/", verbose=1, save_weights_only=True, save_freq=64*batch_size)

training = model.fit(
    train_ds, epochs=3, batch_size=batch_size, validation_data=(val_ds), callbacks=cp_callback
)

# plot accuracy
plt.figure(dpi=100, figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training.history[acc_str], label="Accuracy on training data")
plt.plot(training.history["val_" + acc_str], label="Accuracy on test data")
plt.legend()
plt.title("Accuracy")

# plot loss
plt.subplot(1, 2, 2)
plt.plot(training.history["loss"], label="Loss on training data")
plt.plot(training.history["val_loss"], label="Loss on test data")
plt.legend()
plt.title("Loss")
plt.show()
