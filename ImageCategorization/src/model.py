 # Scripts to train and evaluate the model
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
keras = tf.keras
# from tf import

from data import create_data_generators


# Constants
IM_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "C:\\Users\\Greg\\source\\repos\\SortBricks\\ImageCategorization\\data\\raw"

train_generator, val_generator, _ = create_data_generators(DATA_DIR, BATCH_SIZE, (IM_SIZE, IM_SIZE))

model = tf.keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)), 
    keras.layers.Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2, strides=2),

    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(8, activation="softmax")  # Change to 8 for 8 output classes
])

# Compile and fit the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # or 'sparse_categorical_crossentropy'
              metrics=['accuracy'])


history = model.fit(train_generator, validation_data=val_generator, epochs=100, verbose=1)

# Save your model
model.save("your_model_name.h5")