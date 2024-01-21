import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
keras = tf.keras
# from tf import

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
IM_SIZE = 224
BATCH_SIZE = 32
def splits(dataset, train_ratio, val_ratio, test_ratio):
    dataset_size = len(dataset)
    train_dataset = dataset.take(int(train_ratio * dataset_size))
    val_dataset = dataset.skip(int(train_ratio * dataset_size)).take(int(val_ratio * dataset_size))
    test_dataset = dataset.skip(int((train_ratio  + val_ratio) * dataset_size)).take(int(test_ratio * dataset_size))
    return train_dataset, val_dataset, test_dataset

def resizing_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

dataset, dataset_info  = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])


# dataset = tf.data.Dataset.range(10)
train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

train_dataset = train_dataset.map(resizing_rescale)
val_dataset = val_dataset.map(resizing_rescale)

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
                            keras.layers.InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)), 
                            keras.layers.Conv2D(filters=6, kernel_size=3, strides=1, padding='valid',activation="relu" ),
                            keras.layers.BatchNormalization(),
                            keras.layers.MaxPool2D(pool_size=2, strides=2),
                            keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='valid',activation="relu" ),
                            keras.layers.BatchNormalization(),
                            keras.layers.MaxPool2D(pool_size=2, strides=2),

                            keras.layers.Flatten(),
                            keras.layers.Dense(100, activation="relu"),
                            keras.layers.BatchNormalization(),
                            keras.layers.Dense(10, activation="relu"),
                            keras.layers.BatchNormalization(),
                            keras.layers.Dense(1, activation="sigmoid")


])

model.summary()

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01),
              loss = keras.losses.BinaryCrossentropy(),
              metrics = 'accuracy')

history = model.fit(train_dataset, validation_data = val_dataset, epochs = 10, verbose = 1)

model.save("malaria_first_model")