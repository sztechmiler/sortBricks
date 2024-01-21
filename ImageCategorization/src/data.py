# Scripts to download or generate dat
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def create_data_generators(data_dir="../data/raw", batch_size=32, target_size=(256, 256)):
    """
    Creates and returns an ImageDataGenerator for training.

    :param data_dir: Directory where the training data is located.
    :param batch_size: Size of the batches of data (default: 32).
    :param target_size: The dimensions to which all images found will be resized (default: (256, 256)).
    :return: A Keras ImageDataGenerator object.
    """
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% of data will be used for validation
    )

    train_generator = datagen_train.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Set as training data
    )

    validation_generator = datagen_train.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Set as validation data
    )

    # Assuming no separate test set; modify if you have a separate test set
    test_generator = None

    return train_generator, validation_generator, test_generator
