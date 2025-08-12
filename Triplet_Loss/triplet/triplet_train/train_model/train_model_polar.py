import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_transform as io

import tensorflow as tf
import warnings
import polarTransform as pt

from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

import matplotlib.pyplot as plt

from configs import *
from images_utils import *

warnings.filterwarnings("ignore")

#https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
def plot_model_histogram(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    print(history)
    print(epochs_range)

    plt.figure() #figsize=(8, 8)
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plt.savefig(path_as_string(results_path)+"/model_histogram.png")

#https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
def create_image_data_generator():
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        brightness_range=(0.4, 0.9), #the range starts from zero which signifies no brightness of the image. the upper range is 1 which signifies the maximum range of the brightness.
        # rotation_range=180,
        # width_shift_range=5,
        # height_shift_range=5,
    )
    return image_data_generator;

def get_train_data():
    image_data_generator = create_image_data_generator()
    path_polar = path_as_string(split_dataset_train_path)

    print(path_polar)
    return image_data_generator.flow_from_directory(
        path_polar,
        # save_to_dir=path_as_string(augmentation_results_path) + "/train",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    )

def get_validation_data():
    image_data_generator = create_image_data_generator()
    path_polar = path_as_string(split_dataset_validation_path)
    print(path_polar)

    return image_data_generator.flow_from_directory(
        path_polar,
        # save_to_dir=path_as_string(augmentation_results_path) + "/validation",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    )

def load_pre_trained_model():
    # Load pre-trained model
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    return base_model

def create_model(base_model):
    # Unfreeze the pretrained weights
    base_model.trainable = True
    
    print('number of layers in base model ' + str(len(base_model.layers)))

    frozen_num_layer = -50 # minus przed liczbą oznacza pomiń tyle elementów od końca tablicy 
    # Freeze some layers [!]
    for layer in base_model.layers[:frozen_num_layer]:
        # if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")
    ])

    print(model.summary())

    return model;

def compile_model(model):
    learning_rate = 1e-4
    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
    return model

def create_callbacks_for_fit_model():
    logdir = path_as_string(log_path) + '/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=3)
    checkpoint = ModelCheckpoint(filepath=path_as_string(model_path) + "/" + model_name,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    callbacks = [checkpoint, earlystopping, reduce_lr, tensorboard_callback]
    return callbacks

def main():
    # Load data
    train_data_iterator = get_train_data()
    validation_data_iterator = get_validation_data()

    base_model = load_pre_trained_model()
    model = create_model(base_model)
    model = compile_model(model)
    
    # Train model
    callbacks = create_callbacks_for_fit_model()
    validation_steps = validation_data_iterator.n // batch_size
    steps_per_epoch = train_data_iterator.n // batch_size
    epochs = 20

    hist = model.fit(
        train_data_iterator,
        validation_data=validation_data_iterator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        batch_size=batch_size,
        use_multiprocessing ='true'
    )

    print("finish training model")

    plot_model_histogram(hist, epochs)
    print(model.summary())
    
    print("model is saved")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
# Restrict TensorFlow to only use the first GPU
    try:
        # tf.config.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        
        main()
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)