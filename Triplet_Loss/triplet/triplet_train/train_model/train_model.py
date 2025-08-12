import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
import warnings
import numpy as np
from datetime import datetime
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

import time
import matplotlib.pyplot as plt

from configs import *

warnings.filterwarnings("ignore")

# add tensorboard
#https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
def plot_model_histogram(history, epochs):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    triplet_acc = history.history.get('triplet_accuracy', [])
    val_triplet_acc = history.history.get('val_triplet_accuracy', [])

    epochs_range = range(len(loss))  

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, triplet_acc, label='Training Triplet Accuracy')
    plt.plot(epochs_range, val_triplet_acc, label='Validation Triplet Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Triplet Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(path_as_string(results_path) + "/model_histogram.png")
    plt.show()








def create_image_data_generator():
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        #brightness_range=(0.2, 0.9), #the range starts from zero which signifies no brightness of the image. the upper range is 1 which signifies the maximum range of the brightness.
        brightness_range=[0.2, 1.5],
        # rotation_range=350,
        # rotation_range=30,
        # vertical_flip=True,
        # zoom_range=0.3,
        # width_shift_range=0.2, height_shift_range=0.2
    )
    return image_data_generator

def create_image_data_generator_with_split():
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        #brightness_range=(0.2, 0.9), #the range starts from zero which signifies no brightness of the image. the upper range is 1 which signifies the maximum range of the brightness.
        brightness_range=[0.2, 1.5],
        # rotation_range=350,
        # rotation_range=30,
        # vertical_flip=True,
        # zoom_range=0.3,
        # width_shift_range=0.2, height_shift_range=0.2
        validation_split=0.2
    )
    return image_data_generator

def get_data():
    image_data_generator = create_image_data_generator_with_split()
    print(path_as_string(split_dataset_train_path))

    train =  image_data_generator.flow_from_directory(
        path_as_string(split_dataset_train_path),
        # save_to_dir=path_as_string(augmentation_results_path) + "/train",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation,
        shuffle=True,
        # seed=42,
        class_mode='categorical',
        subset='training',
    )

    val = image_data_generator.flow_from_directory(
        path_as_string(split_dataset_train_path),
        #save_to_dir=path_as_string(augmentation_results_path) + "/validation",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation,
        subset='validation',
        shuffle=True,
        # seed=42,
        class_mode='categorical'
    )

    return train, val

def get_train_data():
    image_data_generator = create_image_data_generator()
    print(path_as_string(split_dataset_train_path))

    return image_data_generator.flow_from_directory(
        path_as_string(split_dataset_train_path),
        # save_to_dir=path_as_string(augmentation_results_path) + "/train",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation,
    )

def get_validation_data():
    image_data_generator = create_image_data_generator()
    print(path_as_string(split_dataset_validation_path))
    return image_data_generator.flow_from_directory(
        path_as_string(split_dataset_validation_path),
        #save_to_dir=path_as_string(augmentation_results_path) + "/validation",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation,
    )

def load_pre_trained_model():
    # Load pre-trained model
    # base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    # base_model = tf.keras.applications.efficientnet.EfficientNetB7(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    # base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    return base_model

def create_model(base_model):
    base_model.trainable = True

    frozen_num_layer = -40
    for layer in base_model.layers[:frozen_num_layer]:
        layer.trainable = False

    input_anchor = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="anchor_input")
    input_positive = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="positive_input")
    input_negative = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="negative_input")

    embedding_anchor = base_model(input_anchor)
    embedding_positive = base_model(input_positive)
    embedding_negative = base_model(input_negative)

  
    output = tf.stack([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=output)

    print(model.summary())
    return model






def create_callbacks_for_fit_model():
    logdir = path_as_string(log_path) + '/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5)
    checkpoint = ModelCheckpoint(filepath=path_as_string(model_path) + "/" + model_name,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    callbacks = [checkpoint, earlystopping, reduce_lr, tensorboard_callback]
    return callbacks
#---------------------------------------------------
def triplet_loss(y_true, y_pred, margin=0.5):
    
    anchor = y_pred[:, 0, :]  
    positive = y_pred[:, 1, :]  
    negative = y_pred[:, 2, :] 
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1) 
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))  
    return loss


def triplet_generator(data_iterator):
    while True:
        x, y = next(data_iterator) 
        anchors, positives, negatives = [], [], []
        for i in range(len(x)):
            anchors.append(x[i])
            positives.append(x[(i + 1) % len(x)])  
            negatives.append(x[(i + 2) % len(x)])  
     
        yield [np.array(anchors), np.array(positives), np.array(negatives)], np.zeros((len(anchors), 1))




def compile_model(model):
    learning_rate = 1e-3
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=triplet_loss, metrics=[triplet_accuracy])
    return model




def triplet_accuracy(y_true, y_pred):
    anchor = y_pred[:, 0, :]
    positive = y_pred[:, 1, :]
    negative = y_pred[:, 2, :]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
   
    return tf.reduce_mean(tf.cast(pos_dist < neg_dist, tf.float32))


#---------------------------------------------------
def train():
    train_data_iterator, validation_data_iterator = get_data()

    base_model = load_pre_trained_model()
    model = create_model(base_model)
    model = compile_model(model)
    
    triplet_train_gen = triplet_generator(train_data_iterator)
    triplet_val_gen = triplet_generator(validation_data_iterator)

    steps_per_epoch = train_data_iterator.n // batch_size
    validation_steps = validation_data_iterator.n // batch_size
    epochs = 200

    hist = model.fit(
        triplet_train_gen,
        validation_data=triplet_val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=create_callbacks_for_fit_model(),
        verbose=1
    )
    
    final_train_loss = hist.history['loss'][-1]
    final_val_loss = hist.history['val_loss'][-1]
    final_train_acc = hist.history['triplet_accuracy'][-1]  
    final_val_acc = hist.history['val_triplet_accuracy'][-1] 

    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Training Triplet Accuracy: {final_train_acc:.4f}")  
    print(f"Final Validation Triplet Accuracy: {final_val_acc:.4f}")  


    print("Model zapisany")
    plot_model_histogram(hist, epochs)
    print("HISTOGRAM ZAPISANY")




physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
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
        
        train()
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("bez gpu")
    train()
