import sys
from os import path
from scipy import dtype
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from knn import *
from inference import *

import cv2
import time
from datetime import timedelta
import random
import warnings
import tensorflow as tf
import numpy as np

from configs import *
import pyarrow.parquet as pq
import pyarrow as pa

from tensor_rt import *

warnings.filterwarnings("ignore")
random.seed(42)

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print(cm)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp = disp.plot(cmap='Blues')
    plt.show()
    plt.savefig(path_as_string(results_path)+"/confusion_matrix.png")

def load_model():
    # Select model
    model = tf.keras.models.load_model(path_as_string(model_path) + '/' + model_name)
    # model = EfficientNetB1(weights="imagenet", include_top=False, pooling="avg")
    # model = MobileNet(weights="imagenet", include_top=False, pooling="avg")
    model_without_output = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-1].input)
    print(model_without_output.summary())
    return model_without_output

def get_images_paths(image_folders):
    out_list_ds = []

    for image_folder in image_folders:
        list_ds = list_files(image_folder)
        for element_ds in list_ds:
            out_list_ds.append(element_ds.numpy().decode('utf-8'))
    return out_list_ds

def get_embedings_paths():
    path_polar = path_as_string(split_dataset_polar_test_path)
    image_folders=[path_polar + "/masked/train", path_polar + "/masked/val"]
    out_list_ds = get_images_paths(image_folders)
    
    return out_list_ds

def get_image_dataset_from_directory(folder_path):
    return tf.keras.utils.image_dataset_from_directory(path_as_string(folder_path),
        image_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def get_image_from_directory(image_path):
    picture = tf.keras.preprocessing.image.load_img(path=image_path, 
        grayscale=False, color_mode='rgb', 
        target_size=img_input_size, interpolation=image_interpolation
    )
    #print(tf.io.read_file(image_path))
    return picture

def get_test_images():
    path_polar = path_as_string(split_dataset_polar_test_path)
    test_images_paths = get_images_paths([path_polar+"/masked/test"])
    random.shuffle(test_images_paths)
    
    return test_images_paths

def create_image_feature(image_path, prediction):
    parts = image_path.split("/")
    image_name = parts[-1].split(".")[0]

    return {"image_path": image_path, 
            "image_name": image_name, 
            "class": image_name[0:2],
            "embedding":np.array(prediction[0])}

def create_image_datarecord(model, image_path):
    picture = get_image_from_directory(image_path)
        #ValueError: Input 0 is incompatible with layer model: expected shape=(None, 456, 456, 3), found shape=(None, 456, 3)
    picture = tf.keras.preprocessing.image.img_to_array(picture)
    picture = np.array([picture])  # Convert single image to a batch.
    prediction = model.predict(picture, batch_size = batch_size)
    
    return create_image_feature(image_path, prediction)

def create_images_dataset(model, images_paths):
    elements = []
    for image_path in images_paths:
        elements.append(create_image_datarecord(model, image_path))

    elements.sort(key=lambda embeding: embeding.get('image_name'))

    return elements

def main():
    print("load_model")
    model = load_model()
    # model = improve_by_tensor_rt()

    images_paths = get_embedings_paths()
    image_dataset = create_images_dataset(model, images_paths)

    test_paths = get_test_images()
    print(len(test_paths))
    acc = 0
    n_acc = 0
    # conf matrix
    y_true = []
    y_pred = []
    process_time = []

    embeddings = np.stack([ element["embedding"] for element in image_dataset ])
    index = build_index(embeddings)

    # print("embeddings")
    # print(embeddings.size)
    # print(embeddings.dtype)
    # print(embeddings.shape)

    for test_path in test_paths:
        print(test_path)
        real_time = 0

        #print(test_image)
        # Load query and emb
        start = time.time()
        test_image = create_image_datarecord(model, test_path)

        # Build index
        embedding = test_image["embedding"]
        # print("embedding")
        # print(embedding.size)
        # print(embedding.dtype)
        # print(embedding.shape)

        results = search_results(index, embedding, image_dataset)
        end = time.time()
        real_time = (end - start)
        process_time.append(real_time)
        # Display results
        display_results(test_image, results)
        
        # Calculate accuracy
        top_knn = results[0]
        element = top_knn['element']
        predicted_class = element['class']
        real_class = test_image['class']
        # print(predicted_class)
        # print(real_class)

        y_true.append(real_class)
        y_pred.append(predicted_class)
        
        n_acc+=1
        if real_class == predicted_class:
            acc += 1
        print(f"processing: {test_path}, mean time: {real_time}, mean acc: {acc / n_acc}")
    
    # meantime_insecond = timedelta(seconds=round(process_time))

    # meantime_insecond_2= time.strftime("%H:%M:%S.%f",time.gmtime(process_time))
    # meantime_insecond_n_acc = time.strftime("%H:%M:%S.%f",time.gmtime(process_time/ n_acc))

    print(f"\n Max time in second: {np.max(process_time)}")
    print(f"\n Min time in second: {np.min(process_time)}")
    print(f"\n Min time in second: {np.mean(process_time)}")
    print(f"\n std time in second: {np.std(process_time)}")

    # print(f"\nMean time in second: {meantime_insecond / n_acc}\nTotal time: {meantime_insecond} \n")

    # print(f"\nMean accuracy: {acc / n_acc}\nMean time: {process_time / n_acc}\nTotal time: {process_time}\n")

    with open(path_as_string(log_path) +'/emb_fast01.txt', 'a') as file:
        file.write(str("Mean accuracy: " + str(acc / n_acc) + ", mean time: " + str(np.mean(process_time)) + ", std time: " + str(np.std(process_time)) + '\n'))

    display_confusion_matrix(y_true, y_pred)

#https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
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
        print("main")

        main()
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)