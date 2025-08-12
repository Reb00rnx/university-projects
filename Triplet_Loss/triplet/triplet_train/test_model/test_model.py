import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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
import pandas as pd
from configs import *
import pyarrow.parquet as pq
import pyarrow as pa
import csv

warnings.filterwarnings("ignore")
random.seed(42)

def triplet_loss(y_true, y_pred, margin=0.5):
    anchor = y_pred[:, 0, :]
    positive = y_pred[:, 1, :]
    negative = y_pred[:, 2, :]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def list_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def display_confusion_matrix(y_true, y_pred):
    labels = np.unique(np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)
    df_cm = pd.DataFrame(cm).transpose()

    df_cm.to_csv(path_as_string(results_path) + "/confusion_matrix.csv")
    df_cm.to_html(path_as_string(results_path) + "/confusion_matrix.html")

    cr = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(cr).transpose()

    df.to_csv(path_as_string(results_path) + "/classification_report.csv")
    df.to_html(path_as_string(results_path) + "/classification_report.html")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()
    plt.savefig(path_as_string(results_path) + "/confusion_matrix.png")

def triplet_accuracy(y_true, y_pred):
    anchor = y_pred[:, 0, :]
    positive = y_pred[:, 1, :]
    negative = y_pred[:, 2, :]

    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    
    return tf.reduce_mean(tf.cast(pos_dist < neg_dist, tf.float32))


def load_model():
    model = tf.keras.models.load_model(path_as_string(model_path) + '/' + model_name, custom_objects={'triplet_loss': triplet_loss, 'triplet_accuracy': triplet_accuracy})
    print(model.summary())
    return model


def get_images_paths(image_folders):
    out_list_ds = []
    for image_folder in image_folders:
        list_ds = list_files(image_folder)
        for element_ds in list_ds:
            out_list_ds.append(element_ds)
    
    print(f"Found {len(out_list_ds)} image files.")
    return out_list_ds

def get_embedings_paths():
    image_folders = [path_as_string(split_dataset_embedings_path)]
    return get_images_paths(image_folders)

def get_image_dataset_from_directory(folder_path):
    return tf.keras.utils.image_dataset_from_directory(
        path_as_string(folder_path),
        image_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def get_image_from_directory(image_path):
    picture = tf.keras.preprocessing.image.load_img(path=image_path, 
        target_size=img_input_size, interpolation=image_interpolation
    )
    return picture

def get_test_images():
    test_images_paths = get_images_paths([path_as_string(split_dataset_test_path)])
    random.shuffle(test_images_paths)
    return test_images_paths

def create_image_feature(image_path, prediction):
    parts = image_path.split("/")
    image_name = parts[-1].split(".")[0]
    label = parts[-2]
    return {"image_path": image_path, "image_name": image_name, "class": label, "embedding": np.array(prediction[0])}

def create_image_datarecord(model, image_path):
    picture = get_image_from_directory(image_path)
    picture = tf.keras.preprocessing.image.img_to_array(picture)
    picture = np.expand_dims(picture, axis=0)  
    
    dummy_input = np.zeros_like(picture) 
    
    prediction = model.predict([picture, dummy_input, dummy_input], batch_size=batch_size)
    
    return create_image_feature(image_path, prediction)

def create_images_dataset(model, images_paths):
    elements = []
    for image_path in images_paths:
        elements.append(create_image_datarecord(model, image_path))
    elements.sort(key=lambda emb: emb.get('image_name'))
    return elements

def search_results_2(index, emb, dataset):
   
    emb = np.reshape(emb, (1, -1))
    D, I = index.search(emb, 4)  
    
    results = []
    for i in range(len(I[0])):
        idx = I[0][i]
        distance = D[0][i]
        element = dataset[idx]
        results.append({'element': element, 'distance': distance})
    
    return results


def test_model():
    model = load_model()
    print("img_input_size:" + str(IMG_SIZE))

    images_paths = get_embedings_paths()
    image_dataset = create_images_dataset(model, images_paths)
    print("image_dataset: " + str(len(image_dataset)))

    test_paths = get_test_images()
    print(len(test_paths))
    
    y_true, y_pred, process_time = [], [], []

    embeddings = np.stack([element["embedding"] for element in image_dataset])
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    print(f"Shape of embeddings: {embeddings.shape}")
    index = build_index(embeddings)

    rows = []
    for test_path in test_paths:
        start = time.time()
        test_image = create_image_datarecord(model, test_path)
        embedding = test_image["embedding"]
        embedding = embedding.reshape(1, -1)
        results = search_results_2(index, embedding, image_dataset)
        end = time.time()
        process_time.append(end - start)

        top_knn = results[0]
        predicted_class = top_knn['element']['class']
        real_class = test_image['class']

        y_true.append(real_class)
        y_pred.append(predicted_class)

        row = {
            'query': real_class, 'path': test_path,
            'first': results[0]['element']['class'], 'path_first': results[0]['element']['image_path'], 'distance_1': results[0]['distance'], 
            'second': results[1]['element']['class'], 'path_second': results[1]['element']['image_path'], 'distance_2': results[1]['distance'], 
            'third': results[2]['element']['class'], 'path_third': results[2]['element']['image_path'], 'distance_3': results[2]['distance'], 
            'fourth': results[3]['element']['class'], 'path_fourth': results[3]['element']['image_path'], 'distance_4': results[3]['distance'], 
            'time': end - start
        }
        rows.append(row)


    with open(path_as_string(results_path) + '/results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Max time in second: {np.max(process_time)}")
    print(f"\n Min time in second: {np.min(process_time)}")
    print(f"\n Mean time in second: {np.mean(process_time)}")
    print(f"\n Std time in second: {np.std(process_time)}")

    display_confusion_matrix(y_true, y_pred)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        test_model()
    except RuntimeError as e:
        print(e)
else:
    print("bez gpu")
    test_model()
