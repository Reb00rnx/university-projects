from knn import *
from inference import write_tfrecord, run_inference

import cv2
import time
import random
import warnings
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNet
from efficientnet.tfkeras import EfficientNetB0


warnings.filterwarnings("ignore")
random.seed(42)

# Select model
# model = tf.keras.models.load_model('models/final_model.hdf5')
# model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
model = MobileNet(weights="imagenet", include_top=False, pooling="avg")

path_images = "data/split_dataset/split_dataset_sequential/train"
path_tfrecords = "data/tfrecords"
path_embeddings = "data/embeddings"

write_tfrecord(image_folder=path_images, output_folder=path_tfrecords, num_shards=1)
run_inference(model, tfrecords_folder=path_tfrecords, output_folder=path_embeddings, batch_size=32)

inputPath = Path("data/split_dataset/split_dataset_sequential/test")
inputFiles = inputPath.glob("**/*.jpeg")

paths = []
for path_to_q_img in inputFiles:
    paths.append(path_to_q_img)
random.shuffle(paths)

acc = 0
n_acc = 0
mean_time = 0
# Mean square error
y_true = []
y_pred = []
for path_to_q_img in paths:
    # Load query and emb
    n_acc += 1
    image = cv2.imread(str(path_to_q_img))
    start = time.time()
    [id_to_name, name_to_id, embeddings] = knn.add_to_embeddings_no_tf(image, path_embeddings, model)

    # Build index
    index = knn.build_index(embeddings)
    p = len(embeddings) - 1
    results = knn.search(index, id_to_name, embeddings[p])
    x = results.pop('query', None)
    if x is None: results.pop(list(results)[-1], None)

    top_knn = next(iter(results))

    end = time.time()
    # Threshold
    threshold = 3

    # Display results
    # knn.display_results_seq(path_to_q_img, path_images, results, threshold)    # Comment this line when GPU is testing

    # Calculate accuracy
    q = str(path_to_q_img.stem)[0:4]
    p = str(top_knn)[0:4]
    y_true.append(q)
    y_pred.append(p)
    if abs(int(q) - int(p)) <= threshold:
        acc += 1
    mean_time += (end - start)
    print(f"processing: {n_acc}/{len(paths)}, mean time: {mean_time / n_acc}, mean acc: {acc / n_acc}")

print(f"\nMean accuracy: {acc / n_acc}\nMean time: {mean_time / n_acc}\nTotal time: {mean_time}\n")

with open('logs/seq_01.txt', 'a') as file:
    file.write(str("Mean accuracy: " + str(acc / n_acc) + ", mean time: " + str(mean_time / n_acc) + ", total time: " + str(mean_time) + '\n'))
