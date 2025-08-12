import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
import numpy as np

from configs import *
import matplotlib.pyplot as plt
# https://towardsdatascience.com/writing-a-custom-data-augmentation-layer-in-keras-2b53e048a98
# https://www.tensorflow.org/api_docs/python/tf/image/stateless_random_brightness

#the benefit of the generator is when your data set is too big, 
# you can't put all the data to your limited memory, 
# but, with the generator you can generate one batch data each time. 
# and the ImageDataGenerator works with model.fit_generator(), model.predict_generator()

def create_image_data_generator():
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        brightness_range = (0.4, 0.9), #the range starts from zero which signifies no brightness of the image. the upper range is 1 which signifies the maximum range of the brightness.
        rotation_range=180,
        # width_shift_range=5,
        # height_shift_range=5
    )
    return image_data_generator;

def create_directory_iterator():
    image_data_generator = create_image_data_generator()
    # train_data_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    #     path_as_string(data_folder_path) + '/augmentation', image_data_generator, target_size=(img_input_size),
    #     color_mode='rgb', classes=None, class_mode='categorical',
    #     batch_size=batch_size, shuffle=True, seed=None, data_format=None, save_to_dir=path_as_string(augmentation_results_path) + "/train",
    #     save_prefix='', save_format='jpg', follow_links=False,
    #     subset=None, interpolation='nearest', dtype=None
    # )

    # return tf.keras.utils.image_dataset_from_directory(
    #     path_as_string(data_folder_path) + '/augmentation',
    #     image_size=(img_input_size), 
    #     batch_size=batch_size
    # )

    return image_data_generator.flow_from_directory(
        path_as_string(data_folder_path) + '/augmentation',
        # save_to_dir=path_as_string(augmentation_results_path) + "/train",
        target_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    )

AUTOTUNE = tf.data.AUTOTUNE
train_ds = create_directory_iterator()
# train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
#  tf.keras.layers.RandomRotation(0.2),
#   tf.keras.layers.RandomContrast(factor = (0.8, 0.8)),
tf.keras.layers.Rescaling(1./255)
])

print("train_ds.samples")



# data_list = []
# batch_index = 0

# while batch_index <= train_ds.batch_index:
#     data = train_ds.next()
#     data_list.append(data[0])
#     batch_index = batch_index + 1

# print(batch_index)

total_images = train_ds.n  
steps = total_images//batch_size 
#iterations to cover all data, so if batch is 5, it will take total_images/5  iteration 

x , y = [] , []
for i in range(steps):
    a , b = train_ds.next()
    x.extend(a) 
    y.extend(b)

print(len(train_ds))
print(len(np.array(x)))
print(len(np.array(y)))

for i in range(9):
    	# define subplot
	plt.subplot(3, 3, i + 1)
	# generate batch of images
	img, label = train_ds.next()
	# convert to unsigned integers for viewing
	#image = batch[0].astype('uint8')
	# plot raw pixel data
	plt.imshow(img[0].astype("uint8"))

plt.show()
plt.savefig(path_as_string(augmentation_results_path) + "/train/test.jpg")


# for j in range(6):
    
# 	plt.subplot(3, 2, j + 1)
	
# 	chunk = train_ds.next()

# 	sub_img = chunk[0].astype('uint8')
	
# 	plt.imshow(sub_img)

# plt.savefig(path_as_string(augmentation_results_path) + "/train/test.jpg")

# for image, _ in train_ds:
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(20):
#     ax = plt.subplot(4, 5, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0])
#     plt.axis('off')
    
# plt.savefig(path_as_string(augmentation_results_path) + "/train/test.jpg")

# print(train_ds)
# number_of_examples = len(train_ds)
# print(number_of_examples)
# print("end")
# for images, labels in train_ds.take(1):
#     print(len(images))
#     print (labels[0])

    #   for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title(class_names[labels[i]])
    #     tf.keras.utils.save_img(x = train_ds[0].numpy(), path=path_as_string(augmentation_results_path) + "/train/test.jpg", file_format="jpg")

        
# for _ in range(15):
#     img, label = directory_iterator.next()
#     print(directory_iterator.__len__())

