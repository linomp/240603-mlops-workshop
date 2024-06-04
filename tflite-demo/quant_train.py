from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

KERAS_OUTPUT_PATH = "./model_q_aware.keras"
TFLITE_OUTPUT_PATH = "./model_q_aware.tflite"

model = tf.keras.models.load_model('model.keras')

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.compat.v1.losses.sparse_softmax_cross_entropy,
              metrics=['accuracy'])

q_aware_model.summary()

# Retraining block
(images, labels), _ = keras.datasets.cifar10.load_data()

# there are 10 classes of images
all_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# choose four classes (feel free to change this!)
class_names = ["bird", "cat", "deer", "dog"]
print("Class names:", class_names)

# only keep images of these classes
class_indexes = [all_classes.index(c) for c in class_names]
to_keep = np.array([l in class_indexes for l in labels])
images = images[to_keep]
labels = labels[to_keep]

# change indexes from 10 to 2 classes
labels = np.array([class_indexes.index(l) for l in labels])

# normalize pixels between 0 and 1
images = images / 255.0

# split into train and test set
split = round(len(images) * 0.8)
train_images = images[:split]
train_labels = labels[:split]
test_images = images[split:]
test_labels = labels[split:]
print("Number of train images:", len(train_images))
print("Number of test images:", len(test_images))

history = q_aware_model.fit(train_images, train_labels,
                  batch_size=500, epochs=10,
                    validation_data=(test_images, test_labels))

q_aware_model.save(KERAS_OUTPUT_PATH)


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

with open(TFLITE_OUTPUT_PATH, 'wb') as f:
  f.write(quantized_tflite_model)