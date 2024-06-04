from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

KERAS_OUTPUT_PATH = "./model_q_aware2.keras"
TFLITE_OUTPUT_PATH = "./model_q_aware2.tflite"

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
train_images = np.load('./train_images.npy')
train_labels = np.load('./train_labels.npy')
test_images = np.load('./test_images.npy')
test_labels = np.load('./test_labels.npy')
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