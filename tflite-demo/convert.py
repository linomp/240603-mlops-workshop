"""
This is meant to be run on the dev machine (requires full tensorflow installation)
"""

import tensorflow as tf

KERAS_INPUT_PATH = "model.keras"
TFLITE_OUTPUT_PATH = "model.tflite"

my_model = tf.keras.models.load_model(KERAS_INPUT_PATH)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(my_model)
tflite_model = converter.convert()

# Save the model.
with open(TFLITE_OUTPUT_PATH, 'wb') as f:
  f.write(tflite_model)


"""
If the conversion script fails it may be due to keras version mismatch between the latest tensorflow version and the tf lite converter

Solution that worked:  

- downgrade local installation of tensorflow: pip install tensorflow==2.15.0
- generate again the model artifact, this time with the older keras version
- run again this conversion script

- source: https://stackoverflow.com/questions/78254377/keras-h5-model-to-tensorflow-lite-tflite-model-conversion
"""