"""
This is meant to be run on the dev machine (requires full tensorflow installation)
"""

import tensorflow as tf

my_model = tf.keras.models.load_model('model.keras')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(my_model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


"""
This conversion script was failing due to keras version mismatch between the latest tensorflow version and what the tf lite converter expects.

Solution that worked:  

- downgrade local installation of tensorflow: pip install tensorflow==2.15.0
- run again exercise 4 jupyter notebook to generate again the model artifact, this time with the older keras version
- run again this conversion script

- source: https://stackoverflow.com/questions/78254377/keras-h5-model-to-tensorflow-lite-tflite-model-conversion
"""