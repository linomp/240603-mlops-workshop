import tensorflow as tf

my_model = tf.keras.models.load_model('model_tf215.keras')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(my_model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
