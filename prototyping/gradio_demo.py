import gradio as gr
import numpy as np
import tensorflow as tf

my_model = tf.keras.models.load_model('exercise4.keras')

def image_classifier(inp):
    prediction = my_model.predict(np.array([inp]))[0].tolist()
    class_names = ["bird", "cat", "deer", "dog"]
    print({ k:v for (k,v) in zip(class_names, prediction)} )
    return { k:v for (k,v) in zip(class_names, prediction)} 


demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()