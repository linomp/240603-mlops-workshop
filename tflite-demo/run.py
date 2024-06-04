
import gradio as gr
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="./model.tflite")

def image_classifier(inp):
    # prediction = my_model.predict(np.array([inp]))[0].tolist()
    # class_names = ["bird", "cat", "deer", "dog"]
    # print({ k:v for (k,v) in zip(class_names, prediction)} )
    # return { k:v for (k,v) in zip(class_names, prediction)} 
    return {"cat": 0.3, "dog":0.98}

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()