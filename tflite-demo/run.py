"""
This is meant to be run on the device (requires tflite runtime)
"""

import gradio as gr
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

def image_classifier(inp):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input and do inference
    input_data = np.array([inp], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)ù
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    prediction = interpreter.get_tensor(output_details[0]['index'])[0].tolist()
    print(prediction)

    class_names = ["bird", "cat", "deer", "dog"]
    print({ k:v for (k,v) in zip(class_names, prediction)} )
    return { k:v for (k,v) in zip(class_names, prediction)} 

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()