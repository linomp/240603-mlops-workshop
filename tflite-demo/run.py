"""
This is meant to be run on the edge device (requires tflite runtime)

Source: https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
"""

import gradio as gr
import numpy as np
import tflite_runtime.interpreter as tflite

class_names = ["bird", "cat", "deer", "dog"]

interpreter = tflite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter_q = tflite.Interpreter(model_path="./model_q_aware.tflite")
interpreter_q.allocate_tensors()
input_details_q = interpreter_q.get_input_details()
output_details_q = interpreter_q.get_output_details()

def image_classifier(inp):
    # Set input, do inference and retrieve output
    input_data = np.array([inp], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0].tolist()

    # Same thing with the quantized model to compare
    interpreter_q.set_tensor(input_details_q[0]['index'], input_data)
    interpreter_q.invoke()
    prediction_q = interpreter_q.get_tensor(output_details_q[0]['index'])[0].tolist()

    print("raw model: ", { k:v for (k,v) in zip(class_names, prediction)} )
    print("q_aware model: ", { k:v for (k,v) in zip(class_names, prediction_q)} )
    return { k:v for (k,v) in zip(class_names, prediction)} 

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()