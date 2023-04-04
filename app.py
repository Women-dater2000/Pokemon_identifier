from fastai.vision.all import *
import gradio as gr
from pathlib import Path
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = load_learner(r'C:\Users\admin\Guess_The_Pokemon\pokemon.pkl')

def predict(img):
    name,_,prob = model.predict(img)
    pred = float(max(prob))
    return { name : pred}

inputs = gr.inputs.Image(shape=(512,512))
outputs = gr.outputs.Label(num_top_classes=1)
gr.Interface(fn=predict , inputs = inputs, outputs =outputs).launch()