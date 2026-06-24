import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from torchvision import models

# Sustituir por la ruta local del modelo ResNet en precisión FP32 previamente entrenado
PTH_PATH = "RUTA_AL_MODELO/modelo_MakeUp_FP32_ResNet.pth"
# Sustituir por la ruta local donde se guardará el modelo en formato ONNX 
ONNX_PATH = "RUTA_DE_GUARDADO_ONNX/modelo_MakeUp_FP32_ResNet.onnx"
# Sustituir por la ruta local donde se guardará el modelo en formato TensorFlow SavedModel
TF_PATH = "RUTA_DE_GUARDADO_TF/modelo_MakeUp_FP32_ResNet_tf"
# Sustituir por la ruta local donde se guardará el modelo en formato TensorFlow Lite 
TFLITE_PATH = "RUTA_DE_GUARDADO_TFLITE/modelo_MakeUp_FP32_ResNet.tflite"

model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

checkpoint = torch.load(PTH_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "modelo_state_dict" in checkpoint:
    checkpoint = checkpoint["modelo_state_dict"]

model.load_state_dict(checkpoint)
model.eval()

dummy_input = torch.rand(1, 3, 224, 224, dtype=torch.float32)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

dummy_input = (dummy_input - mean) / std

with torch.no_grad():
    pytorch_output = model(dummy_input).cpu().numpy()

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True
)

print(f"Modelo exportado a ONNX: {ONNX_PATH}")

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

tf_rep = prepare(
    onnx_model,
    strict=True,
    auto_cast=True
)

tf_rep.export_graph(TF_PATH)

print(f"Modelo convertido a TensorFlow SavedModel: {TF_PATH}")

converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

converter.optimizations = []

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Modelo TFLite FP32 generado: {TFLITE_PATH}")