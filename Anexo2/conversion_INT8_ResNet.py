import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from PIL import Image
import os


# Sustituir por la ruta local del modelo ResNet previamente entrenado.
PTH_PATH = "RUTA_AL_MODELO/modelo_MakeUp_ResNet.pth"

# Sustituir por la ruta local donde se guardará el modelo en formato ONNX
ONNX_PATH = "RUTA_DE_GUARDADO_ONNX/modelo_MakeUp_ResNet_INT8.onnx"

# Sustituir por la ruta local donde se guardará el modelo en formato TensorFlow SavedModel
TF_PATH = "RUTA_DE_GUARDADO_TF/modelo_MakeUp_ResNet_INT8_tf"

# Sustituir por la ruta local del conjunto de imágenes representativas usado para calibrar la cuantización INT8
REP_DATASET_PATH = "RUTA_AL_DATASET_REPRESENTATIVO"

# Sustituir por la ruta local donde se guardará el modelo final convertido a TensorFlow Lite INT8
TFLITE_PATH = "RUTA_DE_GUARDADO_TFLITE/modelo_MakeUp_ResNet_FULL_INT8.tflite"

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
checkpoint = torch.load(PTH_PATH, map_location="cpu")
if "modelo_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["modelo_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

onnx_model = onnx.load(ONNX_PATH)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(TF_PATH)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)

def representative_dataset():

    image_files = os.listdir(REP_DATASET_PATH)[:100]

    for file in image_files:
        img = preprocess_image(os.path.join(REP_DATASET_PATH, file))
        yield [img]
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(" MODELO FULL INT8 GENERADO")