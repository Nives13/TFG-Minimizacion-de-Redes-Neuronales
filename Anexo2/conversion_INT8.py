import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from PIL import Image

class BinaryClassifier_021(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Sustituir por la ruta local del modelo en precisión FP32 previamente entrenado
PTH_PATH = "RUTA_AL_MODELO/modelo_MakeUp_FP32.pth"

# Sustituir por la ruta local donde se guardará el modelo intermedio en formato ONNX
ONNX_PATH = "RUTA_DE_GUARDADO_ONNX/modelo_MakeUp_INT8.onnx"

# Sustituir por la ruta local donde se guardará el modelo intermedio en formato TensorFlow SavedModel
TF_PATH = "RUTA_DE_GUARDADO_TF/modelo_MakeUp_INT8_tf"

# Sustituir por la ruta local donde se guardará el modelo final convertido a TensorFlow Lite 
TFLITE_PATH = "RUTA_DE_GUARDADO_TFLITE/modelo_MakeUp_INT8.tflite"

# Sustituir por la ruta local del conjunto de imágenes representativas usado para calibrar la cuantización INT8
REP_DATASET_PATH = "RUTA_AL_DATASET_REPRESENTATIVO"

model = BinaryClassifier_021()

checkpoint = torch.load(PTH_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "modelo_state_dict" in checkpoint:
    checkpoint = checkpoint["modelo_state_dict"]

model.load_state_dict(checkpoint)
model.eval()

dummy_input = torch.rand(1, 3, 224, 224, dtype=torch.float32)
dummy_input = (dummy_input - 0.5) / 0.5

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

def preprocess_image_nchw(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)

    image = np.array(image).astype(np.float32) / 255.0

    image = (image - 0.5) / 0.5

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


def representative_dataset():
    image_paths = []

    for root, dirs, files in os.walk(REP_DATASET_PATH):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                image_paths.append(os.path.join(root, file))

    image_paths = image_paths[:300]

    if len(image_paths) == 0:
        raise RuntimeError("No hay imágenes en el dataset")

    for image_path in image_paths:
        img = preprocess_image_nchw(image_path)
        yield [img]

converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Modelo TFLite FULL INT8 generado: {TFLITE_PATH}")