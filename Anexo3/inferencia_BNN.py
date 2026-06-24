import time
import math
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sustituir por la ruta local de la BNN que se utilizará para la inferencia
MODEL_PATH = "RUTA_AL_MODELO/modelo_MakeUp.pth"
# Sustituir por la ruta local de la imagen sobre la que se realizará la inferencia
IMAGE_PATH = "RUTA_A_LA_IMAGEN/imagen.jpg"

CLASS_NAMES = ["No Maquillado", "Maquillado"]
UMBRAL = 0.55
DEVICE = torch.device("cpu")


class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = input.clamp(-1, 1)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize_activation(x):
    return BinaryActivation.apply(x)


class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        real_weight = self.weight
        bin_weight = real_weight.sign()
        bin_weight = real_weight + (bin_weight - real_weight).detach()

        return F.conv2d(
            input,
            bin_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class BinaryLinear(nn.Linear):
    def forward(self, input):
        real_weight = self.weight
        bin_weight = real_weight.sign()
        bin_weight = real_weight + (bin_weight - real_weight).detach()

        return F.linear(input, bin_weight, self.bias)


def bin_to_float(bin_w, shape):
    b = bin_w.cpu().numpy()
    unpacked = np.unpackbits(b)[:np.prod(shape)]
    unpacked = unpacked.astype(np.float32) * 2 - 1
    return torch.from_numpy(unpacked).view(shape)

class BinaryClassifier_BNN(nn.Module):
    def __init__(self):
        super(BinaryClassifier_BNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = BinaryConv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = BinaryConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = BinaryConv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = BinaryConv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = BinaryLinear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(binarize_activation(self.bn2(self.conv2(x))))
        x = self.pool(binarize_activation(self.bn3(self.conv3(x))))
        x = self.pool(binarize_activation(self.bn4(self.conv4(x))))
        x = self.pool(binarize_activation(self.bn5(self.conv5(x))))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = binarize_activation(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)

        return x

    def load_binary(self, path, map_location="cpu"):
        bin_dict = torch.load(path, map_location=map_location)

        restored = {}

        for k, item in bin_dict.items():
            if item["binary"]:
                restored[k] = bin_to_float(item["packed"], item["shape"])
            else:
                restored[k] = item["tensor"]

        self.load_state_dict(restored)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = (image_np - 0.5) / 0.5

    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    tensor = torch.from_numpy(image_np).float().to(DEVICE)

    return tensor


def inferir_bnn(model, image_input, umbral=0.55):
    start_time = time.time()

    with torch.no_grad():
        output = model(image_input)
        logit = output.item()
        prob = torch.sigmoid(output).item()

    end_time = time.time()

    prediction = 1 if prob >= umbral else 0

    return CLASS_NAMES[prediction], prob, logit, end_time - start_time


if __name__ == "__main__":
    print("Imagen:", IMAGE_PATH)
    print("Modelo:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No existe la imagen: {IMAGE_PATH}")

    model = BinaryClassifier_BNN().to(DEVICE)
    model.load_binary(MODEL_PATH, map_location=DEVICE)
    model.eval()

    image_input = preprocess_image(IMAGE_PATH)

    tiempos = []

    print("\nRESULTADOS")

    for i in range(10):
        clase, probabilidad, logit, tiempo = inferir_bnn(
            model,
            image_input,
            umbral=UMBRAL
        )

        tiempos.append(tiempo)

        print(
            f"[{i+1}] "
            f"Logit: {logit:.4f} | "
            f"Prob: {probabilidad:.4f} | "
            f"Predicción: {clase} | "
            f"Tiempo: {tiempo:.4f}s"
        )

    print(f"\nTiempo medio inferencia: {sum(tiempos) / len(tiempos):.4f}s")