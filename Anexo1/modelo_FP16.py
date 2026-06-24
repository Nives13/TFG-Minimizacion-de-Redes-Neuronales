import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local del modelo FP32 previamente entrenado 
FP32_PATH = "RUTA MODELO FP32"
# Sustituir por la ruta local donde se vaya a guardar el modelo en precisión FP16
FP16_PATH = "RUTA DE GUARDADO"

batch_size = 64
umbral = 0.55


class BinaryClassifier_021(nn.Module):
    def __init__(self):
        super(BinaryClassifier_021, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
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
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def crear_test_dataset(path):
    base_dataset = datasets.ImageFolder(root=path)
    targets = base_dataset.targets

    train_idx, temp_idx = train_test_split(
        range(len(targets)),
        test_size=0.30,
        stratify=targets,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=[targets[i] for i in temp_idx],
        random_state=42
    )

    test_full = datasets.ImageFolder(root=path, transform=transform_val)
    test_dataset = Subset(test_full, test_idx)

    return test_dataset

def calcular_metricas(modelo, dataloader, device, fp16=False, umbral=0.5):
    modelo.eval()

    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if fp16:
                images = images.half()

            outputs = modelo(images)

            probs = torch.sigmoid(outputs.float()).squeeze(1)
            predicted = (probs >= umbral).int()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()

    accuracy = 100 * correct / total
    precision = 0 if tp + fp == 0 else 100 * tp / (tp + fp)
    recall = 0 if tp + fn == 0 else 100 * tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1, (tn, fp, fn, tp)


def get_size_model(path):
    return os.path.getsize(path) / (1024 * 1024)


if __name__ == "__main__":

    test_ds = crear_test_dataset(DATASET_PATH)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    
    model_fp32 = BinaryClassifier_021().to(device)

    checkpoint = torch.load(FP32_PATH, map_location=device)

    if isinstance(checkpoint, BinaryClassifier_021):
        model_fp32 = checkpoint.to(device)

    elif isinstance(checkpoint, dict) and "modelo_state_dict" in checkpoint:
        model_fp32.load_state_dict(checkpoint["modelo_state_dict"])

    elif isinstance(checkpoint, dict):
        model_fp32.load_state_dict(checkpoint)

    else:
        raise TypeError(f"Formato de checkpoint no soportado: {type(checkpoint)}")

    model_fp32.eval()

    print(" MÉTRICAS FP32 ")

    acc32, prec32, rec32, f132, conf32 = calcular_metricas(
        model_fp32,
        test_loader,
        device,
        fp16=False,
        umbral=umbral
    )

    print(f"ACCURACY:   {acc32:.2f}")
    print(f"PRECISION:  {prec32:.2f}")
    print(f"RECALL:     {rec32:.2f}")
    print(f"F1 SCORE:   {f132:.2f}")
    print("CONF MATRIX:")
    print(conf32)

    model_fp16 = BinaryClassifier_021().to(device)
    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16.half()
    model_fp16.eval()

    torch.save(model_fp16.state_dict(), FP16_PATH)

    print("MÉTRICAS FP16")

    acc16, prec16, rec16, f116, conf16 = calcular_metricas(
        model_fp16,
        test_loader,
        device,
        fp16=True,
        umbral=umbral
    )

    print(f"ACCURACY:   {acc16:.2f}")
    print(f"PRECISION:  {prec16:.2f}")
    print(f"RECALL:     {rec16:.2f}")
    print(f"F1 SCORE:   {f116:.2f}")
    print("CONF MATRIX:")
    print(conf16)

    size_fp32 = get_size_model(FP32_PATH)
    size_fp16 = get_size_model(FP16_PATH)
    reduction = 100 * (size_fp32 - size_fp16) / size_fp32

    print("\nComparación F1")
    print(f"F1 FP32 : {f132:.4f}")
    print(f"F1 FP16 : {f116:.4f}")
    print(f"Diferencia: {f132 - f116:.4f}")

    print("\nTamaño Modelo")
    print(f"FP32 : {size_fp32:.2f} MB")
    print(f"FP16 : {size_fp16:.2f} MB")
    print(f"Reducción aproximada: {reduction:.2f}%")