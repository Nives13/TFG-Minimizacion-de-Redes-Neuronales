import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = "cpu"  
print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local del modelo FP32 previamente entrenado 
FP32_PATH = "RUTA MODELO FP32"
# Sustituir por la ruta local donde se vaya a guardar el modelo en precisión INT8 PTQ
INT8_PATH = "RUTA DE GUARDADO"

class_names = ["No Maquillado", "Maquillado"]
batch_size = 64
umbral = 0.55


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def create_datasetsDef(path, transform_val=None):
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

    val_full = datasets.ImageFolder(root=path, transform=transform_val)
    test_full = datasets.ImageFolder(root=path, transform=transform_val)

    val_dataset = Subset(val_full, val_idx)
    test_dataset = Subset(test_full, test_idx)

    print("Clases detectadas:", base_dataset.class_to_idx)
    print(f"Validation: {len(val_dataset)} imágenes")
    print(f"Test: {len(test_dataset)} imágenes")

    return val_dataset, test_dataset


class BinaryClassifier_021(nn.Module):
    def __init__(self):
        super(BinaryClassifier_021, self).__init__()

        self.quant = tq.QuantStub()

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

        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

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

        x = self.dequant(x)

        return x


def calcular_metricas_binariasDef(modelo, dataloader, device, umbral=0.5):
    modelo.eval()

    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = modelo(images)
            probs = torch.sigmoid(outputs).squeeze(1)
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

    matriz_confusion = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return accuracy, precision, recall, f1, matriz_confusion


def fusionar_modelo(model):
    model.eval()

    tq.fuse_modules(
        model,
        [
            ["conv1", "bn1"],
            ["conv2", "bn2"],
            ["conv3", "bn3"],
            ["conv4", "bn4"],
            ["conv5", "bn5"]
        ],
        inplace=True
    )

    return model


def cargar_modelo_fp32(path_fp32):
    model = BinaryClassifier_021()

    checkpoint = torch.load(path_fp32, map_location="cpu")

    if "modelo_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["modelo_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preparar_modelo_int8_desde_modelo_fp32(model_fp32, val_loader, quantized_path):
    torch.backends.quantized.engine = "fbgemm"

    model_int8 = BinaryClassifier_021()
    model_int8.load_state_dict(model_fp32.state_dict())
    model_int8.eval()

    model_int8 = fusionar_modelo(model_int8)

    model_int8.qconfig = tq.get_default_qconfig("fbgemm")

    tq.prepare(model_int8, inplace=True)

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Calibración INT8"):
            images = images.to("cpu")
            model_int8(images)

    tq.convert(model_int8, inplace=True)

    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)
    torch.save(model_int8.state_dict(), quantized_path)

    print(f"Modelo INT8 guardado en: {quantized_path}")

    return model_int8


def get_size_model(path):
    return os.path.getsize(path) / (1024 * 1024)


if __name__ == "__main__":

    val_ds, test_ds = create_datasetsDef(
        DATASET_PATH,
        transform_val=transform_val
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    model_fp32 = cargar_modelo_fp32(FP32_PATH)

    print("\nMÉTRICAS TEST FP32")

    acc, precision, recall, f1, matriz_confusion = calcular_metricas_binariasDef(
        model_fp32,
        test_loader,
        device,
        umbral=umbral
    )

    print(f"ACCURACY:   {acc:.2f}")
    print(f"PRECISION:  {precision:.2f}")
    print(f"RECALL:     {recall:.2f}")
    print(f"F1 SCORE:   {f1:.2f}")
    print("CONF MATRIX:")
    print(matriz_confusion)

    model_int8 = preparar_modelo_int8_desde_modelo_fp32(
        model_fp32=model_fp32,
        val_loader=val_loader,
        quantized_path=INT8_PATH
    )

    print("\nMÉTRICAS TEST INT8")

    acc_q, precision_q, recall_q, f1_q, matriz_q = calcular_metricas_binariasDef(
        model_int8,
        test_loader,
        "cpu",
        umbral=umbral
    )

    print(f"ACCURACY:   {acc_q:.2f}")
    print(f"PRECISION:  {precision_q:.2f}")
    print(f"RECALL:     {recall_q:.2f}")
    print(f"F1 SCORE:   {f1_q:.2f}")
    print("CONF MATRIX:")
    print(matriz_q)

    size_fp32 = get_size_model(FP32_PATH)
    size_int8 = get_size_model(INT8_PATH)

    reduction = 100 * (1 - size_int8 / size_fp32)

    print("\nTamaño Modelo")
    print(f"FP32 : {size_fp32:.2f} MB")
    print(f"INT8 : {size_int8:.2f} MB")
    print(f"Reducción aproximada: {reduction:.2f}%")

    print("\nComparación F1")
    print(f"F1 FP32 : {f1:.4f}")
    print(f"F1 INT8 : {f1_q:.4f}")
    print(f"Diferencia: {f1 - f1_q:.4f}")