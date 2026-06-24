import os
import torch
import torch.nn as nn
import torch.quantization as tq
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import datasets
from torchvision.models.quantization import resnet18
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split



device = "cpu"  
print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local del modelo ResNet FP32 previamente entrenado
FP32_PATH = "RUTA MODELO FP32"
# Sustituir por la ruta local donde se vaya a guardar el modelo en precisión INT8 PTQ
INT8_PATH = "RUTA DE GUARDADO"

batch_size = 64
umbral = 0.55

os.makedirs("./saveDoc", exist_ok=True)


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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

    print("Número de clases:", base_dataset.class_to_idx)
    print(f"Validation: {len(val_dataset)} imágenes")
    print(f"Test: {len(test_dataset)} imágenes")

    return val_dataset, test_dataset


def crear_resnet18_binaria():
    model = resnet18(weights=None, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def cargar_modelo_fp32(path):
    model = crear_resnet18_binaria()

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    elif isinstance(checkpoint, dict) and "modelo_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["modelo_state_dict"])

    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)

    else:
        raise TypeError(f"Formato de checkpoint no soportado: {type(checkpoint)}")

    model.eval()
    return model


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


def get_size_model(path):
    return os.path.getsize(path) / (1024 * 1024)


def cuantizar_resnet18_int8_desde_fp32(model_fp32, val_loader, int8_path):

    torch.backends.quantized.engine = "fbgemm"

    model_int8 = crear_resnet18_binaria()
    model_int8.load_state_dict(model_fp32.state_dict())
    model_int8.eval()

    model_int8.fuse_model()

    model_int8.qconfig = tq.get_default_qconfig("fbgemm")

    tq.prepare(model_int8, inplace=True)

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Calibración INT8"):
            images = images.to("cpu")
            model_int8(images)

    tq.convert(model_int8, inplace=True)

    torch.save(
        {"modelo_state_dict": model_int8.state_dict()},
        int8_path
    )

    print(f"Modelo INT8 guardado en: {int8_path}")

    return model_int8


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
    model_fp32.to("cpu")
    model_fp32.eval()

    print("\nMÉTRICAS TEST FP32")

    acc, precision, recall, f1, matriz_confusion = calcular_metricas_binariasDef(
        model_fp32,
        test_loader,
        "cpu",
        umbral=umbral
    )

    print(f"ACCURACY:   {acc:.2f}")
    print(f"PRECISION:  {precision:.2f}")
    print(f"RECALL:     {recall:.2f}")
    print(f"F1 SCORE:   {f1:.2f}")
    print("CONF MATRIX:")
    print(matriz_confusion)

    model_int8 = cuantizar_resnet18_int8_desde_fp32(
        model_fp32=model_fp32,
        val_loader=val_loader,
        int8_path=INT8_PATH
    )

    print("\nMÉTRICAS TEST INT8 PTQ")

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

    reduction = 100 * (size_fp32 - size_int8) / size_fp32

    print("\nComparación F1")
    print(f"F1 FP32 : {f1:.4f}")
    print(f"F1 INT8 : {f1_q:.4f}")
    print(f"Diferencia: {f1 - f1_q:.4f}")

    print("\nTamaño Modelo")
    print(f"FP32 : {size_fp32:.2f} MB")
    print(f"INT8 : {size_int8:.2f} MB")
    print(f"Reducción aproximada: {reduction:.2f}%")