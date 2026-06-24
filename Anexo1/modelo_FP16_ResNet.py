import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local del modelo ResNet FP32 previamente entrenado
FP32_PATH = "RUTA MODELO FP32"
# Sustituir por la ruta local donde se vaya a guardar el modelo en precisión FP16
FP16_PATH = "RUTA DE GUARDADO"

batch_size = 64
umbral = 0.5


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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

    test_full = datasets.ImageFolder(
        root=path,
        transform=transform_val
    )

    test_dataset = Subset(
        test_full,
        test_idx
    )

    return test_dataset


def crear_resnet():

    model = models.resnet18( weights=None)
    model.fc = nn.Linear(model.fc.in_features,1)

    return model


def cargar_modelo(path, device):

    model = crear_resnet()

    checkpoint = torch.load(
        path,
        map_location=device
    )

    if isinstance(checkpoint, nn.Module):

        model.load_state_dict(
            checkpoint.state_dict()
        )

    elif (
        isinstance(checkpoint, dict)
        and "modelo_state_dict" in checkpoint
    ):

        model.load_state_dict(checkpoint["modelo_state_dict"])

    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def calcular_metricas(
    modelo,
    dataloader,
    device,
    fp16=False,
    umbral=0.5
):

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

            probs = torch.sigmoid(
                outputs.float()
            ).squeeze(1)

            predicted = (
                probs >= umbral
            ).int()

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            tp += ((predicted == 1)&(labels == 1)).sum().item()

            fp += ((predicted == 1)&(labels == 0)).sum().item()

            fn += ((predicted == 0)&(labels == 1)).sum().item()

            tn += ((predicted == 0)&(labels == 0)).sum().item()

    accuracy = (100 * correct / total)

    precision = 0 if tp + fp == 0 else 100 * tp / (tp + fp)

    recall = 0 if tp + fn == 0 else 100 * tp / (tp + fn)

    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return (accuracy,precision,recall,f1,(tn,fp,fn,tp))


def get_size_model(path):
    return (os.path.getsize(path)/(1024*1024))

if __name__ == "__main__":

    test_ds = crear_test_dataset(DATASET_PATH)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model_fp32 = cargar_modelo(FP32_PATH,device)

    print("\nMÉTRICAS FP32")

    acc32, prec32, rec32, f132, conf32 = (
        calcular_metricas(
            model_fp32,
            test_loader,
            device,
            fp16=False,
            umbral=umbral
        )
    )

    print(f"ACCURACY: {acc32:.2f}")
    print(f"PRECISION: {prec32:.2f}")
    print(f"RECALL: {rec32:.2f}")
    print(f"F1 SCORE: {f132:.2f}")
    print(conf32)

    model_fp16 = crear_resnet()
    model_fp16.load_state_dict(model_fp32.state_dict())
    model_fp16 = (model_fp16.to(device).half())
    model_fp16.eval()

    torch.save({"modelo_state_dict":model_fp16.state_dict()},
        FP16_PATH)

    print("\nMÉTRICAS FP16")

    acc16, prec16, rec16, f116, conf16 = (
        calcular_metricas(
            model_fp16,
            test_loader,
            device,
            fp16=True,
            umbral=umbral
        )
    )

    print(f"ACCURACY: {acc16:.2f}")
    print(f"PRECISION: {prec16:.2f}")
    print(f"RECALL: {rec16:.2f}")
    print(f"F1 SCORE: {f116:.2f}")
    print(conf16)

    size_fp32 = get_size_model(FP32_PATH)
    size_fp16 = get_size_model(FP16_PATH)

    reduction = (100*( size_fp32-size_fp16)/size_fp32)

    print("\nComparación F1")
    print(f"F1 FP32 : {f132:.4f}")
    print(f"F1 FP16 : {f116:.4f}")

    print(f"Diferencia: {f132-f116:.4f}")

    print("\nTamaño Modelo")
    print(f"FP32 : {size_fp32:.2f} MB")
    print(f"FP16 : {size_fp16:.2f} MB")

    print(f"Reducción aproximada: "f"{reduction:.2f}%")
