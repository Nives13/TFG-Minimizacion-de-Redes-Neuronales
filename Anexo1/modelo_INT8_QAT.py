import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.quantization as tq
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local del modelo FP32 previamente entrenado 
FP32_PATH = "RUTA MODELO FP32"
# Sustituir por la ruta local donde se vaya a guardar el modelo en precisión INT8 QAT
INT8_PATH = "RUTA DE GUARDADO"

batch_size = 64
umbral = 0.55
qat_epochs = 5

device = "cpu"  
print(f"Usando dispositivo: {device}")


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def create_datasetsDef(path, transform_train=None, transform_val=None):
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

    train_full = datasets.ImageFolder(root=path, transform=transform_train)
    val_full = datasets.ImageFolder(root=path, transform=transform_val)
    test_full = datasets.ImageFolder(root=path, transform=transform_val)

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)
    test_dataset = Subset(test_full, test_idx)

    print("Número de clases:", base_dataset.class_to_idx)
    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Validation: {len(val_dataset)} imágenes")
    print(f"Test: {len(test_dataset)} imágenes")

    return train_dataset, val_dataset, test_dataset


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

        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                init.zeros_(conv.bias)

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

def fuse_model(model):
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


def cargar_modelo_fp32(path):
    model = BinaryClassifier_021()

    checkpoint = torch.load(path, map_location="cpu")

    if "modelo_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["modelo_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def Entrenar_QAT(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0

        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f"QAT Epoch {epoch}/{epochs}"):
            images = images.to("cpu")
            labels = labels.float().unsqueeze(1).to("cpu")

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_acc = train_correct / len(train_loader.dataset)

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, total=len(val_loader), desc="QAT Validation"):
                images = images.to("cpu")
                labels = labels.float().unsqueeze(1).to("cpu")

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_acc = val_correct / len(val_loader.dataset)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(
            f"QAT Epoch {epoch}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Train Acc = {avg_train_acc:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Val Acc = {avg_val_acc:.4f}"
        )

    return history


def get_size_model(path):
    return os.path.getsize(path) / (1024 * 1024)


if __name__ == "__main__":

    os.makedirs(os.path.dirname(INT8_PATH), exist_ok=True)

    train_ds, validation_ds, test_ds = create_datasetsDef(
        DATASET_PATH,
        transform_train=transform_train,
        transform_val=transform_val
    )

    labels_train = [train_ds.dataset.targets[i] for i in train_ds.indices]
    num_pos = sum(labels_train)
    num_neg = len(labels_train) - num_pos

    print(f"Train Clase 0: {num_neg}")
    print(f"Train Clase 1: {num_pos}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        validation_ds,
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
        "cpu",
        umbral=umbral
    )

    print(f"ACCURACY:   {acc:.2f}")
    print(f"PRECISION:  {precision:.2f}")
    print(f"RECALL:     {recall:.2f}")
    print(f"F1 SCORE:   {f1:.2f}")
    print("CONF MATRIX:")
    print(matriz_confusion)

    torch.backends.quantized.engine = "fbgemm"

    model_qat = BinaryClassifier_021().to("cpu")
    model_qat.load_state_dict(model_fp32.state_dict())

    model_qat.eval()
    fuse_model(model_qat)

    model_qat.train()
    model_qat.qconfig = tq.get_default_qat_qconfig("fbgemm")

    tq.prepare_qat(model_qat, inplace=True)

    optimizer_qat = optim.Adam(
        model_qat.parameters(),
        lr=1e-5
    )

    loss_function_qat = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([num_neg / num_pos], dtype=torch.float32)
    )

    history_qat = Entrenar_QAT(
        model=model_qat,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_function_qat,
        optimizer=optimizer_qat,
        epochs=qat_epochs
    )

    model_qat.eval()
    model_int8 = tq.convert(model_qat, inplace=False)

    torch.save(model_int8.state_dict(), INT8_PATH)

    print(f"Modelo QAT INT8 guardado en: {INT8_PATH}")

    print("\nMÉTRICAS TEST QAT INT8")

    acc_int8, precision_int8, recall_int8, f1_int8, matriz_int8 = calcular_metricas_binariasDef(
        model_int8,
        test_loader,
        "cpu",
        umbral=umbral
    )

    print(f"ACCURACY:   {acc_int8:.2f}")
    print(f"PRECISION:  {precision_int8:.2f}")
    print(f"RECALL:     {recall_int8:.2f}")
    print(f"F1 SCORE:   {f1_int8:.2f}")
    print("CONF MATRIX:")
    print(matriz_int8)

    size_fp32 = get_size_model(FP32_PATH)
    size_int8 = get_size_model(INT8_PATH)
    reduction = 100 * (size_fp32 - size_int8) / size_fp32

    print("\nTamaño Modelo")
    print(f"FP32 : {size_fp32:.2f} MB")
    print(f"QAT INT8 : {size_int8:.2f} MB")
    print(f"Reducción aproximada: {reduction:.2f}%")

    print("\nComparación F1")
    print(f"F1 FP32 : {f1:.4f}")
    print(f"F1 QAT INT8 : {f1_int8:.4f}")
    print(f"Diferencia: {f1 - f1_int8:.4f}")