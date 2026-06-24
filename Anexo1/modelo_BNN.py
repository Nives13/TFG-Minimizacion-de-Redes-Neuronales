import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local donde se vaya a guardar el modelo
path_save = "RUTA DE GUARDADO"

class_names = ["No Maquillado", "Maquillado"]
batch_size = 64
Epocas = 30


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

    print("Clases detectadas:", base_dataset.class_to_idx)
    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Validation: {len(val_dataset)} imágenes")
    print(f"Test: {len(test_dataset)} imágenes")

    return train_dataset, val_dataset, test_dataset


def mostrar_imagenesDef(dataset, title="Imágenes", num_images=4, class_names=None):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        img, label = dataset[i]

        image = img.clone().cpu() * 0.5 + 0.5
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        axes[i].imshow(image)
        axes[i].set_title(class_names[label] if class_names else f"Clase {label}")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.clamp(-1, 1)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


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


def float_to_bin(weights):
    w = weights.detach().cpu().numpy()
    bin_w = (w > 0).astype(np.uint8)
    packed = np.packbits(bin_w.flatten())
    return torch.from_numpy(packed)


def bin_to_float(bin_w, shape):
    b = bin_w.cpu().numpy()
    unpacked = np.unpackbits(b)[:np.prod(shape)]
    unpacked = unpacked.astype(np.float32) * 2 - 1
    return torch.from_numpy(unpacked).view(shape)


class BinaryClassifier_BNN(nn.Module):
    def __init__(self):
        super(BinaryClassifier_BNN, self).__init__()

        # Entrada FP32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Capas medias binarias
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (BinaryConv2d, BinaryLinear, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def summary(self, input_size, batch_size=1, device="cuda"):
        summary(self, input_size=input_size, batch_size=batch_size, device=device)

    def save_binary(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_dict = self.state_dict()
        bin_dict = {}

        binary_weight_names = {
            "conv2.weight",
            "conv3.weight",
            "conv4.weight",
            "conv5.weight",
            "fc1.weight"
        }

        for k, v in state_dict.items():
            if k in binary_weight_names:
                bin_dict[k] = {
                    "packed": float_to_bin(v.sign()),
                    "shape": tuple(v.shape),
                    "binary": True
                }
            else:
                bin_dict[k] = {
                    "tensor": v.cpu(),
                    "binary": False
                }

        torch.save(bin_dict, path)
        print(f"Modelo BNN guardado en formato binario en: {path}")

    def load_binary(self, path, map_location="cpu"):
        bin_dict = torch.load(path, map_location=map_location)
        state_dict = self.state_dict()

        restored = {}

        for k, item in bin_dict.items():
            if item["binary"]:
                restored[k] = bin_to_float(item["packed"], item["shape"])
            else:
                restored[k] = item["tensor"]

        self.load_state_dict(restored)
        print(f"Modelo BNN cargado desde: {path}")


def calcular_metricas_binariasDef(modelo, dataloader, device, umbral=0.5):
    modelo.eval()

    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = modelo(images)
            predicted = (torch.sigmoid(outputs).squeeze(1) >= umbral).int()

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


def guardar_historial(directorio, epoca, historial):
    os.makedirs(directorio, exist_ok=True)

    datos_a_guardar = {
        "epoca": epoca,
        "historial": historial
    }

    with open(os.path.join(directorio, "historial.json"), "w") as f:
        json.dump(datos_a_guardar, f)


def guardar_checkpoint(directorio, epoca, modelo, optimizador):
    os.makedirs(directorio, exist_ok=True)

    torch.save(
        {
            "modelo_state_dict": modelo.state_dict(),
            "optimizador_state_dict": optimizador.state_dict(),
            "epoca": epoca
        },
        os.path.join(directorio, "checkpoint.pth")
    )


def Entrenar(model, train_loader, val_loader, criterion, optimizer, device,
             epochs=20, scheduler=None, run="./.run/"):

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

        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f"Train: Epoch {epoch}/{epochs}"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            
            for m in model.modules():
                if isinstance(m, (BinaryConv2d, BinaryLinear)):
                    m.weight.data.clamp_(-1, 1)

            train_loss_sum += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_acc = train_correct / len(train_loader.dataset)

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, total=len(val_loader), desc="Validation"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_acc = val_correct / len(val_loader.dataset)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        guardar_historial(run, epoch, history)
        guardar_checkpoint(run, epoch, model, optimizer)

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Train Acc = {avg_train_acc:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, "
            f"Val Acc = {avg_val_acc:.4f}"
        )

    return history


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":

    train_ds, validation_ds, test_ds = create_datasetsDef(
        DATASET_PATH,
        transform_train=transform_train,
        transform_val=transform_val
    )

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

    mostrar_imagenesDef(
        train_ds,
        title="Imágenes de Entrenamiento",
        num_images=4,
        class_names=class_names
    )

    mostrar_imagenesDef(
        validation_ds,
        title="Imágenes de Validación",
        num_images=4,
        class_names=class_names
    )

    model = BinaryClassifier_BNN().to(device)
    model.summary((3, 224, 224), batch_size)

    labels_train = [train_ds.dataset.targets[i] for i in train_ds.indices]
    num_pos = sum(labels_train)
    num_neg = len(labels_train) - num_pos

    print(f"Train Clase 0: {num_neg}")
    print(f"Train Clase 1: {num_pos}")

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0015,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    print(f"LOSS: {loss_function.__class__.__name__}")
    print(f"OPTIMIZER: {optimizer.__class__.__name__}")

    history = Entrenar(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=Epocas,
        scheduler=scheduler
    )

    model.save_binary(path_save)

    plot_training_history(history)

    print("METRICAS TEST")

    acc, precision, recall, f1, matriz_confusion = calcular_metricas_binariasDef(
        model,
        test_loader,
        device,
        umbral=0.55
    )

    print(f"ACCURACY:   {acc:.2f}")
    print(f"PRECISION:  {precision:.2f}")
    print(f"RECALL:     {recall:.2f}")
    print(f"F1 SCORE:   {f1:.2f}")
    print("CONF MATRIX:")
    print(matriz_confusion)

    size_bnn = os.path.getsize(path_save) / (1024 * 1024)
    print(f"Tamaño BNN guardado con pesos binarios empaquetados: {size_bnn:.2f} MB")

    model_loaded = BinaryClassifier_BNN().to(device)
    model_loaded.load_binary(path_save, map_location=device)
    model_loaded.eval()