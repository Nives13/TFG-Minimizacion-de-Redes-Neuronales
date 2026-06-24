from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Sustituir por la ruta local donde se haya descargado el dataset
DATASET_PATH = "RUTA AL DATASET"
# Sustituir por la ruta local donde se vaya a guardar el modelo
path_save = "RUTA DE GUARDADO"

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

    train_dataset = torch.utils.data.Subset(train_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_full, val_idx)
    test_dataset = torch.utils.data.Subset(test_full, test_idx)

    print("Clases detectadas:", base_dataset.class_to_idx)
    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Validation: {len(val_dataset)} imágenes")
    print(f"Test: {len(test_dataset)} imágenes")

    return train_dataset, val_dataset, test_dataset


def contar_clases(subset):
    indices = subset.indices
    targets = subset.dataset.targets
    return Counter([targets[i] for i in indices])


def mostrar_imagenesDef(dataset, title="Imágenes", num_images=4, class_names=None):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_images):
        img, label = dataset[i]

        image = img.cpu() * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        axes[i].imshow(image)

        if class_names:
            axes[i].set_title(class_names[label])
        else:
            axes[i].set_title(f"Clase {label}")

        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def calcular_metricas_binariasDef(modelo, dataloader, device, umbral=0.5):
    modelo.eval()

    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = modelo(images)
            predicted = (torch.sigmoid(outputs).squeeze(1) >= umbral).int()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()

    accuracy = 100 * correct / total
    precision = 0 if true_positives + false_positives == 0 else 100 * true_positives / (true_positives + false_positives)
    recall = 0 if true_positives + false_negatives == 0 else 100 * true_positives / (true_positives + false_negatives)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    matriz_confusion = {
        "TN": true_negatives,
        "FP": false_positives,
        "FN": false_negatives,
        "TP": true_positives
    }

    return accuracy, precision, recall, f1, matriz_confusion


def guardar_historial(directorio, epoca, historial):
    os.makedirs(directorio, exist_ok=True)

    datos_a_guardar = {
        "epoca": epoca,
        "historial": historial,
    }

    with open(os.path.join(directorio, "historial.json"), "w") as f:
        json.dump(datos_a_guardar, f)


def guardar_checkpoint(directorio, epoca, modelo, optimizador):
    os.makedirs(directorio, exist_ok=True)

    torch.save({
        "modelo_state_dict": modelo.state_dict(),
        "optimizador_state_dict": optimizador.state_dict(),
        "epoca": epoca,
    }, os.path.join(directorio, "checkpoint.pth"))


def Entrenar(model, train_loader, val_loader, criterion, optimizer, device,
             epochs=20, scheduler=None, run="./.run/"):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_model_path = os.path.join(run, "best_model.pth")

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(run, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if __name__ == "__main__":

    train_ds, validation_ds, test_ds = create_datasetsDef(
        DATASET_PATH,
        transform_train=transform_train,
        transform_val=transform_val
    )

    class_to_idx = train_ds.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("class_to_idx:", class_to_idx)
    print("class_names:", class_names)

    print("Distribución train:", contar_clases(train_ds))
    print("Distribución validation:", contar_clases(validation_ds))
    print("Distribución test:", contar_clases(test_ds))

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

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    model = model.to(device)

    summary(model, (3, 224, 224), batch_size=batch_size)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
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
        scheduler=scheduler,
        run="./.run/"
    )

    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    torch.save(model.state_dict(), path_save)
    print(f"Modelo guardado en {path_save}")

    plot_training_history(history)

    print("METRICAS TEST")

    acc, precision, recall, f1, matriz_confusion = calcular_metricas_binariasDef(
        model,
        test_loader,
        device,
        umbral=0.5
    )

    print(f"ACCURACY:   {acc:.2f}")
    print(f"PRECISION:  {precision:.2f}")
    print(f"RECALL:     {recall:.2f}")
    print(f"F1 SCORE:   {f1:.2f}")
    print("CONF MATRIX:")
    print(matriz_confusion)

    model_loaded = models.resnet18(weights=None)
    model_loaded.fc = nn.Linear(model_loaded.fc.in_features, 1)
    model_loaded.load_state_dict(torch.load(path_save, map_location=device))
    model_loaded = model_loaded.to(device)
    model_loaded.eval()