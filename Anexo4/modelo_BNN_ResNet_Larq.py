import os
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import larq as lq
import larq_compute_engine as lce

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)


@dataclass
class Config:
    # Sustituir por la ruta local donde se ha descargado el conjunto de datos
    dataset_dir: str = "RUTA_AL_DATASET"
    image_size: tuple = (224, 224)
    
    batch_size: int = 16

    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 42

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    threshold: float = 0.55

    initial_filters: int = 64
    num_layers: int = 18

   # Sustituir por la ruta local donde se guardarán los modelos y resultados generados
    output_dir: str = "RUTA_DE_GUARDADO"
    best_model_name: str = "best_binary_resnet_makeup.weights.h5"
    final_model_name: str = "final_binary_resnet_makeup.keras"
    lce_tflite_name: str = "binary_resnet_makeup_lce.tflite"


CFG = Config()

CLASS_TO_LABEL = {
    "NoMakeup": 0,
    "SiMakeup": 1,
}

LABEL_TO_TEXT = {
    0: "No Maquillado",
    1: "Maquillado",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_ratios():
    total = CFG.train_ratio + CFG.val_ratio + CFG.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio debe sumar 1.0 y ahora suma {total}"
        )


def list_images_and_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe dataset_dir: {dataset_dir}")

    filepaths = []
    labels = []

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Falta la carpeta de clase: {class_dir}")

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_ext:
                filepaths.append(str(path))
                labels.append(label)

    if len(filepaths) == 0:
        raise ValueError("No se encontraron imágenes en el dataset.")

    return np.array(filepaths), np.array(labels)


def make_splits(filepaths, labels):
    x_train, x_temp, y_train, y_temp = train_test_split(
        filepaths,
        labels,
        test_size=(1.0 - CFG.train_ratio),
        random_state=CFG.seed,
        stratify=labels,
    )

    val_relative = CFG.val_ratio / (CFG.val_ratio + CFG.test_ratio)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1.0 - val_relative),
        random_state=CFG.seed,
        stratify=y_temp,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def decode_and_preprocess(path, label, training=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, CFG.image_size)
    image = tf.cast(image, tf.float32) / 255.0

    image = (image - 0.5) / 0.5

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)

    label = tf.cast(label, tf.float32)

    return image, label


def build_tf_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=CFG.seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda x, y: decode_and_preprocess(x, y, training=training),
        num_parallel_calls=2,
    )

    ds = ds.batch(CFG.batch_size)

    ds = ds.prefetch(1)

    return ds



class BinaryResNetE18BinaryHead:
    def __init__(
        self,
        input_shape=(224, 224, 3),
        initial_filters=64,
        num_layers=18,
    ):
        self.input_shape = input_shape
        self.initial_filters = initial_filters
        self.num_layers = num_layers

    @property
    def input_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_quantizer(self):
        return lq.quantizers.SteSign(clip_value=1.25)

    @property
    def kernel_constraint(self):
        return lq.constraints.WeightClip(clip_value=1.25)

    @property
    def spec(self):
        spec = {
            18: ([2, 2, 2, 2], [64, 128, 256, 512]),
            34: ([3, 4, 6, 3], [64, 128, 256, 512]),
            50: ([3, 4, 6, 3], [256, 512, 1024, 2048]),
            101: ([3, 4, 23, 3], [256, 512, 1024, 2048]),
            152: ([3, 8, 36, 3], [256, 512, 1024, 2048]),
        }

        if self.num_layers not in spec:
            raise ValueError(f"num_layers debe ser una de {list(spec.keys())}")

        return spec[self.num_layers]

    def residual_block(self, x, filters, strides=1):
        downsample = x.shape[-1] != filters

        if downsample:
            residual = tf.keras.layers.AvgPool2D(
                pool_size=2,
                strides=2
            )(x)

            residual = tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                use_bias=False,
                kernel_initializer="glorot_normal",
            )(residual)

            residual = tf.keras.layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5
            )(residual)
        else:
            residual = x

        x = lq.layers.QuantConv2D(
            filters,
            kernel_size=3,
            strides=strides,
            padding="same",
            input_quantizer=self.input_quantizer,
            kernel_quantizer=self.kernel_quantizer,
            kernel_constraint=self.kernel_constraint,
            kernel_initializer="glorot_normal",
            use_bias=False,
        )(x)

        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5
        )(x)

        x = tf.keras.layers.Add()([x, residual])

        return x

    def build(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(
            self.initial_filters,
            kernel_size=7,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(inputs)

        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5
        )(x)

        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding="same"
        )(x)

        x = tf.keras.layers.BatchNormalization(
            momentum=0.9,
            epsilon=1e-5
        )(x)

        blocks_per_stage, filters_per_stage = self.spec

        for block_idx, (layers, filters) in enumerate(
            zip(blocks_per_stage, filters_per_stage)
        ):
            for layer_idx in range(layers * 2):
                strides = 1 if block_idx == 0 or layer_idx != 0 else 2
                x = self.residual_block(x, filters, strides=strides)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Salida FP32
        x = tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_normal"
        )(x)

        outputs = tf.keras.layers.Activation(
            "sigmoid",
            dtype="float32"
        )(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="binary_resnet_e18_makeup"
        )

        return model


def build_model():
    return BinaryResNetE18BinaryHead(
        input_shape=(*CFG.image_size, 3),
        initial_filters=CFG.initial_filters,
        num_layers=CFG.num_layers,
    ).build()


def predict_dataset(model, ds):
    y_true = []
    y_prob = []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().astype(np.int32).tolist())

    return np.array(y_true), np.array(y_prob)


def print_final_metrics(y_true, y_prob, threshold=0.55, title="TEST"):
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print(f"\nMÉTRICAS FINALES - {title}")
    print(f"Accuracy  : {acc:.6f}")
    print(f"Precision : {prec:.6f}")
    print(f"Recall    : {rec:.6f}")
    print(f"F1-score  : {f1:.6f}")
    print("Matriz de confusión:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["No Maquillado", "Maquillado"],
            digits=6,
            zero_division=0,
        )
    )


def export_lce_tflite(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = os.path.join(output_dir, CFG.lce_tflite_name)

    flatbuffer = lce.convert_keras_model(
        model,
        inference_input_type=tf.float32,
        inference_output_type=tf.float32,
        target="arm",
    )

    with open(tflite_path, "wb") as f:
        f.write(flatbuffer)

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

    print(f"\nModelo LCE exportado en: {tflite_path}")
    print(f"Tamaño .tflite: {size_mb:.2f} MB")

    return tflite_path


def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CFG.learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="acc",
                threshold=CFG.threshold
            ),
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=CFG.threshold
            ),
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=CFG.threshold
            ),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def get_callbacks():
    os.makedirs(CFG.output_dir, exist_ok=True)

    best_model_path = os.path.join(
        CFG.output_dir,
        CFG.best_model_name
    )

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(CFG.output_dir, "training_log.csv")
        ),
    ]


def main():
    ensure_ratios()
    set_seed(CFG.seed)

    os.makedirs(CFG.output_dir, exist_ok=True)

    filepaths, labels = list_images_and_labels(CFG.dataset_dir)

    print(f"Total imágenes: {len(filepaths)}")
    print(f"NoMakeUp: {(labels == 0).sum()}")
    print(f"SiMakeUP: {(labels == 1).sum()}")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_splits(
        filepaths,
        labels
    )

    print("\nReparto:")
    print(f"Train: {len(x_train)}")
    print(f"Val  : {len(x_val)}")
    print(f"Test : {len(x_test)}")

    train_ds = build_tf_dataset(x_train, y_train, training=True)
    val_ds = build_tf_dataset(x_val, y_val, training=False)
    test_ds = build_tf_dataset(x_test, y_test, training=False)

    model = build_model()
    compile_model(model)

    print("\nResumen del modelo:")
    model.summary()

    print("\nResumen Larq:")
    lq.models.summary(model)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CFG.epochs,
        callbacks=get_callbacks(),
        verbose=1,
    )

    best_model_path = os.path.join(
        CFG.output_dir,
        CFG.best_model_name
    )

    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)
        print(f"\nPesos mejores cargados desde: {best_model_path}")

    print("\nEvaluación Keras en VALIDACIÓN:")
    val_results = model.evaluate(val_ds, verbose=1)
    for name, value in zip(model.metrics_names, val_results):
        print(f"{name}: {value:.6f}")

    print("\nEvaluación Keras en TEST:")
    test_results = model.evaluate(test_ds, verbose=1)
    for name, value in zip(model.metrics_names, test_results):
        print(f"{name}: {value:.6f}")

    y_true_val, y_prob_val = predict_dataset(model, val_ds)
    print_final_metrics(
        y_true_val,
        y_prob_val,
        threshold=CFG.threshold,
        title="VALIDACIÓN"
    )

    y_true_test, y_prob_test = predict_dataset(model, test_ds)
    print_final_metrics(
        y_true_test,
        y_prob_test,
        threshold=CFG.threshold,
        title="TEST"
    )

    final_model_path = os.path.join(
        CFG.output_dir,
        CFG.final_model_name
    )

    model.save(final_model_path)
    print(f"\nModelo final guardado en: {final_model_path}")

    keras_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
    print(f"Tamaño .keras: {keras_size_mb:.2f} MB")

    try:
        tflite_path = export_lce_tflite(model, CFG.output_dir)
        tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

        reduction = 100 * (1 - (tflite_size_mb / keras_size_mb))

        print(f"Tamaño .tflite LCE: {tflite_size_mb:.2f} MB")
        print(f"Reducción respecto a .keras: {reduction:.2f}%")

    except Exception as e:
        print("\nNo se pudo exportar con LCE.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
