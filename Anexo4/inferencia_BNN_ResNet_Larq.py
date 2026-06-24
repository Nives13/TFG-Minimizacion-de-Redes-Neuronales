import os
import time
import numpy as np
import tensorflow as tf
import larq as lq

from PIL import Image

# Sustituir por la ruta local de la BNN en formato Keras que se utilizará para la inferencia
MODEL_PATH = "RUTA_AL_MODELO/modelo_MakeUp.keras"
# Sustituir por la ruta local de la imagen sobre la que se realizará la inferencia
IMAGE_PATH = "RUTA_A_LA_IMAGEN/imagen.jpg"

CLASS_NAMES = ["No Maquillado", "Maquillado"]
UMBRAL = 0.55
IMAGE_SIZE = (224, 224)



def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = (image_np - 0.5) / 0.5

    image_np = np.expand_dims(image_np, axis=0)

    return image_np.astype(np.float32)


def cargar_modelo_keras(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "QuantConv2D": lq.layers.QuantConv2D,
            "SteSign": lq.quantizers.SteSign,
            "WeightClip": lq.constraints.WeightClip,
        },
        compile=False
    )

    return model


def inferir(model, image_input, umbral=0.55):
    start_time = time.time()

    prob = float(model.predict(image_input, verbose=0).reshape(-1)[0])

    end_time = time.time()

    prediction = 1 if prob >= umbral else 0
    clase = CLASS_NAMES[prediction]

    return clase, prob, end_time - start_time


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No existe la imagen: {IMAGE_PATH}")

    print("Modelo:", MODEL_PATH)
    print("Imagen:", IMAGE_PATH)

    print("\nCargando modelo Keras/Larq...")
    model = cargar_modelo_keras(MODEL_PATH)
    print("Modelo cargado correctamente")

    image_input = preprocess_image(IMAGE_PATH)

    tiempos = []

    print("\nRESULTADOS")
    print("-" * 80)

    for i in range(10):
        clase, probabilidad, tiempo = inferir(
            model,
            image_input,
            umbral=UMBRAL
        )

        tiempos.append(tiempo)

        print(
            f"[{i+1}] Predicción: {clase}, "
            f"Prob={probabilidad:.4f}, "
            f"Tiempo={tiempo:.4f}s"
        )

    tiempo_promedio = sum(tiempos) / len(tiempos)

    print("\n" + "=" * 80)
    print(f"Tiempo promedio de inferencia: {tiempo_promedio:.4f} segundos")