import os
import time
import math
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

CLASS_NAMES = ["No Maquillado", "Maquillado"]
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def cuantizar_input(x_float, input_details):
    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]

    if dtype in [np.int8, np.uint8]:
        x_q = x_float / scale + zero_point
        x_q = np.round(x_q)

        if dtype == np.int8:
            x_q = np.clip(x_q, -128, 127)
        else:
            x_q = np.clip(x_q, 0, 255)

        return x_q.astype(dtype)

    return x_float.astype(dtype)


def dequantizar_output(output, output_details):
    dtype = output_details["dtype"]
    scale, zero_point = output_details["quantization"]

    if dtype in [np.int8, np.uint8]:
        return scale * (output.astype(np.float32) - zero_point)

    return output.astype(np.float32)


def preprocess_image(image_path, input_details):
    input_shape = input_details["shape"]

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)

    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = (image_np - 0.5) / 0.5

    if list(input_shape) == [1, 224, 224, 3]:
        image_np = np.expand_dims(image_np, axis=0)

    elif list(input_shape) == [1, 3, 224, 224]:
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)

    else:
        raise ValueError(f"Forma de entrada no esperada: {input_shape}")

    return cuantizar_input(image_np, input_details)


def inferir_imagen_tflite(ruta_modelo, ruta_imagen, umbral=0.55):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No existe el modelo: {ruta_modelo}")

    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    interpreter = tflite.Interpreter(model_path=ruta_modelo)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]


    print("Modelo:", ruta_modelo)
    print("Imagen:", ruta_imagen)
    print("Entrada shape:", input_details["shape"])
    print("Entrada dtype:", input_details["dtype"])

    image_input = preprocess_image(ruta_imagen, input_details)

    tiempos = []

    print("\nRESULTADOS")

    for i in range(10):
        start = time.time()

        interpreter.set_tensor(input_index, image_input)
        interpreter.invoke()

        output_q = interpreter.get_tensor(output_index)
        output_float = dequantizar_output(output_q, output_details)

        tiempo = time.time() - start
        tiempos.append(tiempo)

        logit = float(output_float.reshape(-1)[0])
        prob = sigmoid(logit)

        if prob >= umbral:
            clase = CLASS_NAMES[1]
        else:
            clase = CLASS_NAMES[0]

        print(
            f"[{i+1}] "
            f"Logit: {logit:.4f} | "
            f"Prob: {prob:.4f} | "
            f"Predicción: {clase} | "
            f"Tiempo: {tiempo:.4f}s"
        )

    print(f"Tiempo medio inferencia: {sum(tiempos) / len(tiempos):.4f}s")

if __name__ == "__main__":

    # Sustituir por la ruta local del modelo TensorFlow Lite INT8 que se utilizará para la inferencia
    MODEL_PATH = "RUTA_AL_MODELO/modelo_MakeUp.tflite"
    # Sustituir por la ruta local de la imagen sobre la que se realizará la inferencia
    IMAGE_PATH = "RUTA_A_LA_IMAGEN/imagen.jpg"

    inferir_imagen_tflite(
        ruta_modelo=MODEL_PATH,
        ruta_imagen=IMAGE_PATH,
        umbral=0.55
    )
