import time
import math
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Sustituir por la ruta local del modelo TensorFlow Lite que se utilizará para la inferencia
MODEL_PATH = "RUTA_AL_MODELO/modelo_MakeUp.tflite"
# Sustituir por la ruta local de la imagen sobre la que se realizará la inferencia
IMAGE_PATH = "RUTA_A_LA_IMAGEN/imagen.jpg"

CLASS_NAMES = ["No Maquillado", "Maquillado"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def preprocess_image(image_path, input_details):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = (image_np - 0.5) / 0.5

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    print("Input esperado:", input_shape, input_dtype)
    if input_shape[1] == 3:
        image_np = np.transpose(image_np, (2, 0, 1))  
        image_np = np.expand_dims(image_np, axis=0)

    else:
        image_np = np.expand_dims(image_np, axis=0)

    image_np = image_np.astype(input_dtype)

    print("Input usado:", image_np.shape)

    return image_np

def inferir_maquillaje_tflite(interpreter, image_input, umbral=0.55):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start_time = time.time()

    interpreter.set_tensor(input_details[0]["index"], image_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    end_time = time.time()

    raw = float(output.flatten()[0])

    if math.isnan(raw) or math.isinf(raw):
        return "ERROR: salida NaN/Inf", float("nan"), end_time - start_time

    prob = sigmoid(raw)
    prediction = 1 if prob >= umbral else 0

    return CLASS_NAMES[prediction], prob, end_time - start_time

if __name__ == "__main__":
    tiempo_total_inicio = time.time()

    print("Imagen:", IMAGE_PATH)
    print("Modelo:", MODEL_PATH)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_input = preprocess_image(IMAGE_PATH, input_details)

    tiempos = []

    for i in range(10):
        clase, probabilidad, tiempo = inferir_maquillaje_tflite(
            interpreter,
            image_input,
            umbral=0.55
        )

        tiempos.append(tiempo)

        print(
            f"[{i+1}] Predicción: {clase}, "
            f"Prob={probabilidad:.4f}, "
            f"Tiempo={tiempo:.4f}s"
        )

    tiempo_promedio = sum(tiempos) / len(tiempos)

    print(f"\n Tiempo promedio de inferencia: {tiempo_promedio:.4f} segundos")

    tiempo_total_fin = time.time()
    print(f" Tiempo total: {tiempo_total_fin - tiempo_total_inicio:.4f} segundos")