import time
import math
import cv2
import numpy as np
import tensorflow as tf

# Sustituir por la ruta local del modelo TensorFlow Lite utilizado para la clasificación de maquillaje
TFLITE_PATH = "RUTA_AL_MODELO/modelo_MakeUp_ResNet.tflite"

# Sustituir por la ruta local del archivo de configuración (.prototxt) del detector facial SSD de OpenCV
FACE_PROTO = "RUTA_AL_DETECTOR/deploy.prototxt"

# Sustituir por la ruta local de los pesos (.caffemodel) del detector facial SSD de OpenCV
FACE_MODEL = "RUTA_AL_DETECTOR/res10_300x300_ssd_iter_140000.caffemodel"

CLASS_NAMES = {
    0: "No maquillada",
    1: "Maquillada"
}

UMBRAL = 0.4
FACE_CONFIDENCE = 0.5

CAMERA_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

INFER_EVERY_SECONDS = 5


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def detectar_cara_dnn(frame, net):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    mejor_conf = 0
    mejor_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf > mejor_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            mejor_box = box.astype(int)
            mejor_conf = conf

    if mejor_box is None or mejor_conf < FACE_CONFIDENCE:
        return None

    x1, y1, x2, y2 = mejor_box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2, mejor_conf


def recortar_cara_cuadrada(frame, box, margen=0.10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2, conf = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    ancho = x2 - x1
    alto = y2 - y1

    lado = int(max(ancho, alto) * (1 + margen))

    nx1 = cx - lado // 2
    ny1 = cy - lado // 2
    nx2 = cx + lado // 2
    ny2 = cy + lado // 2

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    cara = frame[ny1:ny2, nx1:nx2]

    return cara, (nx1, ny1, nx2, ny2)


def preprocess_frame_resnet(frame_bgr, input_shape, input_dtype):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(frame_rgb, (224, 224))

    image_np = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_np = (image_np - mean) / std

    if list(input_shape) == [1, 3, 224, 224]:
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)

    elif list(input_shape) == [1, 224, 224, 3]:
        image_np = np.expand_dims(image_np, axis=0)

    else:
        raise ValueError(f"Forma de entrada no esperada: {input_shape}")

    return image_np.astype(input_dtype)


def abrir_camara():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2) 

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    return cap


def main():

    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    print("Modelo TFLite:", TFLITE_PATH)
    print("Entrada:", input_shape, input_dtype)
    print("Salida:", output_details[0]["shape"], output_details[0]["dtype"])
    print("Pulsa CTRL+C para salir")
    print()

    cap = abrir_camara()

    if cap is None:
        print("No se pudo abrir la cámara")
        return

    ultima_inferencia = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error capturando frame")
                time.sleep(1)
                continue

            ahora = time.time()

            if ahora - ultima_inferencia < INFER_EVERY_SECONDS:
                continue

            ultima_inferencia = ahora

            box = detectar_cara_dnn(frame, face_net)

            if box is None:
                print("No se detectó cara")
                continue

            cara, box_cuadrada = recortar_cara_cuadrada(frame, box)

            if cara is None or cara.size == 0:
                print("Cara inválida")
                continue

            cara_224 = cv2.resize(cara, (224, 224))

            image_input = preprocess_frame_resnet(
                frame_bgr=cara_224,
                input_shape=input_shape,
                input_dtype=input_dtype
            )

            start = time.time()

            interpreter.set_tensor(input_index, image_input)
            interpreter.invoke()

            output = interpreter.get_tensor(output_index)

            tiempo = time.time() - start

            logit = float(output.reshape(-1)[0])

            if np.isnan(logit):
                print(
                    f"| ERROR NaN | "
                )
                continue

            prob = sigmoid(logit)

            pred = 1 if prob >= UMBRAL else 0
            clase = CLASS_NAMES[pred]

            x1, y1, x2, y2 = box_cuadrada

            print(
                f"Prob: {prob:.4f} | "
                f"{clase} | "
                f"Logit: {logit:.4f} | "
                f"Tiempo: {tiempo:.4f}s | "
                f"Cara: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
            )

    except KeyboardInterrupt:
        print("\nSaliendo")

    finally:
        cap.release()


if __name__ == "__main__":
    main()