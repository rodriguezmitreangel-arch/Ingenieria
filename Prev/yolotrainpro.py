###############################################################
###      ENTRENAMIENTO YOLOv8 DETECTOR + GRÁFICAS PRO      ###
###   + EXPORTACIÓN MODELOS + GUARDADO PREDICCIONES + LOG   ###
###############################################################

import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import random
from tqdm import tqdm
import shutil
import tensorflow as tf   # 🔥 necesario para limitar VRAM de TF

# ================================================================
# 🔥 LIMITAR VRAM A 12 GB (12,000 MB) — TensorFlow
# ================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=12000)]
        )
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("🔥 TensorFlow: VRAM limitada a 12 GB correctamente.")
    except Exception as e:
        print("⚠️ No se pudo limitar la VRAM en TensorFlow:", e)

# ================================================================
# 🔥 LIMITAR VRAM EN PYTORCH (YOLO usa esto)
# ================================================================
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.90, 0)  
    print("🔥 PyTorch: La memoria máxima se redujo al 90% de disponibilidad.")

# ================================================================
# 🔥 REPRODUCIBILIDAD
# ================================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("\n==============================")
print(" 🔒 ENTORNO REPRODUCIBLE INICIALIZADO")
print("==============================\n")


# ================================================================
# 🔥 SELECCIONAR GPU
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🟢 Usando dispositivo:", device)


# ================================================================
# 🔥 VALIDACIÓN DE DATASET
# ================================================================
def revisar_dataset():
    errores = 0
    for folder in ["train", "validation"]:
        img_dir = f"{folder}/images"
        lbl_dir = f"{folder}/labels"

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"❌ ERROR: falta {folder}/images o {folder}/labels")
            return False

        for img in os.listdir(img_dir):
            base = img.split(".")[0]
            lbl = base + ".txt"
            if not os.path.exists(os.path.join(lbl_dir, lbl)):
                print(f"⚠️ WARNING: Falta etiqueta para {img}")
                errores += 1

    print(f"✔ Revisión dataset completada ({errores} advertencias).")
    return True

revisar_dataset()


# ================================================================
# 🔥 CARGAR MODELO YOLOv8 DETECTOR
# ================================================================
model = YOLO("yolov8x.pt")  # Puedes bajar a yolov8s.pt si tienes pocas imágenes
model.to(device)

print("\n==============================")
print(" 🔥 MODELO CARGADO CORRECTAMENTE")
print("==============================\n")


# ================================================================
# 🔥 ENTRENAMIENTO YOLOv8
# ================================================================
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=7,
    device=device,
    workers=4,
    optimizer="Adam",
    verbose=True
)

print("\n✅ Entrenamiento finalizado\n")

# Ruta donde YOLO guarda resultados
run_dir = model.trainer.save_dir
print("📁 Carpeta del entrenamiento:", run_dir)

# Guardar log automático
with open(os.path.join(run_dir, "log.txt"), "w") as f:
    f.write(str(results))


# ================================================================
# 🔥 MOSTRAR RESULTS.PNG (loss, mAP, cls, obj)
# ================================================================
results_png = os.path.join(run_dir, "results.png")

if os.path.exists(results_png):
    img = plt.imread(results_png)
    plt.figure(figsize=(12,7))
    plt.imshow(img)
    plt.axis("off")
    plt.title("📉 Gráfica completa YOLOv8 (losses + mAP)")
    plt.show()
else:
    print("⚠️ No se encontró results.png.")


# ================================================================
# 🔥 EVALUACIÓN EN VALIDATION SET
# ================================================================
print("\n==============================")
print(" 🔥 EVALUACIÓN EN VALIDACIÓN")
print("==============================\n")

metrics = model.val()
print(metrics)


# ================================================================
# 🔥 FUNCIÓN PROFESIONAL DE PREDICCIÓN + GUARDADO DE IMAGEN
# ================================================================
os.makedirs("runs/predictions", exist_ok=True)
os.makedirs("runs/errors", exist_ok=True)

def predecir_imagen(imagen, guardar=True):
    if not os.path.exists(imagen):
        print("❌ No existe la imagen:", imagen)
        return

    res = model(imagen)[0]

    # Mostrar ventana
    res.show()

    # Guardar imagen predicha con bounding boxes
    if guardar:
        out_name = os.path.basename(imagen).replace(".jpg", "_pred.jpg")
        save_path = os.path.join("runs/predictions", out_name)
        res.save(filename=save_path)
        print(f"💾 Guardado: {save_path}")

    # Mostrar info de detecciones
    if len(res.boxes) == 0:
        print("⚠️ No se detectaron objetos.")
        shutil.copy(imagen, "runs/errors/")
        return

    for box in res.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        print(f"Clase: {model.names[cls]}  |  Confianza: {conf:.2f}")


# ================================================================
# 🔥 PROBAR 3 IMÁGENES ALEATORIAS DE VALIDACIÓN
# ================================================================
print("\n==============================")
print(" 🔥 PREDICCIONES DE EJEMPLO")
print("==============================\n")

val_images = os.listdir("validation/images")
val_paths = [os.path.join("validation/images", x) for x in val_images]
prueba = random.sample(val_paths, min(3, len(val_paths)))

for p in prueba:
    print("\n🖼 Imagen:", p)
    predecir_imagen(p)


# ================================================================
# 🔥 MATRIZ DE CONFUSIÓN YOLO
# ================================================================
print("\n==============================")
print(" 🔥 MATRIZ DE CONFUSIÓN")
print("==============================\n")

conf_png = os.path.join(run_dir, "confusion_matrix.png")

if os.path.exists(conf_png):
    img = plt.imread(conf_png)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("🧩 Matriz de Confusión YOLOv8")
    plt.show()
else:
    print("⚠️ confusion_matrix.png no encontrado")


# ================================================================
# 🔥 EXPORTAR EL MODELO A FORMATOS PROFESIONALES
# ================================================================
print("\n==============================")
print(" 🔥 EXPORTANDO MODELOS")
print("==============================\n")

model.export(format="onnx")
model.export(format="torchscript")
model.export(format="engine")  # TensorRT

print("✔ Modelos exportados correctamente.")


# ================================================================
# 🔥 EVALUAR TEST SET SI EXISTE
# ================================================================
if os.path.exists("test/images"):
    print("\n==============================")
    print(" 🔥 EVALUACIÓN EN TEST SET")
    print("==============================\n")
    metrics_test = model.val(split="test")
    print(metrics_test)
else:
    print("\nℹ️ No existe carpeta test/, se omitió la evaluación.")
