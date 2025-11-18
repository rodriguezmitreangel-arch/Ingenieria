import cv2
import numpy as np
import torch
import threading
import time
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import trimesh
from PIL import Image, ImageTk
import multiprocessing

NUCLEOS = 2

datos_calibracion = np.load("calibracion_charuco.npz")
K, distorsion = datos_calibracion["K"], datos_calibracion["D"]

tam_base_cubo = 0.04
escala_cubo = 0.5
offset_z = 0.025
tam_objetivo = tam_base_cubo * escala_cubo

def cargar_objeto(ruta, tam_objetivo=0.07):
    malla = trimesh.load(ruta, force='mesh')
    vertices = np.array(malla.vertices, np.float32)
    caras = np.array(malla.faces, np.int32)
    min_b, max_b = vertices.min(0), vertices.max(0)
    tam = max_b - min_b
    escala = tam_objetivo / np.max(tam)
    vertices = (vertices - (min_b + max_b) / 2) * escala
    vertices[:, 2] *= -1
    return vertices, caras

vertices_objeto, caras_objeto = cargar_objeto("modelo.obj", tam_objetivo=tam_objetivo)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
modelo = YOLO("yolov8x.pt").to(dispositivo)
id_objetivo = [k for k, v in modelo.names.items() if v == "book"][0]

def crear_kalman(dt=1/30):
    kalman = KalmanFilter(dim_x=4, dim_z=2)
    kalman.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], np.float32)
    kalman.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], np.float32)
    kalman.P *= 500.0
    kalman.R *= 25
    kalman.Q *= 5
    return kalman

seguidores, libros_estables = {}, {}

camara_activa = False
detectar_activo = False
proyeccion_activa = False
ultimo_frame = None
frame_listo = False

def proyectar_vertices_parcial(vertices, rvec, tvec, K, dist):
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, K, dist)
    return imgpts.reshape(-1, 2)

def proyectar_multiproceso(vertices, rvec, tvec, K, dist, n_cores=4):
    partes = np.array_split(vertices, n_cores)
    with multiprocessing.Pool(n_cores) as pool:
        resultados = pool.starmap(proyectar_vertices_parcial,
                                  [(p, rvec, tvec, K, dist) for p in partes])
    return np.vstack(resultados)

def proyectar_objeto(frame, caja, tvec_manual=None):
    x1, y1, x2, y2 = map(int, caja)
    esquinas_img = np.float32([[x1, y1],
                               [x2, y1],
                               [x2, y2],
                               [x1, y2]])

    libro_w, libro_h = 0.15, 0.22
    esquinas_obj = np.float32([
        [0, 0, 0],
        [libro_w, 0, 0],
        [libro_w, libro_h, 0],
        [0, libro_h, 0]
    ])

    exito, rvec, tvec = cv2.solvePnP(esquinas_obj, esquinas_img, K, dist)
    if not exito:
        return

    if tvec_manual is not None:
        tvec = tvec_manual

    offset_x = (libro_w - tam_objetivo) / 2
    offset_y = (libro_h - tam_objetivo) / 2
    offset_local_z = offset_z
    vertices_ajustados = vertices_objeto + np.array([offset_x, offset_y, offset_local_z], np.float32)

    imgpts = proyectar_multiproceso(vertices_ajustados, rvec, tvec, K, dist, NUCLEOS)
    imgpts = np.int32(np.round(imgpts)).reshape(-1, 2)

    for c in caras_objeto:
        pts = imgpts[c]
        cv2.polylines(frame, [pts], True, (255, 200, 100), 1)

def hilo_camara():
    global ultimo_frame, frame_listo, camara_activa, detectar_activo, proyeccion_activa

    cap = cv2.VideoCapture(2, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    tvec_manual = np.array([[0.0], [0.0], [0.5]], np.float32)
    tvec_guardado = tvec_manual.copy()

    while camara_activa:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        if detectar_activo:
            resultados = modelo.predict(frame, classes=[id_objetivo],
                                        imgsz=960, conf=0.75, iou=0.5,
                                        device=dispositivo, verbose=False)
            for r in resultados:
                for caja in r.boxes:
                    x1, y1, x2, y2 = map(int, caja.xyxy[0].cpu().numpy())
                    libros_estables[0] = (x1, y1, x2, y2, 1.0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    tvec_guardado = np.array([[0.0], [0.0], [0.5]], np.float32)

        if proyeccion_activa and libros_estables:
            detectar_activo = False
            inicio_t = time.time()

            for _, (x1, y1, x2, y2, _) in libros_estables.items():
                proyectar_objeto(frame, (x1, y1, x2, y2), tvec_guardado)

            duracion = time.time() - inicio_t
            print(f"Tiempo total de proyección ({NUCLEOS} núcleos): {duracion:.3f} s")

            proyeccion_activa = False

        ultimo_frame = frame
        frame_listo = True
        time.sleep(1 / 60)

    cap.release()

def actualizar_gui():
    global ultimo_frame, frame_listo
    if frame_listo and ultimo_frame is not None:
        frame_listo = False
        img = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        etiqueta_video.imgtk = imgtk
        etiqueta_video.configure(image=imgtk)
    ventana.after(15, actualizar_gui)

def iniciar_camara():
    global camara_activa
    if not camara_activa:
        camara_activa = True
        threading.Thread(target=hilo_camara, daemon=True).start()

def activar_deteccion():
    global detectar_activo
    detectar_activo = True
    print("Detección de libros activada")

def proyectar_objetos():
    global proyeccion_activa
    proyeccion_activa = True
    print("Proyección iniciada: detección pausada")

def detener_todo():
    global detectar_activo, proyeccion_activa
    detectar_activo = False
    proyeccion_activa = False
    print("Proyección y detección detenidas")

def salir():
    global camara_activa
    camara_activa = False
    ventana.destroy()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    ventana = tk.Tk()
    ventana.title("Proyección multiproceso del cubo.glb sobre libro")
    ventana.geometry("1280x800")
    try:
        ventana.state("zoomed")
    except:
        pass

    marco_controles = ttk.Frame(ventana)
    marco_controles.pack(side="top", pady=10)

    ttk.Button(marco_controles, text="Iniciar cámara", command=iniciar_camara).grid(row=0, column=0, padx=10)
    ttk.Button(marco_controles, text="Detectar libros", command=activar_deteccion).grid(row=0, column=1, padx=10)
    ttk.Button(marco_controles, text="Proyectar GLB", command=proyectar_objetos).grid(row=0, column=2, padx=10)
    ttk.Button(marco_controles, text="Detener todo", command=detener_todo).grid(row=0, column=3, padx=10)
    ttk.Button(marco_controles, text="Salir", command=salir).grid(row=0, column=4, padx=10)

    etiqueta_video = ttk.Label(ventana)
    etiqueta_video.pack(expand=True, fill="both", padx=10, pady=10)

    ventana.after(15, actualizar_gui)
    ventana.mainloop()
