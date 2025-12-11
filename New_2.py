"""
app_complete_es.py
Versión completa: Detección automática por frame + Proyección 3D + Manipulación por mano + GUI CustomTkinter

Requisitos:
- OpenCV (cv2)
- numpy
- torch
- ultralytics
- trimesh
- filterpy
- mediapipe
- Pillow
- customtkinter

Instalación:
pip install opencv-python-headless numpy torch ultralytics trimesh filterpy mediapipe Pillow customtkinter

"""

import cv2
import numpy as np
import threading
import time
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import trimesh
from filterpy.kalman import KalmanFilter
import mediapipe as mp
import os
import sys

# ------------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------------

ARCHIVO_CALIBRACION = "calibracion_charuco.npz"   # archivo con matriz K y distorsión
RUTA_MODELO = "yolov8n.pt"                        # modelo YOLO
RUTA_MODELO_3D = "cubo.glb"                       # modelo 3D
USAR_GPU = torch.cuda.is_available()

# Ejecutar YOLO cada N frames (auto-reajuste)
YOLO_CADA_N_CUADROS = 6

# ------------------------------------------------------------------------
# Validación de archivo de calibración
# ------------------------------------------------------------------------

if not os.path.exists(ARCHIVO_CALIBRACION):
    print(f"Error: No se encontró el archivo de calibración: {ARCHIVO_CALIBRACION}")
    sys.exit(1)

# ------------------------------------------------------------------------
# Cargar calibración
# ------------------------------------------------------------------------

datos_calib = np.load(ARCHIVO_CALIBRACION)
CAM_K, CAM_DIST = datos_calib["K"], datos_calib["D"]

# ------------------------------------------------------------------------
# Inicializar modelo YOLO
# ------------------------------------------------------------------------

dispositivo = "cuda" if USAR_GPU else "cpu"
print(f"Iniciando YOLO en dispositivo: {dispositivo}")

modelo_yolo = YOLO(RUTA_MODELO)
modelo_yolo.to(dispositivo)

# Buscar clase 'book', si no existe usar la clase 0
id_objetivo = None
for k, v in modelo_yolo.names.items():
    if v.lower() == "book":
        id_objetivo = k
        break

if id_objetivo is None:
    id_objetivo = 0
    print("Advertencia: Clase 'book' no encontrada. Se usará clase 0.")

# ------------------------------------------------------------------------
# Cargar modelo 3D con Trimesh
# ------------------------------------------------------------------------

def cargar_modelo_3d(ruta, tam_objetivo=0.12):
    """Carga y normaliza un modelo 3D."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")

    malla = trimesh.load(ruta, force='mesh')
    if malla is None:
        raise RuntimeError("No se pudo cargar la malla.")

    vertices = np.array(malla.vertices, dtype=np.float32)
    caras = np.array(malla.faces, dtype=np.int32)

    minb, maxb = vertices.min(axis=0), vertices.max(axis=0)
    dimensiones = maxb - minb

    escala = tam_objetivo / np.max(dimensiones) if np.max(dimensiones) > 0 else 1.0
    vertices = (vertices - (minb + maxb) / 2.0) * escala

    # Ajuste de eje Z (dependiendo del sistema de coordenadas del modelo)
    vertices[:, 2] *= -1
    return vertices, caras


# Intentar cargar modelo, si falla usar un cubo simple
try:
    if os.path.exists(RUTA_MODELO_3D):
        vertices_modelo, caras_modelo = cargar_modelo_3d(RUTA_MODELO_3D)
        print(f"Modelo 3D cargado: {RUTA_MODELO_3D}")
    else:
        raise FileNotFoundError
except Exception:
    print("No se cargó el modelo. Se usará un cubo por defecto.")
    s = 0.05
    vertices_modelo = np.array([
        [-s,-s,0],[ s,-s,0],[ s,s,0],[-s,s,0],
        [-s,-s,2*s],[ s,-s,2*s],[ s,s,2*s],[-s,s,2*s],
    ], dtype=np.float32)
    caras_modelo = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [1,2,6],[1,6,5],[0,3,7],[0,7,4]
    ], dtype=np.int32)

# ------------------------------------------------------------------------
# Creador de filtro Kalman
# ------------------------------------------------------------------------

def crear_filtro_kalman(dt=1/30.0):
    """Crea un filtro Kalman 2D para suavizar posiciones."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.P *= 500.
    kf.R *= 25.
    kf.Q *= 5.
    kf.x = np.zeros((4,1), dtype=np.float32)
    return kf

filtros_kalman = {}

# ------------------------------------------------------------------------
# MediaPipe Hands
# ------------------------------------------------------------------------

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils

detector_manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Variables para manipulación
UMBRAL_AGARRE = 0.025
SUAVIZADO = 0.6
distancia_filtrada = 1.0
cuadros_cercanos = 0
CUADROS_REQUERIDOS_PARA_AGARRE = 5
objeto_agarrado = None
objeto_bloqueado = False

# Búfer de objetos detectados y su pose
objetos_estables = {}

# Banderas globales
camara_activa = False
proyeccion_activa = False
manipulacion_activa = False

# Frame compartido para GUI
ultimo_frame = None
frame_listo = False

# ------------------------------------------------------------------------
# Función para renderizar modelo 3D proyectado
# ------------------------------------------------------------------------

def renderizar_modelo(frame, vertices, caras, vec_rot, vec_trasl, color=(255,200,100)):
    """Dibuja el modelo 3D proyectado en la imagen."""
    proy, _ = cv2.projectPoints(vertices, vec_rot, vec_trasl, CAM_K, CAM_DIST)
    proy2 = np.int32(np.round(proy)).reshape(-1, 2)

    for f in caras:
        cv2.polylines(frame, [proy2[f]], True, color, 1, cv2.LINE_AA)

# ------------------------------------------------------------------------
# Resolver pose (solvePnP)
# ------------------------------------------------------------------------

def calcular_pose_objeto(frame, caja, id_obj):
    """Calcula la pose del objeto detectado mediante solvePnP."""
    x1,y1,x2,y2 = map(int, caja)
    img_puntos = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

    # Tamaño aproximado del objeto real (metros) - ajustar según el objeto
    ancho_obj = 0.15
    alto_obj = 0.22
    obj_puntos = np.float32([[0,0,0],[ancho_obj,0,0],[ancho_obj,alto_obj,0],[0,alto_obj,0]])

    try:
        sp = cv2.solvePnP(obj_puntos, img_puntos, CAM_K, CAM_DIST)
    except:
        return

    # cv2.solvePnP puede devolver tuplas con diferentes longitudes según la versión
    if len(sp) == 3:
        ok, vec_rot, vec_trasl = sp
    else:
        vec_rot, vec_trasl = sp
        ok = True

    if not ok:
        return

    objetos_estables[id_obj] = {"vec_rot": vec_rot, "vec_trasl": vec_trasl, "ultima_vez": time.time()}

# ------------------------------------------------------------------------
# Hilo principal de la cámara
# ------------------------------------------------------------------------

def hilo_camara(indice_dispositivo=0, ancho=1280, alto=720, fps=30):
    """Hilo que captura video, ejecuta YOLO periódicamente y dibuja proyección."""
    global ultimo_frame, frame_listo
    global distancia_filtrada, cuadros_cercanos
    global objeto_agarrado, objeto_bloqueado

    cap = cv2.VideoCapture(indice_dispositivo)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ultima_limpieza = time.time()
    id_cuadro = 0

    while camara_activa:
        id_cuadro += 1
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        resultados_manos = None
        if manipulacion_activa:
            resultados_manos = detector_manos.process(rgb)

        detecciones = []

        # Ejecutar YOLO cada N cuadros
        if id_cuadro % YOLO_CADA_N_CUADROS == 0:
            try:
                resultados = modelo_yolo.predict(
                    frame, classes=[id_objetivo],
                    imgsz=640, conf=0.5, iou=0.45,
                    device=dispositivo, verbose=False
                )
                for r in resultados:
                    for caja in r.boxes:
                        x1,y1,x2,y2 = caja.xyxy[0].cpu().numpy()
                        conf = float(caja.conf[0])
                        detecciones.append((int(x1),int(y1),int(x2),int(y2),conf))
            except Exception as e:
                print("Error YOLO:", e)

        # Tracking con Kalman
        ids_nuevos = []
        for idx, det in enumerate(detecciones):
            x1,y1,x2,y2,conf = det
            cx, cy = (x1+x2)/2, (y1+y2)/2
            wbox, hbox = (x2-x1)/2, (y2-y1)/2

            if idx not in filtros_kalman:
                filtros_kalman[idx] = crear_filtro_kalman()
                filtros_kalman[idx].x = np.array([[cx],[cy],[0],[0]], dtype=np.float32)

            kf = filtros_kalman[idx]
            kf.predict()
            kf.update(np.array([cx, cy]))

            cx_f, cy_f = kf.x[0], kf.x[1]
            x1f, y1f = int(cx_f - wbox), int(cy_f - hbox)
            x2f, y2f = int(cx_f + wbox), int(cy_f + hbox)

            # Dibujar bbox
            cv2.rectangle(frame, (x1f,y1f), (x2f,y2f), (0,255,0), 2)

            if proyeccion_activa:
                calcular_pose_objeto(frame, (x1f,y1f,x2f,y2f), idx)

            ids_nuevos.append(idx)

        # Manipulación por mano
        if manipulacion_activa and resultados_manos is not None and resultados_manos.multi_hand_landmarks:
            for mano_landmarks in resultados_manos.multi_hand_landmarks:
                pulgar = mano_landmarks.landmark[mp_manos.HandLandmark.THUMB_TIP]
                indice = mano_landmarks.landmark[mp_manos.HandLandmark.INDEX_FINGER_TIP]

                p_pulgar = np.array([pulgar.x*w, pulgar.y*h])
                p_indice = np.array([indice.x*w, indice.y*h])
                distancia = np.linalg.norm(p_pulgar - p_indice) / w

                distancia_filtrada = distancia_filtrada*SUAVIZADO + distancia*(1-SUAVIZADO)
                centro_mano = np.mean([p_pulgar,p_indice], axis=0)

                # Distancia mano → objetos
                distancias = {}
                for lid, datos in objetos_estables.items():
                    vec_rot, vec_trasl = datos["vec_rot"], datos["vec_trasl"]
                    proj3, _ = cv2.projectPoints(np.array([[0,0,0]], np.float32), vec_rot, vec_trasl, CAM_K, CAM_DIST)
                    centro_obj = proj3[0][0]
                    distancias[lid] = np.linalg.norm(centro_mano - centro_obj)

                if distancias:
                    id_obj_mas_cercano = min(distancias, key=distancias.get)

                    # Detectar agarre
                    if not objeto_bloqueado:
                        if distancia_filtrada < UMBRAL_AGARRE:
                            cuadros_cercanos += 1
                            if cuadros_cercanos >= CUADROS_REQUERIDOS_PARA_AGARRE:
                                objeto_agarrado = id_obj_mas_cercano
                                objeto_bloqueado = True
                                cuadros_cercanos = 0
                        else:
                            cuadros_cercanos = 0
                    else:
                        if distancia_filtrada > UMBRAL_AGARRE*1.4:
                            objeto_agarrado = None
                            objeto_bloqueado = False

                    # Mover objeto agarrado
                    if objeto_agarrado is not None:
                        cx, cy = centro_mano
                        factor = 1.0/1200.0
                        objetos_estables[objeto_agarrado]["vec_trasl"][0][0] = (cx - w/2)*factor
                        objetos_estables[objeto_agarrado]["vec_trasl"][1][0] = (cy - h/2)*factor

                mp_dibujo.draw_landmarks(frame, mano_landmarks, mp_manos.HAND_CONNECTIONS)

        # Dibujar modelos 3D persistentes
        if proyeccion_activa:
            for lid, datos in objetos_estables.items():
                vec_rot, vec_trasl = datos["vec_rot"], datos["vec_trasl"]
                renderizar_modelo(frame, vertices_modelo, caras_modelo, vec_rot, vec_trasl)

        # Mostrar estado
        estado_texto = "LIBRE"
        if objeto_agarrado is not None:
            estado_texto = f"AGARRANDO ID {objeto_agarrado}"

        cv2.putText(frame, estado_texto, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        ultimo_frame = frame
        frame_listo = True

        # Eliminar objetos no vistos por mucho tiempo
        if time.time() - ultima_limpieza > 1.0:
            a_borrar = []
            for lid,datos in objetos_estables.items():
                if time.time() - datos["ultima_vez"] > 3.0:
                    a_borrar.append(lid)
            for d in a_borrar:
                objetos_estables.pop(d, None)
            ultima_limpieza = time.time()

        time.sleep(1.0/max(10,fps))

    cap.release()

# ------------------------------------------------------------------------
# GUI con CustomTkinter
# ------------------------------------------------------------------------

class Aplicacion:
    """Interfaz gráfica principal."""
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Proyección 3D - Sistema Completo")
        raiz.geometry("1000x720")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        top = ctk.CTkFrame(raiz)
        top.pack(side="top", fill="x", padx=10, pady=8)

        self.btn_camara = ctk.CTkButton(top, text="Iniciar Cámara", command=self.alternar_camara)
        self.btn_camara.grid(row=0, column=0, padx=6)

        self.btn_proyeccion = ctk.CTkButton(top, text="Activar Proyección 3D", command=self.activar_proyeccion)
        self.btn_proyeccion.grid(row=0, column=1, padx=6)

        self.btn_manip = ctk.CTkButton(top, text="Activar Manipulación", command=self.activar_manipulacion)
        self.btn_manip.grid(row=0, column=2, padx=6)

        self.btn_detener = ctk.CTkButton(top, text="Detener Todo", command=self.detener_todo)
        self.btn_detener.grid(row=0, column=3, padx=6)

        self.btn_salir = ctk.CTkButton(top, text="Salir", command=self.salir)
        self.btn_salir.grid(row=0, column=4, padx=6)

        self.etiqueta_video = ctk.CTkLabel(raiz, text="")
        self.etiqueta_video.pack(padx=10, pady=6, expand=True, fill="both")

        self.estado_label = ctk.CTkLabel(raiz, text="Estado: Idle")
        self.estado_label.pack(side="bottom", fill="x")

        self.actualizar_gui()

    # -----------------------------
    # Control de cámara
    # -----------------------------
    def alternar_camara(self):
        global camara_activa
        if not camara_activa:
            camara_activa = True
            threading.Thread(target=hilo_camara, daemon=True).start()
            self.btn_camara.configure(text="Cámara Activa")
            self.estado_label.configure(text="Estado: Cámara activa")
        else:
            camara_activa = False
            self.btn_camara.configure(text="Iniciar Cámara")
            self.estado_label.configure(text="Estado: Cámara detenida")

    # -----------------------------
    # Activar proyección 3D
    # -----------------------------
    def activar_proyeccion(self):
        global proyeccion_activa
        proyeccion_activa = True
        self.estado_label.configure(text="Estado: Proyección activa")

    # -----------------------------
    # Activar manipulación por mano
    # -----------------------------
    def activar_manipulacion(self):
        global manipulacion_activa
        manipulacion_activa = True
        self.estado_label.configure(text="Estado: Manipulación activa")

    # -----------------------------
    # Detener funciones
    # -----------------------------
    def detener_todo(self):
        global proyeccion_activa, manipulacion_activa
        proyeccion_activa = False
        manipulacion_activa = False
        self.estado_label.configure(text="Estado: Detenido")

    # -----------------------------
    # Salir
    # -----------------------------
    def salir(self):
        global camara_activa
        camara_activa = False
        time.sleep(0.2)
        self.raiz.destroy()

    # -----------------------------
    # Actualizar video en GUI
    # -----------------------------
    def actualizar_gui(self):
        global ultimo_frame, frame_listo
        if frame_listo and ultimo_frame is not None:
            frame_listo = False
            img = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)

            w = max(320, self.etiqueta_video.winfo_width())
            h = max(240, self.etiqueta_video.winfo_height())
            img_pil = img_pil.resize((w, h))

            imgtk = ImageTk.PhotoImage(img_pil)
            self.etiqueta_video.configure(image=imgtk)
            self.etiqueta_video.image = imgtk

        # llamar de nuevo sin ejecutar la función inmediatamente
        self.raiz.after(10, self.actualizar_gui)

# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

if __name__ == "__main__":
    raiz = ctk.CTk()
    app = Aplicacion(raiz)
    raiz.mainloop()
