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
import mediapipe as mp

# ============================================================
# Calibración
# ============================================================
data = np.load("calibracion_charuco.npz")
K, dist = data["K"], data["D"]

# ============================================================
# Parámetros base
# ============================================================
base_cube_size = 0.1
cube_scale = 0.5
z_offset = 0.025
target_size = base_cube_size * cube_scale
distancia_umbral = 10 # px

# ============================================================
# Cargar modelo 3D
# ============================================================
def cargar_obj(path, target_size=1):
    mesh = trimesh.load(path, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    min_bounds, max_bounds = vertices.min(axis=0), vertices.max(axis=0)
    size = max_bounds - min_bounds
    scale = target_size / np.max(size)
    vertices = (vertices - (min_bounds + max_bounds) / 2) * scale
    vertices[:, 2] *= -1
    return vertices, faces

vertices_obj, faces_obj = cargar_obj("cubo.glb", target_size)

# ============================================================
# YOLO + Kalman
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)
target_id = [k for k, v in model.names.items() if v == "book"][0]

def create_kalman(dt=1/30):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]], np.float32)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], np.float32)
    kf.P *= 500.0
    kf.R *= 25
    kf.Q *= 5
    return kf

trackers, libros_estables = {}, {}

# ============================================================
# MediaPipe Hands
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

umbral_agarre = 0.020
suavizado = 0.5
distancia_filtrada = 0
obj_agarrado = None

# ============================================================
# Bloqueo de objetos
# ============================================================
obj_bloqueado = False  # si True, ningún otro objeto puede moverse
frames_cercanos = 0
frames_requeridos_para_agarre = 5

# ============================================================
# Estados globales
# ============================================================
camara_activa = False
deteccion_activa = False
proyeccion_activa = False
manipulacion_activa = False
ultimo_frame = None
frame_ready = False

# ============================================================
# Render y proyección centrada
# ============================================================
def render_model(frame, vertices, rvec, tvec):
    proj, _ = cv2.projectPoints(vertices, rvec, tvec, K, dist)
    proj = np.int32(np.round(proj)).reshape(-1, 2)
    for f in faces_obj:
        cv2.polylines(frame, [proj[f]], True, (255, 200, 100), 1)

def proyectar_objeto(frame, box, libro_id):
    x1, y1, x2, y2 = map(int, box)
    img_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    book_w, book_h = 0.15, 0.22
    obj_corners = np.float32([[0, 0, 0],
                              [book_w, 0, 0],
                              [book_w, book_h, 0],
                              [0, book_h, 0]])
    success, rvec, tvec = cv2.solvePnP(obj_corners, img_corners, K, dist)
    if not success:
        return

    offset_x = book_w / 2
    offset_y = book_h / 2
    offset_z = z_offset
    vertices_centrados = vertices_obj + np.array([
        offset_x - target_size/2,
        offset_y - target_size/2,
        offset_z
    ], np.float32)

    if libro_id not in libros_estables:
        libros_estables[libro_id] = {"rvec": rvec, "tvec": tvec}
    else:
        libros_estables[libro_id]["rvec"], libros_estables[libro_id]["tvec"] = rvec, tvec

    render_model(frame, vertices_centrados, rvec, tvec)

# ============================================================
# Hilo principal de cámara
# ============================================================
def cam_thread():
    global ultimo_frame, frame_ready, obj_agarrado, distancia_filtrada, frames_cercanos, obj_bloqueado

    cap = cv2.VideoCapture(1, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while camara_activa:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(rgb)
        h, w, _ = frame.shape

        # === YOLO detección ===
        if deteccion_activa:
            results = model.predict(frame, classes=[target_id],
                                    imgsz=960, conf=0.75, iou=0.5,
                                    device=device, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append((x1, y1, x2, y2, conf))

            for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
                cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
                dx, dy = (x2 - x1)/2.0, (y2 - y1)/2.0
                if idx not in trackers:
                    trackers[idx] = create_kalman()
                kf = trackers[idx]
                kf.predict()
                kf.update(np.array([cx, cy], np.float32))
                cx_f, cy_f = float(kf.x[0]), float(kf.x[1])
                x1f, y1f, x2f, y2f = cx_f - dx, cy_f - dy, cx_f + dx, cy_f + dy

                if not manipulacion_activa:
                    cv2.rectangle(frame, (int(x1f), int(y1f)), (int(x2f), int(y2f)), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{idx}", (int(x1f), int(y1f)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                if proyeccion_activa:
                    proyectar_objeto(frame, (x1f, y1f, x2f, y2f), idx)

        # === Manipulación con bloqueo de objetos ===
        if manipulacion_activa and result_hands.multi_hand_landmarks and libros_estables:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                pulgar = np.array([
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h
                ])
                indice = np.array([
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
                ])
                dist_mano = np.linalg.norm(pulgar - indice) / w
                distancia_filtrada = suavizado * distancia_filtrada + (1 - suavizado) * dist_mano
                centro_mano = np.mean([pulgar, indice], axis=0)

                # Calcular distancias a los objetos
                distancias = {}
                for lid, data in libros_estables.items():
                    rvec, tvec = data["rvec"], data["tvec"]
                    proj, _ = cv2.projectPoints(np.array([[0, 0, 0]], np.float32),
                                                rvec, tvec, K, dist)
                    centro_obj = proj[0][0]
                    distancias[lid] = np.linalg.norm(centro_mano - centro_obj)
                if not distancias:
                    continue
                obj_cercano = min(distancias, key=distancias.get)

                # Si no hay bloqueo, permitir agarre
                if not obj_bloqueado:
                    if distancia_filtrada < umbral_agarre:
                        frames_cercanos += 1
                        if frames_cercanos >= frames_requeridos_para_agarre:
                            obj_agarrado = obj_cercano
                            obj_bloqueado = True  # bloquear los demás
                    else:
                        frames_cercanos = 0
                else:
                    # Liberar cuando se abre la mano
                    if obj_agarrado is not None and distancia_filtrada > umbral_agarre * 1.4:
                        obj_agarrado = None
                        obj_bloqueado = False  # desbloquear los demás

                # Solo mover el objeto agarrado
                if obj_agarrado is not None:
                    cx, cy = centro_mano
                    libros_estables[obj_agarrado]["tvec"][0][0] = (cx - 640) / 2000
                    libros_estables[obj_agarrado]["tvec"][1][0] = (cy - 360) / 2000

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === Render persistente ===
        if proyeccion_activa and libros_estables:
            for lid, data in libros_estables.items():
                vertices_centrados = vertices_obj + np.array([
                    0.075 - target_size/2,
                    0.11 - target_size/2,
                    z_offset
                ], np.float32)
                render_model(frame, vertices_centrados, data["rvec"], data["tvec"])

        # Estado visual
        estado = f"AGARRANDO ID:{obj_agarrado}" if obj_agarrado is not None else "LIBRE"
        if obj_bloqueado and obj_agarrado is not None:
            estado += " | OTROS BLOQUEADOS"
        cv2.putText(frame, estado, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,0) if obj_agarrado is not None else (0,0,255), 3)

        ultimo_frame = frame
        frame_ready = True
        time.sleep(1/60)

    cap.release()

# ============================================================
# GUI
# ============================================================
def actualizar_gui():
    global ultimo_frame, frame_ready
    if frame_ready and ultimo_frame is not None:
        frame_ready = False
        img = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(15, actualizar_gui)

def iniciar_camara():
    global camara_activa
    if not camara_activa:
        camara_activa = True
        threading.Thread(target=cam_thread, daemon=True).start()

def activar_deteccion():
    global deteccion_activa, manipulacion_activa
    deteccion_activa = True
    manipulacion_activa = False

def activar_manipulacion():
    global deteccion_activa, manipulacion_activa
    deteccion_activa = False
    manipulacion_activa = True

def proyectar_objetos():
    global proyeccion_activa
    proyeccion_activa = True

def detener_todo():
    global deteccion_activa, proyeccion_activa, manipulacion_activa
    deteccion_activa = False
    proyeccion_activa = False
    manipulacion_activa = False

def salir():
    global camara_activa
    camara_activa = False
    root.destroy()

# ============================================================
# Interfaz Tkinter
# ============================================================
root = tk.Tk()
root.title("📘 Proyección y Manipulación 3D con bloqueo de objetos")
root.geometry("900x700")

frame_controls = ttk.Frame(root)
frame_controls.pack(side="top", pady=10)

ttk.Button(frame_controls, text="📷 Iniciar cámara", command=iniciar_camara).grid(row=0, column=0, padx=10)
ttk.Button(frame_controls, text="🔍 Detectar libros", command=activar_deteccion).grid(row=0, column=1, padx=10)
ttk.Button(frame_controls, text="📦 Proyectar .OBJ", command=proyectar_objetos).grid(row=0, column=2, padx=10)
ttk.Button(frame_controls, text="🖐️ Manipular objetos", command=activar_manipulacion).grid(row=0, column=3, padx=10)
ttk.Button(frame_controls, text="⏹ Detener todo", command=detener_todo).grid(row=0, column=4, padx=10)
ttk.Button(frame_controls, text="❌ Salir", command=salir).grid(row=0, column=5, padx=10)

video_label = ttk.Label(root)
video_label.pack(pady=10)

root.after(15, actualizar_gui)
root.mainloop()
