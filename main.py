
# -*- coding: utf-8 -*-
"""
V12_2

Real-time AR con detección de superficie (YOLO + refinamiento), pausa de detección
cuando hay gesto de mano (MediaPipe) y manipulación 3D. 

Controles:
- ESC: salir
- h: mostrar/ocultar landmarks de la mano
- i: mostrar/ocultar overlay de información del gesto
- r: reanclar (reacquire) la pose base al marcador (si hay esquinas válidas)
- f: alternar seguimiento del marcador en TRACK (si está activado, rehace base continuamente)


Nota: Si no se cuenta con calibración de cámara (calibracion_cam.npz), el script
usa intrínsecos aproximados según el tamaño del frame.
"""

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from queue import Queue

# Render 3D
import trimesh
import pyrender

# MediaPipe Hands
import mediapipe as mp

# =====================================================
# CONFIGURACIÓN
# =====================================================
ID_CAMARA = 2
RUTA_MODELO = "yolov8s-seg_openvino_model"  # Ruta a modelo YOLOv8 de segmentación
UMBRAL_CONF = 0.4
TAM_IMAGEN = 320

# Dimensiones reales del plano (metros) - hoja A4
ANCHO_REAL = 0.21
ALTO_REAL = 0.297

# Render 3D
ZCERCA, ZLEJOS = 0.01, 10.0
INVERTIR_Y = True
INVERTIR_Z = True
ELEVACION_MODELO_MM = 1
FACTOR_LONGITUD_EJES = 1.0
FACTOR_RADIO_EJES = 0.04
INTENSIDAD_LUZ = 4.0
MOSTRAR_VENTANAS_DEBUG = True
TITULO_VENTANA = "AR con Gestos (TRACK/GESTURE) — Offset baked"

# Flujo óptico (Lucas–Kanade)
parametros_lk = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Visualización de landmarks
ui = {
    'mostrar_landmarks_mano': True,
    'mostrar_info_gesto': True,
}
RADIO_PUNTO_LM = 4
COLOR_PUNTO_LM = (0, 255, 0)
COLOR_ID_LM = (255, 255, 255)
COLOR_LINEA_LM = (255, 0, 255)

# Parámetros del gesto
PINCH_ON = 35.0      # px para "agarrar"
PINCH_OFF = 45.0     # px para "soltar"
TIEMPO_ESPERA_MANO = 0.6   # s sin mano para volver a TRACK
ESCALA_MOV_X = ANCHO_REAL / 400.0  # m/px aprox plano X
ESCALA_MOV_Y = ALTO_REAL / 400.0   # m/px aprox plano Y
EMA_ROT = 0.7        # suavizado rotación
EMA_ESC = 0.7        # suavizado escala
ESCALA_MIN, ESCALA_MAX = 0.3, 3.0

# Reanclado/reacquire de base
SEGUIR_MARCADOR_EN_TRACK = False   # si True, rehace base continuamente en TRACK
ENFRIAMIENTO_REANCLADO_S = 0.8     # tras soltar, espera antes de reanclar automáticamente

# =====================================================
# INICIALIZACIÓN
# =====================================================
# Calibración de cámara (opcional)
matriz_camara = None
coef_distorsion = None
try:
    datos_cal = np.load("calibracion_cam.npz")
    matriz_camara = datos_cal["cameraMatrix"].astype(np.float32)
    coef_distorsion = datos_cal["distCoeffs"].astype(np.float32).reshape(-1)
    print("Calibración cargada de 'calibracion_cam.npz'")
except Exception:
    print("No se encontró 'calibracion_cam.npz'. Se usarán intrínsecos aproximados.")

# Modelo YOLO de segmentación
modelo_yolo = YOLO(RUTA_MODELO, task="segment")

# Puntos 3D del plano (centro en el origen)
puntos_objeto = np.array([
    [-ANCHO_REAL/2, -ALTO_REAL/2, 0],  # TL
    [ ANCHO_REAL/2, -ALTO_REAL/2, 0],  # TR
    [ ANCHO_REAL/2,  ALTO_REAL/2, 0],  # BR
    [-ANCHO_REAL/2,  ALTO_REAL/2, 0],  # BL
], dtype=np.float32)

# Colas para el trabajador de YOLO
cola_frames = Queue(maxsize=1)
cola_resultados_yolo = Queue(maxsize=1)
ejecucion_activa = True

# =====================================================
# TRABAJADOR YOLO
# =====================================================
def trabajador_yolo():
    """
    Hilo de ejecución que consume frames y produce cajas ROI con detecciones de YOLO.
    """
    while ejecucion_activa:
        if cola_frames.empty():
            time.sleep(0.01)
            continue
        frame = cola_frames.get()
        try:
            resultados = modelo_yolo(frame, conf=UMBRAL_CONF, imgsz=TAM_IMAGEN, verbose=False)
        except Exception as e:
            print("Error en YOLO:", e)
            continue
        # Validar resultados y extraer primera caja
        if resultados and len(resultados) > 0 and getattr(resultados[0], "boxes", None) is not None and len(resultados[0].boxes) > 0:
            caja = resultados[0].boxes[0]
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            if not cola_resultados_yolo.empty():
                _ = cola_resultados_yolo.get()
            cola_resultados_yolo.put((x1, y1, x2, y2))

# =====================================================
# Utilidades de detección/refinamiento
# =====================================================
def ordenar_esquinas(puntos):
    """
    Ordena las 4 esquinas como TL, TR, BR, BL a partir de un conjunto no ordenado.
    """
    puntos = puntos.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    suma = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(suma)]  # TL
    rect[2] = puntos[np.argmax(suma)]  # BR
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]  # TR
    rect[3] = puntos[np.argmax(diff)]  # BL
    return rect

def refinar_deteccion_robusta(frame, caja_roi):
    """
    Aplica Canny + dilatación + convex hull + approxPolyDP para obtener 4 esquinas del plano.
    Devuelve (esquinas, imagen_debug, roi).
    """
    x1, y1, x2, y2 = caja_roi
    h_img, w_img = frame.shape[:2]
    p = 30  # padding para contexto alrededor de la caja
    x1, y1 = max(0, x1-p), max(0, y1-p)
    x2, y2 = min(w_img, x2+p), min(h_img, y2+p)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None, None
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilatado = cv2.dilate(bordes, kernel, iterations=1)
    contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejores_esquinas = None
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        casco = cv2.convexHull(c)
        perimetro = cv2.arcLength(casco, True)
        aprox = cv2.approxPolyDP(casco, 0.04 * perimetro, True)
        if len(aprox) == 4:
            pts = aprox.reshape(-1, 2)
            pts[:, 0] += x1
            pts[:, 1] += y1
            mejores_esquinas = ordenar_esquinas(pts)
    return mejores_esquinas, dilatado, roi

# =====================================================
# Matrices de transformación
# =====================================================
def matriz_rotacion_x(grados):
    ang = np.deg2rad(grados)
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(ang), -np.sin(ang), 0],
                     [0, np.sin(ang),  np.cos(ang), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def matriz_rotacion_y(grados):
    ang = np.deg2rad(grados)
    return np.array([[ np.cos(ang), 0, np.sin(ang), 0],
                     [0, 1, 0, 0],
                     [-np.sin(ang), 0, np.cos(ang), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def matriz_rotacion_z(grados):
    ang = np.deg2rad(grados)
    M = np.eye(4, dtype=np.float32)
    c, s = np.cos(ang), np.sin(ang)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s,  c
    return M

def matriz_traslacion(tx, ty, tz=0.0):
    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return M

def matriz_escala(s):
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = M[1, 1] = M[2, 2] = float(s)
    return M

# =====================================================
# Flechas 3D
# =====================================================
def crear_flecha_z(longitud, radio, color_rgba):
    """
    Crea una flecha orientada en +Z construida con cilindro + cono y material PBR.
    """
    altura_cil = longitud * 0.8
    altura_cono = longitud * 0.2
    cilindro = trimesh.creation.cylinder(radius=radio, height=altura_cil, sections=24)
    cilindro.apply_translation([0, 0, altura_cil/2.0])
    cono = trimesh.creation.cone(radius=radio*1.8, height=altura_cono, sections=24)
    cono.apply_translation([0, 0, altura_cil + altura_cono/2.0])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color_rgba, metallicFactor=0.0, roughnessFactor=0.7
    )
    combinado = trimesh.util.concatenate([cilindro, cono])
    return pyrender.Mesh.from_trimesh(combinado, material=material, smooth=True)

# =====================================================
# Dibujo de landmarks y overlay
# =====================================================
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_dibujo = mp.solutions.drawing_utils

def dibujar_landmarks_mano(img, lm, w_img, h_img, mostrar_ids=True):
    """
    Dibuja los landmarks y conexiones de la mano y devuelve lista de puntos (x, y) en píxeles.
    """
    mp_dibujo.draw_landmarks(img, lm, mp_manos.HAND_CONNECTIONS)
    puntos = []
    for i in range(21):
        x = int(lm.landmark[i].x * w_img)
        y = int(lm.landmark[i].y * h_img)
        puntos.append((x, y))
        cv2.circle(img, (x, y), RADIO_PUNTO_LM, COLOR_PUNTO_LM, -1)
        if mostrar_ids:
            cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ID_LM, 1, cv2.LINE_AA)
    return puntos

def dibujar_debug_pinch(img, punto_pulgar, punto_indice, punto_muñeca, distancia_pinch_px):
    """
    Dibuja indicadores para el gesto tipo 'pinch': línea, centro y textos auxiliares.
    """
    cv2.line(img, punto_pulgar, punto_indice, COLOR_LINEA_LM, 2)
    cx = int((punto_pulgar[0] + punto_indice[0]) / 2.0)
    cy = int((punto_pulgar[1] + punto_indice[1]) / 2.0)
    cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
    cv2.circle(img, punto_muñeca, 5, (0, 128, 255), -1)
    cv2.putText(img, f"pinch: {distancia_pinch_px:.1f}px", (punto_pulgar[0], punto_pulgar[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINEA_LM, 1, cv2.LINE_AA)

def poner_texto_info(img, texto, x=10, y=25, color=(0, 255, 0)):
    """
    Texto informativo sobre el frame de visualización.
    """
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

# =====================================================
# LOOP PRINCIPAL
# =====================================================
def principal():
    """
    Bucle principal: captura cámara, detección y refinamiento del marcador,
    manejo de gesto de mano, cálculo de pose y renderizado con horneado de transformaciones.
    """
    global SEGUIR_MARCADOR_EN_TRACK, ejecucion_activa

    # Abrir cámara
    captura = cv2.VideoCapture(ID_CAMARA)
    if not captura.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Arrancar hilo de YOLO
    threading.Thread(target=trabajador_yolo, daemon=True).start()

    caja_seguimiento = None
    esquinas_actuales = None
    gris_anterior = None
    contador_frames = 0

    # Renderer y nodos
    renderizador = None
    nodo_camara, nodo_luz = None, None
    transform_cv_a_gl = None

    # Ejes 3D
    longitud_ejes = ANCHO_REAL * FACTOR_LONGITUD_EJES
    radio_ejes = ANCHO_REAL * FACTOR_RADIO_EJES
    mallado_eje_x = crear_flecha_z(longitud_ejes, radio_ejes, [1.0, 0.2, 0.2, 1.0])
    mallado_eje_y = crear_flecha_z(longitud_ejes, radio_ejes, [0.2, 1.0, 0.2, 1.0])
    mallado_eje_z = crear_flecha_z(longitud_ejes, radio_ejes, [0.2, 0.4, 1.0, 1.0])
    T_z_a_x = matriz_rotacion_y(+90.0)
    T_z_a_y = matriz_rotacion_x(-90.0)

    # Máquina de estados y gesto
    modo = 'TRACK'  # 'TRACK' | 'GESTURE'
    agarrado = False
    mano_inicio_px = None
    angulo_inicio = 0.0
    distancia_ref_escala = 1.0
    angulo_suavizado = 0.0
    escala_suavizada = 1.0
    ts_ultima_mano = 0.0

    # Pose base (baked) y offset (si se usa)
    T_cam_modelo_base = None  # se inicializa al tener esquinas válidas
    T_offset = np.eye(4, dtype=np.float32)  # acumulador (se hornea al soltar)

    # Variables del frame del gesto
    distancia_pinch = None
    angulo_deg = 0.0
    dx = 0.0
    dy = 0.0

    # Reanclado
    tiempo_siguiente_reanclado = 0.0

    while True:
        ret, frame = captura.read()
        if not ret:
            break
        frame_mostrar = frame.copy()
        h_img, w_img = frame_mostrar.shape[:2]
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Intrínsecos locales (si no hay calibración)
        if matriz_camara is None or coef_distorsion is None:
            focal = float(w_img)
            matriz_camara_local = np.array([[focal, 0, w_img/2],
                                            [0, focal, h_img/2],
                                            [0, 0, 1]], dtype=np.float32)
            coef_distorsion_local = np.zeros(5, dtype=np.float32)
        else:
            matriz_camara_local = matriz_camara
            coef_distorsion_local = coef_distorsion
        fx = float(matriz_camara_local[0, 0])
        fy = float(matriz_camara_local[1, 1])
        cx = float(matriz_camara_local[0, 2])
        cy = float(matriz_camara_local[1, 2])

        # Configurar renderizador la primera vez
        if renderizador is None:
            renderizador = pyrender.OffscreenRenderer(viewport_width=w_img, viewport_height=h_img)
            nodo_camara = pyrender.Node(
                camera=pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=ZCERCA, zfar=ZLEJOS), matrix=np.eye(4)
            )
            nodo_luz = pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=INTENSIDAD_LUZ), matrix=np.eye(4)
            )
            transform_cv_a_gl = np.eye(4, dtype=np.float32)
            if INVERTIR_Y: transform_cv_a_gl[1, 1] = -1.0
            if INVERTIR_Z: transform_cv_a_gl[2, 2] = -1.0

        contador_frames += 1

        # 0) Mano + visualización (MediaPipe)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado_manos = manos.process(rgb)
        if resultado_manos.multi_hand_landmarks:
            lm = resultado_manos.multi_hand_landmarks[0]
            if ui['mostrar_landmarks_mano']:
                puntos_mano = dibujar_landmarks_mano(frame_mostrar, lm, w_img, h_img, mostrar_ids=True)
            else:
                puntos_mano = [(int(lm.landmark[i].x * w_img), int(lm.landmark[i].y * h_img)) for i in range(21)]
            punto_pulgar = puntos_mano[4]
            punto_indice = puntos_mano[8]
            punto_muñeca = puntos_mano[0]
            distancia_pinch = float(np.linalg.norm(np.array(punto_pulgar) - np.array(punto_indice)))
            centro_mano = np.array([(punto_pulgar[0] + punto_indice[0]) / 2.0, (punto_pulgar[1] + punto_indice[1]) / 2.0], dtype=np.float32)
            vec = np.array(punto_indice, dtype=np.float32) - np.array(punto_pulgar, dtype=np.float32)
            angulo_rad = float(np.arctan2(vec[1], vec[0]))
            distancia_mano = float(np.linalg.norm(centro_mano - np.array(punto_muñeca, dtype=np.float32)))

            dibujar_debug_pinch(frame_mostrar, punto_pulgar, punto_indice, punto_muñeca, distancia_pinch)
            ts_ultima_mano = time.time()

            # Entrar a GESTURE
            if modo == 'TRACK' and distancia_pinch < PINCH_ON and esquinas_actuales is not None:
                exito, rvec, tvec = cv2.solvePnP(puntos_objeto, esquinas_actuales, matriz_camara_local, coef_distorsion_local)
                if exito:
                    R_cam_obj, _ = cv2.Rodrigues(rvec)
                    t_marcador_modelo = np.array([0.0, 0.0, ELEVACION_MODELO_MM/1000.0], dtype=np.float32).reshape(3, 1)
                    t_cam_modelo = R_cam_obj @ t_marcador_modelo + tvec.reshape(3, 1)
                    T_cam_modelo_base = np.eye(4, dtype=np.float32)
                    T_cam_modelo_base[:3, :3] = R_cam_obj
                    T_cam_modelo_base[:3, 3] = t_cam_modelo.flatten()

                    modo = 'GESTURE'
                    agarrado = True
                    mano_inicio_px = centro_mano.copy()
                    angulo_inicio = angulo_rad
                    distancia_ref_escala = max(distancia_mano, 1e-6)
                    angulo_suavizado = 0.0
                    escala_suavizada = 1.0
                    dx, dy = 0.0, 0.0
                    angulo_deg = 0.0
                    # Al iniciar gesto, el offset deja de ser necesario
                    T_offset = np.eye(4, dtype=np.float32)

            # Actualizar gesto mientras está activo
            if modo == 'GESTURE':
                if distancia_pinch >= PINCH_OFF:
                    agarrado = False
                else:
                    agarrado = True
                    delta_px = centro_mano - mano_inicio_px
                    dx = float(delta_px[0]) * ESCALA_MOV_X
                    dy = float(delta_px[1]) * ESCALA_MOV_Y
                    angulo_deg_inst = (angulo_rad - angulo_inicio) * (180.0 / np.pi)
                    angulo_suavizado = EMA_ROT * angulo_suavizado + (1.0 - EMA_ROT) * angulo_deg_inst
                    escala_bruta = distancia_mano / distancia_ref_escala
                    escala_suavizada = EMA_ESC * escala_suavizada + (1.0 - EMA_ESC) * escala_bruta
                    escala_suavizada = float(np.clip(escala_suavizada, ESCALA_MIN, ESCALA_MAX))
                    angulo_deg = angulo_suavizado
        else:
            distancia_pinch = None

        # Volver a TRACK (y hornear en base) si no hay mano o se soltó
        if modo == 'GESTURE':
            sin_mano = (time.time() - ts_ultima_mano) > TIEMPO_ESPERA_MANO
            if sin_mano or not agarrado:
                # Horneado: T_base = T_base @ T_usuario
                T_usuario = matriz_traslacion(dx, dy, 0.0) @ matriz_rotacion_z(angulo_deg) @ matriz_escala(escala_suavizada)
                if T_cam_modelo_base is None:
                    T_cam_modelo_base = np.eye(4, dtype=np.float32)
                T_cam_modelo_base = T_cam_modelo_base @ T_usuario
                # Reset de variables del gesto
                dx, dy = 0.0, 0.0
                angulo_deg = 0.0
                escala_suavizada = 1.0
                modo = 'TRACK'
                agarrado = False
                # Enfriamiento para evitar reanclado inmediato
                tiempo_siguiente_reanclado = time.time() + ENFRIAMIENTO_REANCLADO_S

        # 1) Flujo óptico (solo TRACK)
        if modo == 'TRACK' and gris_anterior is not None and esquinas_actuales is not None:
            p0 = esquinas_actuales.reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(gris_anterior, frame_gris, p0, None, **parametros_lk)
            if p1 is not None and np.sum(st) == 4:
                nuevos_puntos = p1.reshape(4, 2)
                esquinas_actuales = nuevos_puntos
                min_x, max_x = np.min(nuevos_puntos[:, 0]), np.max(nuevos_puntos[:, 0])
                min_y, max_y = np.min(nuevos_puntos[:, 1]), np.max(nuevos_puntos[:, 1])
                caja_seguimiento = (int(min_x), int(min_y), int(max_x), int(max_y))
            else:
                esquinas_actuales = None
                caja_seguimiento = None

        # 2) YOLO + refinamiento (solo TRACK)
        if modo == 'TRACK':
            # Alimentar al hilo de YOLO cada cierto número de frames
            if contador_frames % 10 == 0 and not cola_frames.full():
                cola_frames.put(frame.copy())
            # Consumir resultados de YOLO
            if not cola_resultados_yolo.empty():
                caja_seguimiento = cola_resultados_yolo.get()
                esquinas_refinadas, _, _ = refinar_deteccion_robusta(frame, caja_seguimiento)
                if esquinas_refinadas is not None:
                    esquinas_actuales = esquinas_refinadas
            # Refinar continuamente sobre la caja actual
            if caja_seguimiento:
                esquinas_refinadas, img_debug, img_roi = refinar_deteccion_robusta(frame, caja_seguimiento)
                if MOSTRAR_VENTANAS_DEBUG and img_debug is not None:
                    cv2.imshow("DEBUG: Canny", img_debug)
                if esquinas_refinadas is not None:
                    esquinas_actuales = esquinas_refinadas
                    min_x, max_x = np.min(esquinas_actuales[:, 0]), np.max(esquinas_actuales[:, 0])
                    min_y, max_y = np.min(esquinas_actuales[:, 1]), np.max(esquinas_actuales[:, 1])
                    caja_seguimiento = (int(min_x), int(min_y), int(max_x), int(max_y))

        # 3) Render
        escena = pyrender.Scene(bg_color=np.array([0, 0, 0, 0], dtype=np.float32))
        escena.add_node(nodo_camara)
        escena.add_node(nodo_luz)

        if modo == 'TRACK':
            # Reanclado opcional de la base al marcador (si está habilitado y hay esquinas válidas)
            if SEGUIR_MARCADOR_EN_TRACK and esquinas_actuales is not None and time.time() > tiempo_siguiente_reanclado:
                exito, rvec, tvec = cv2.solvePnP(puntos_objeto, esquinas_actuales, matriz_camara_local, coef_distorsion_local)
                if exito:
                    R_cam_obj, _ = cv2.Rodrigues(rvec)
                    t_marcador_modelo = np.array([0.0, 0.0, ELEVACION_MODELO_MM/1000.0], dtype=np.float32).reshape(3, 1)
                    t_cam_modelo = R_cam_obj @ t_marcador_modelo + tvec.reshape(3, 1)
                    T_cam_modelo_base = np.eye(4, dtype=np.float32)
                    T_cam_modelo_base[:3, :3] = R_cam_obj
                    T_cam_modelo_base[:3, 3] = t_cam_modelo.flatten()

            # Si no hay base aún y tenemos esquinas, inicializar base
            if T_cam_modelo_base is None and esquinas_actuales is not None:
                exito, rvec, tvec = cv2.solvePnP(puntos_objeto, esquinas_actuales, matriz_camara_local, coef_distorsion_local)
                if exito:
                    R_cam_obj, _ = cv2.Rodrigues(rvec)
                    t_marcador_modelo = np.array([0.0, 0.0, ELEVACION_MODELO_MM/1000.0], dtype=np.float32).reshape(3, 1)
                    t_cam_modelo = R_cam_obj @ t_marcador_modelo + tvec.reshape(3, 1)
                    T_cam_modelo_base = np.eye(4, dtype=np.float32)
                    T_cam_modelo_base[:3, 3] = t_cam_modelo.flatten()
                    T_cam_modelo_base[:3, :3] = R_cam_obj

            # Dibujar contorno del plano si hay esquinas
            if esquinas_actuales is not None:
                cv2.polylines(frame_mostrar, [esquinas_actuales.astype(int)], True, (0, 255, 255), 2)

            # Render con base (baked)
            if T_cam_modelo_base is not None:
                T_gl = transform_cv_a_gl @ T_cam_modelo_base
                escena.add_node(pyrender.Node(mesh=mallado_eje_x, matrix=T_gl @ T_z_a_x))
                escena.add_node(pyrender.Node(mesh=mallado_eje_y, matrix=T_gl @ T_z_a_y))
                escena.add_node(pyrender.Node(mesh=mallado_eje_z, matrix=T_gl))

        elif modo == 'GESTURE':
            # Render con base + transformaciones del usuario
            if T_cam_modelo_base is not None:
                T_usuario = matriz_traslacion(dx, dy, 0.0) @ matriz_rotacion_z(angulo_deg) @ matriz_escala(escala_suavizada)
                T_gl = transform_cv_a_gl @ (T_cam_modelo_base @ T_usuario)
                escena.add_node(pyrender.Node(mesh=mallado_eje_x, matrix=T_gl @ T_z_a_x))
                escena.add_node(pyrender.Node(mesh=mallado_eje_y, matrix=T_gl @ T_z_a_y))
                escena.add_node(pyrender.Node(mesh=mallado_eje_z, matrix=T_gl))

        # Composición RGBA: superponer render sobre el frame de cámara
        try:
            imagen_color, _ = renderizador.render(escena, flags=pyrender.RenderFlags.RGBA)
        except Exception:
            imagen_color, _ = renderizador.render(escena, flags=pyrender.RenderFlags.RGBA)
        if imagen_color.shape[2] == 4:
            color_rgb = imagen_color[:, :, :3].astype(np.float32)
            alfa = (imagen_color[:, :, 3].astype(np.float32) / 255.0).reshape(h_img, w_img, 1)
            frame_rgb = cv2.cvtColor(frame_mostrar, cv2.COLOR_BGR2RGB).astype(np.float32)
            compuesto_rgb = color_rgb * alfa + frame_rgb * (1.0 - alfa)
            frame_mostrar = cv2.cvtColor(compuesto_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Overlay de info (modo y métricas del gesto)
        if ui['mostrar_info_gesto']:
            if distancia_pinch is not None:
                poner_texto_info(frame_mostrar, f"MODO: {modo} | pinch: {distancia_pinch:.1f}px", x=10, y=25, color=(0, 255, 0))
            else:
                poner_texto_info(frame_mostrar, f"MODO: {modo} | pinch: -", x=10, y=25, color=(0, 255, 0))
            if modo == 'GESTURE':
                poner_texto_info(frame_mostrar,
                                 f"dx: {dx:.3f} m  dy: {dy:.3f} m  angulo: {angulo_deg:.1f}°  escala: {escala_suavizada:.2f}",
                                 x=10, y=55, color=(0, 200, 255))

        # Actualizar imagen gris para flujo óptico
        gris_anterior = frame_gris.copy()

        # Mostrar ventana principal
        cv2.imshow(TITULO_VENTANA, frame_mostrar)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        elif tecla == ord('h'):
            ui['mostrar_landmarks_mano'] = not ui['mostrar_landmarks_mano']
        elif tecla == ord('i'):
            ui['mostrar_info_gesto'] = not ui['mostrar_info_gesto']
        elif tecla == ord('r'):
            # Reanclar base al marcador (si hay esquinas válidas)
            if esquinas_actuales is not None:
                exito, rvec, tvec = cv2.solvePnP(puntos_objeto, esquinas_actuales, matriz_camara_local, coef_distorsion_local)
                if exito:
                    R_cam_obj, _ = cv2.Rodrigues(rvec)
                    t_marcador_modelo = np.array([0.0, 0.0, ELEVACION_MODELO_MM/1000.0], dtype=np.float32).reshape(3, 1)
                    t_cam_modelo = R_cam_obj @ t_marcador_modelo + tvec.reshape(3, 1)
                    T_cam_modelo_base = np.eye(4, dtype=np.float32)
                    T_cam_modelo_base[:3, :3] = R_cam_obj
                    T_cam_modelo_base[:3, 3] = t_cam_modelo.flatten()
                    tiempo_siguiente_reanclado = time.time() + ENFRIAMIENTO_REANCLADO_S
        elif tecla == ord('f'):
            SEGUIR_MARCADOR_EN_TRACK = not SEGUIR_MARCADOR_EN_TRACK

    # Salida ordenada
    ejecucion_activa = False
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    principal()
