import cv2
import torch
import numpy as np
import trimesh
import mediapipe as mp
import time
import csv
from queue import Queue
from threading import Thread
from ultralytics import YOLO

# ==============================
# CONFIGURACIÓN
# ==============================
CAMERA_ID = 1
MODEL_PATH = "yolov8s-seg.pt"
TARGET_CLASS = "book"
CONF_THRES = 0.25
YOLO_BATCH_SIZE = 4
YOLO_INTERVAL_FRAMES = 5  # Ejecutar YOLO cada 5 frames

BOOK_WIDTH = 21.0
BOOK_HEIGHT = 29.7
GLB_PATH = "cubo_150x150.glb"
MODEL_SCALE = 5.0

CSV_FILE = "benchmark_AR.csv"

# ==============================
# DISPOSITIVO
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
print("Dispositivo usado:", DEVICE)

# ==============================
# MODELOS
# ==============================
# Carga YOLOv8 directamente con ultralytics
model = YOLO(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ==============================
# MODELO 3D
# ==============================
mesh = trimesh.load(GLB_PATH, force="mesh")
vertices = mesh.vertices * MODEL_SCALE
faces = mesh.faces
vertices[:,0] -= vertices[:,0].mean()
vertices[:,1] -= vertices[:,1].mean()
vertices[:,2] -= vertices[:,2].min()

# ==============================
# UTILIDADES
# ==============================
def ordenar_esquinas(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def extraer_contorno(mask):
    mask = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt if cv2.contourArea(cnt) > 3000 else None

# ==============================
# HILO YOLO ASÍNCRONO CON BATCH
# ==============================
class YOLOThread(Thread):
    def __init__(self, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True

    def run(self):
        while self.running:
            batch = []
            while len(batch) < YOLO_BATCH_SIZE and not self.frame_queue.empty():
                batch.append(self.frame_queue.get())
            if batch:
                results = model(batch, conf=CONF_THRES)
                for res in results:
                    self.result_queue.put(res)

# ==============================
# FUNCION PRINCIPAL AR CON CSV
# ==============================
def main_ar():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    focal = w
    K = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float32)
    dist = np.zeros((4,1))
    obj_pts = np.array([[0,0,0],[BOOK_WIDTH,0,0],[BOOK_WIDTH,BOOK_HEIGHT,0],[0,BOOK_HEIGHT,0]], dtype=np.float32)

    tvec_saved = None
    rvec_saved = None
    grabbed = False
    hand_start_px = None
    tvec_start = None
    angle_start = 0
    scale_start = 1.0
    MOVE_SCALE_X = BOOK_WIDTH/400
    MOVE_SCALE_Y = BOOK_HEIGHT/400
    alpha = 0.7
    tvec_smooth = None
    angle_smooth = 0.0
    scale_smooth = 1.0

    frame_queue = Queue(maxsize=YOLO_BATCH_SIZE*2)
    result_queue = Queue(maxsize=YOLO_BATCH_SIZE*2)

    yolo_thread = YOLOThread(frame_queue, result_queue)
    yolo_thread.start()

    frame_id = 0
    fps_counter = 0
    fps_start = time.time()

    # CSV setup
    with open(CSV_FILE, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'FPS', 'YOLO_Run', 'Time_YOLO'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            fps_counter += 1
            yolo_run = False
            t_yolo = 0.0

            # Ejecutar YOLO cada YOLO_INTERVAL_FRAMES
            if frame_id % YOLO_INTERVAL_FRAMES == 0 and frame_queue.qsize() < YOLO_BATCH_SIZE*2:
                t0 = time.time()
                frame_queue.put(frame.copy())
                yolo_run = True
                t_yolo = time.time()-t0

            # Último resultado YOLO
            res = None
            while not result_queue.empty():
                res = result_queue.get()

            # =========================
            # DETECCIÓN LIBRO
            # =========================
            if res and res.masks:
                for i, box in enumerate(res.boxes):
                    if model.names[int(box.cls[0])] != TARGET_CLASS:
                        continue
                    mask = res.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask,(w,h))
                    cnt = extraer_contorno(mask)
                    if cnt is None:
                        continue
                    rect = cv2.minAreaRect(cnt)
                    box2d = ordenar_esquinas(cv2.boxPoints(rect))
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, box2d, K, dist)
                    if ok:
                        rvec_saved = rvec
                        if tvec_saved is None:
                            tvec_saved = tvec.copy()

            # =========================
            # DETECCIÓN MANO
            # =========================
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            if hand_res.multi_hand_landmarks and tvec_saved is not None:
                lm = hand_res.multi_hand_landmarks[0]
                thumb = np.array([lm.landmark[4].x*w, lm.landmark[4].y*h])
                index = np.array([lm.landmark[8].x*w, lm.landmark[8].y*h])
                wrist = np.array([lm.landmark[0].x*w, lm.landmark[0].y*h])
                hand_center = (thumb+index)/2
                pinch_dist = np.linalg.norm(thumb-index)
                vec = index-thumb
                angle = np.arctan2(vec[1],vec[0])
                dist_hand = np.linalg.norm(hand_center-wrist)
                if pinch_dist<40:
                    if not grabbed:
                        grabbed=True
                        hand_start_px=hand_center.copy()
                        tvec_start=tvec_saved.copy()
                        angle_start=angle
                        scale_start=1.0
                        scale_ref_dist=dist_hand
                    else:
                        delta = hand_center-hand_start_px
                        tvec_saved[0][0] = tvec_start[0][0]+delta[0]*MOVE_SCALE_X
                        tvec_saved[1][0] = tvec_start[1][0]+delta[1]*MOVE_SCALE_Y
                        angle_smooth=angle_smooth*0.7+0.3*((angle-angle_start)*(180/np.pi))
                        scale_smooth=scale_smooth*0.7+0.3*(scale_start*dist_hand/scale_ref_dist)
                else:
                    grabbed=False
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # =========================
            # SUAVIZADO
            # =========================
            if tvec_saved is not None:
                if tvec_smooth is None:
                    tvec_smooth = tvec_saved.copy()
                else:
                    tvec_smooth = alpha*tvec_smooth + (1-alpha)*tvec_saved

            # =========================
            # PROYECCIÓN 3D
            # =========================
            if rvec_saved is not None and tvec_smooth is not None:
                verts = vertices.copy()*scale_smooth
                verts[:,2]*=-1
                Rz = trimesh.transformations.rotation_matrix(np.deg2rad(angle_smooth), [0,0,1])
                verts_h = np.hstack((verts,np.ones((verts.shape[0],1))))
                verts = (Rz @ verts_h.T).T[:,:3]
                imgpts,_ = cv2.projectPoints(verts.astype(np.float32), rvec_saved, tvec_smooth, K, dist)
                imgpts = imgpts.reshape(-1,2).astype(int)
                for face in faces:
                    for j in range(3):
                        p1 = tuple(imgpts[face[j]])
                        p2 = tuple(imgpts[face[(j+1)%3]])
                        cv2.line(frame,p1,p2,(0,255,255),2)

            # Mostrar FPS
            if fps_counter>=30:
                fps_now = fps_counter/(time.time()-fps_start)
                fps_start=time.time()
                fps_counter=0
            else:
                fps_now=None
            if fps_now:
                cv2.putText(frame,f"FPS: {fps_now:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # Guardar en CSV
            with open(CSV_FILE, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([frame_id, fps_now if fps_now else '', yolo_run, t_yolo])

            cv2.imshow("AR — Cubo 3D Optimizado con CSV", frame)
            if cv2.waitKey(1)&0xFF==27:
                break

    yolo_thread.running=False
    cap.release()
    cv2.destroyAllWindows()
    print(f"Benchmark guardado en {CSV_FILE}")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main_ar()
