import cv2
import numpy as np
import torch
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# calibracion
data = np.load("calibracion.npz")
K = data["K"]
dist = data["D"]

# yolo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo en uso:", device)

model = YOLO("yolov8n.pt").to(device)
target_class = "book"
class_names = model.names
target_id = [k for k, v in class_names.items() if v == target_class][0]

# parametros fisicos
book_width = 0.15
book_height = 0.22
cube_height = 0.05

cube_pts = np.float32([
    [0, 0, 0], [book_width, 0, 0], [book_width, book_height, 0], [0, book_height, 0],
    [0, 0, -cube_height], [book_width, 0, -cube_height],
    [book_width, book_height, -cube_height], [0, book_height, -cube_height]
])

# variables globales
cap = None
running = False
detecting = False
projecting = False
frame_display = None

# camara
def camera_loop():
    global running, frame_display

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # deteccion
        if detecting:
            results = model(frame, classes=[target_id], imgsz=960, device=device, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 3)
                    cv2.putText(frame, "📘 Libro detectado", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 150), 2)

                    # proyeccion 3d
                    if projecting:
                        img_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        obj_corners = np.float32([
                            [0, 0, 0],
                            [book_width, 0, 0],
                            [book_width, book_height, 0],
                            [0, book_height, 0]
                        ])
                        success, rvec, tvec = cv2.solvePnP(obj_corners, img_corners, K, dist)
                        if success:
                            imgpts, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, dist)
                            imgpts = np.int32(np.round(imgpts)).reshape(-1, 2)
                            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
                            for j in range(4):
                                frame = cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 2)
                            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)
                            cv2.putText(frame, "🧊 Cubo proyectado", (x1, y2 + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # mostrar frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        frame_display.imgtk = imgtk
        frame_display.configure(image=imgtk)
        frame_display.after(10, camera_loop)
        break

# botones
def start_camera():
    global cap, running
    if running:
        return
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    running = True
    status_label.config(text="📸 Cámara activa", foreground="#00FFAA")
    camera_loop()

def detect_objects():
    global detecting
    if not running:
        status_label.config(text="⚠️ Activa la cámara primero", foreground="#FF3333")
        return
    detecting = not detecting
    if detecting:
        status_label.config(text="🔍 Detección YOLO activada", foreground="#00FF99")
    else:
        status_label.config(text="🕹️ Detección detenida", foreground="#999999")

def project_3d():
    global projecting
    if not detecting:
        status_label.config(text="⚠️ Inicia detección antes de proyectar", foreground="#FF3333")
        return
    projecting = not projecting
    if projecting:
        status_label.config(text="🧊 Proyección 3D activa", foreground="#00BFFF")
    else:
        status_label.config(text="🕹️ Proyección detenida", foreground="#999999")

def stop_camera():
    global running, detecting, projecting
    running = False
    detecting = False
    projecting = False
    if cap:
        cap.release()
    status_label.config(text="🛑 Cámara detenida", foreground="#FF5555")
    frame_display.config(image='')

# interfaz
root = tk.Tk()
root.title("🧠 Interfaz Futurista - YOLO + Cubo 3D")
root.geometry("1000x700")
root.configure(bg="#0A0F0D")

# titulo
title_label = tk.Label(root, text="🧠 Sistema YOLO + Proyección 3D",
                       bg="#0A0F0D", fg="#00FF99", font=("Consolas", 20, "bold"))
title_label.pack(pady=10)

# video
video_frame = tk.Frame(root, bg="#101820", width=900, height=500)
video_frame.pack(pady=10)
frame_display = tk.Label(video_frame, bg="#000000")
frame_display.pack()

# botones
button_frame = tk.Frame(root, bg="#0A0F0D")
button_frame.pack(pady=15)

style = ttk.Style()
style.configure("TButton",
                font=("Consolas", 12, "bold"),
                background="#101820",
                foreground="#00FFAA",
                padding=10)

ttk.Button(button_frame, text="🚀 Iniciar Cámara", command=start_camera).grid(row=0, column=0, padx=12)
ttk.Button(button_frame, text="🔍 Detección de Objetos", command=detect_objects).grid(row=0, column=1, padx=12)
ttk.Button(button_frame, text="🧊 Proyectar Cubo 3D", command=project_3d).grid(row=0, column=2, padx=12)
ttk.Button(button_frame, text="🛑 Detener Cámara", command=stop_camera).grid(row=0, column=3, padx=12)

# estado
status_label = tk.Label(root, text="🕹️ En espera",
                        bg="#0A0F0D", fg="#999999", font=("Consolas", 14, "bold"))
status_label.pack(pady=15)

# pie
footer = tk.Label(root, text="Futuristic Interface - YOLOv8 + OpenCV + Tkinter",
                  bg="#0A0F0D", fg="#00FF99", font=("Consolas", 10))
footer.pack(side=tk.BOTTOM, pady=5)

root.mainloop()

