# Manipulación de Objetos 3D en Realidad Aumentada mediante Reconocimiento de Gestos de la Mano
## Ing. Inteligencia Artificial, 2025
Angel Rodriguez Mitre<sup>1</sup>, Norberto Aziel Mejía Hernández<sup>1</sup>

¹ Instituto Politécnico Nacional  


Proyecto académico de **Ingeniería en Inteligencia Artificial** enfocado en el desarrollo de un sistema de **Realidad Aumentada (RA)** que permite **manipular objetos 3D virtuales** (traslación, rotación y escalado) utilizando **gestos naturales de la mano**, basados exclusivamente en **visión por computadora**, sin sensores físicos adicionales.

---

## 📌 Tabla de Contenidos

1. [Descripción General](#-descripción-general)
2. [Objetivos](#-objetivos)
3. [Alcance del Proyecto](#-alcance-del-proyecto)
4. [Arquitectura del Sistema](#-arquitectura-del-sistema)
5. [Tecnologías Utilizadas](#-tecnologías-utilizadas)
6. [Estructura del Repositorio](#-estructura-del-repositorio)
7. [Instalación y Configuración](#-instalación-y-configuración)
8. [Uso del Sistema](#-uso-del-sistema)
9. [Resultados Esperados](#-resultados-esperados)
10. [Limitaciones](#-limitaciones)
11. [Trabajo Futuro](#-trabajo-futuro)


---

## 🧠 Descripción General

Este proyecto propone un sistema interactivo de **Realidad Aumentada** capaz de proyectar y manipular objetos 3D en un entorno real, utilizando como interfaz principal los **gestos de la mano del usuario**, capturados mediante una cámara RGB convencional.

A diferencia de enfoques basados en sensores especializados o dispositivos hápticos, el sistema se apoya en técnicas de **visión por computadora**, detección de mano, análisis geométrico para interpretar gestos y superficies para realizar la proyección de manera eficiente en tiempo real.

---

## 🎯 Objetivos

### Objetivo General

Desarrollar un sistema de realidad aumentada que permita la manipulación intuitiva de objetos 3D mediante el reconocimiento de gestos de la mano usando visión por computadora en tiempo real.

### Objetivos Específicos

* Detectar y segmentar la mano del usuario.
* Reconocer gestos básicos para interacción (rotar, mover, escalar).
* Estimar la posición y orientación de la mano en el espacio.
* Integrar un motor gráfico para la visualización de objetos 3D en RA.
* Evaluar el desempeño del sistema en términos de precisión y latencia.

---

## 🔍 Alcance del Proyecto

✔ Interacción con **un objeto 3D virtual** a la vez.
✔ Uso de **una cámara RGB estándar**.
✔ Reconocimiento de gestos estáticos y dinámicos básicos.
✔ Procesamiento en tiempo real en un equipo de cómputo personal.

✖ No se contempla el uso de sensores de profundidad (LiDAR, Kinect).
✖ No se incluye retroalimentación háptica.

---

## 🧩 Arquitectura del Sistema

El sistema se divide en los siguientes módulos:

1. **Captura de Video**
   Obtención de imágenes en tiempo real desde la cámara.

2. **Procesamiento de Imagen**

   * Preprocesamiento
   * Detección de mano
   * Extracción de características

3. **Reconocimiento de Gestos**
   Clasificación del gesto y mapeo a acciones 3D.

4. **Motor de Realidad Aumentada**
   Renderizado y manipulación del objeto 3D.

5. **Interfaz de Usuario**
   Visualización del entorno real con superposición virtual.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python
* **Visión por Computadora:** OpenCV
* **Reconocimiento de Mano/Gestos:** MediaPipe / algoritmos personalizados
* **Gráficos 3D:** OpenGL / PyOpenGL / motor gráfico equivalente
* **Modelado 3D:** Blender (para objetos .obj / .stl)
* **Control de Versiones:** Git y GitHub

---

## 📁 Estructura del Repositorio

> La estructura presentada es orientativa 

```bash
├── App_web/            # Datos utilizados o generados por el proyecto
├── models/          # Modelos, configuraciones o archivos relacionados
├── src/             # Código fuente principal
│   └── main.py      # Punto de entrada del sistema
├── docs/            # Documentación y material de apoyo
├── requirements.txt # Dependencias del proyecto
├── README.md        # Documentación principal
└── LICENSE          # Licencia
```

---

## ⚙️ Instalación y Configuración

1. Clonar el repositorio:

```bash
git clone https://github.com/usuario/proyecto-ra-gestos.git
cd proyecto-ra-gestos
```

2. Crear entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```



---

## ▶️ Uso del Sistema

Ejecutar el archivo principal:

```bash
python src/main.py
```

Gestos soportados:

* Selección de objeto
* Traslación
* Escalado
* Rotación

---

## 📊 Resultados Esperados

* Manipulación de objetos 3D en tiempo real.
* Reconocimiento confiable de gestos básicos.
* Baja latencia entre gesto y respuesta visual.
* Interfaz intuitiva para usuarios sin entrenamiento previo.

---

## ⚠️ Limitaciones

* Sensible a condiciones de iluminación.
* Precisión limitada por el uso de cámara RGB.
* Dependencia del fondo y oclusiones parciales de la mano.

---

## 🚀 Trabajo Futuro

* Integración de modelos de aprendizaje profundo más robustos.
* Soporte para múltiples manos.
* Implementación de detección de superficies proyectables.
* Portabilidad a dispositivos móviles o web (WebAR).


---

📌 *Este proyecto fue desarrollado con fines académicos y de investigación.*




