# VinylNet: Audio Representation via Spiral Coordinate Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**VinylNet** es una librería de Python diseñada para convertir señales de audio en representaciones visuales 2D continuas (topología espiral), superando las limitaciones de los espectrogramas tradicionales.

Esta herramienta permite tratar el audio como imágenes, facilitando el entrenamiento de modelos de IA generativa (GANs, VQ-VAEs) y permitiendo una reconstrucción de audio de alta fidelidad.

---

## ⚡ El Problema de la Fase: Comparativa

Los espectrogramas estándar descartan la fase, lo que genera artefactos metálicos al reconstruir el audio. **VinylNet preserva la fase en su geometría**, logrando una reconstrucción casi perfecta.

| Método | Representación Visual (Input para IA) | Calidad de Reconstrucción |
| :--- | :---: | :--- |
| **Estándar (Mel-Spec)** | ![Mel Spectrogram](assets_comparativa/std_mel_spec.png) | **Con Pérdida:** Sonido metálico y robótico.<br>🔊 [▶️ Escuchar Audio (Griffin-Lim)](assets_comparativa/std_reconstructed.wav) |
| **VinylNet (Propuesta)** | ![VinylNet Spiral](assets_comparativa/vinylnet_viz.png) | **Alta Fidelidad:** Sonido indistinguible del original.<br>🔊 [▶️ Escuchar Audio (VinylNet)](assets_comparativa/vinylnet_reconstructed.wav) |

---

## 📦 Instalación

Instala la librería directamente usando pip:

```bash
pip install vinylnet
```

🛠️ Guía de Uso
VinylNet está diseñado para ser extremadamente fácil de usar. Solo necesitas importar la librería y decidir si quieres trabajar en 8-bits (para IA) o 16-bits (para calidad).

1. Flujo para Deep Learning (Modo 8-bit)
Usa este modo para generar datasets masivos. Las imágenes son uint8 (0-255), pesan menos y son rápidas de procesar.

```python
import vinylnet

# --- CODIFICAR (Audio -> Imagen) ---
vinylnet.save_8bits(
    input_path="dataset/voz_original.mp3",
    output_path="dataset/train_001.png",
    duration=5.0,       # Cuantos segundos procesar
    resolution=512      # Tamaño de imagen (512x512)
)

# ... Tu IA entrena con 'train_001.png' ...

# --- DECODIFICAR (Imagen -> Audio) ---
vinylnet.load_8bits(
    input_img_path="dataset/train_001.png",
    output_wav_path="resultado_ia.wav"
)
```

2. Flujo de Alta Fidelidad (Modo 16-bit)
Usa este modo si necesitas calidad de estudio. Las imágenes son uint16 (0-65535) y conservan toda la profundidad del audio original.

```python
import vinylnet

# --- CODIFICAR HI-FI ---
vinylnet.save_16bits(
    input_path="master_track.flac",
    output_path="visual_master.png",
    duration=10.3,      
    resolution=1024     # Mayor resolución para capturar más detalle
)

# --- DECODIFICAR HI-FI ---
# El resultado será virtualmente idéntico al original
vinylnet.load_16bits(
    input_img_path="visual_master.png", 
    output_wav_path="restaurado.wav"
)
```

Explicación de Parámetros
input_path: Ruta al archivo de audio (soporta .mp3, .wav, .flac, .ogg).

output_path: Ruta donde se guardará la imagen PNG.

duration (float): Tiempo en segundos a procesar desde el inicio del audio.

Nota: VinylNet ajusta automáticamente la espiral para que estos segundos llenen el 100% del disco.

resolution (int): Tamaño de la imagen cuadrada (ej. 512, 768, 1024).

Recomendación: Usa 512 para IA y 1024 o más para Alta Fidelidad.

🚀 Características Técnicas
Algoritmo CLV (Constant Linear Velocity): Mapeo matemático determinista. El radio crece proporcionalmente al tiempo, asegurando densidad de datos constante.

Metadatos Estenográficos: La frecuencia de muestreo (sr) y número de muestras se guardan en los primeros píxeles de la imagen (Canal Azul). No necesitas archivos .json extra.

Canales Semánticos:

🔴 R: Audio Izquierdo (L)

🟢 G: Audio Derecho (R)

🔵 B: Magnitud visual (Ayuda a la IA a entender la geometría)

🗺️ Roadmap
[x] Core: Algoritmo de proyección espiral CLV.

[x] Soporte IO: Codificadores/Decodificadores 8-bit y 16-bit.

[ ] Integración: Dataloaders nativos para PyTorch.

[ ] Modelo: VQ-VAE pre-entrenado para compresión latente 64x64.

📄 Licencia
Este proyecto está bajo la licencia MIT.

Desarrollado con ❤️ y ondas sonoras.