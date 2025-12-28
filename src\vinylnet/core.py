import librosa
import numpy as np
import cv2
import struct
import os
import soundfile as sf  # <--- IMPORTANTE: Necesario para guardar el audio reconstruido

def _get_spiral_coords_forced(n_samples, center, radius_min, radius_max):
    """
    Función interna: Calcula la espiral determinista (CLV).
    """
    coords = np.zeros((n_samples, 2), dtype=np.int32)
    
    R2_max = radius_max**2
    R2_min = radius_min**2
    R_diff = R2_max - R2_min
    
    # Vectorización
    indices = np.arange(n_samples)
    if n_samples > 0:
        t = indices / float(n_samples)
    else:
        return coords, np.array([])

    # Fórmula de Área Constante
    radii = np.sqrt(R2_min + (t * R_diff))
    
    # Ángulos acumulativos (con epsilon para evitar div/0)
    d_thetas = 1.0 / (radii + 1e-9)
    thetas = np.cumsum(d_thetas)
    
    # Cartesianas
    x_vals = center[0] + radii * np.cos(thetas)
    y_vals = center[1] + radii * np.sin(thetas)
    
    coords[:, 0] = x_vals.astype(np.int32)
    coords[:, 1] = y_vals.astype(np.int32)
            
    return coords, radii

def _process_audio(input_path, duration):
    """Carga y valida el audio."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo no existe: {input_path}")
        
    y, sr = librosa.load(input_path, mono=False, duration=duration)
    
    if y.ndim == 1:
        y = np.vstack((y, y))
        
    return y, sr

# --- ENCODERS (Audio -> Imagen) ---

def save_8bits(input_path, output_path, duration=10.3, resolution=768):
    """Guarda audio como PNG 8-bit (0-255)."""
    img_size = resolution
    center = (img_size // 2, img_size // 2)
    radius_min = 50
    radius_max = img_size // 2
    
    print(f"VinylNet: Codificando {input_path} -> 8-bit...")
    
    y, sr = _process_audio(input_path, duration)
    n_samples = y.shape[1]

    def to_uint8(arr):
        arr = np.clip(arr, -1.0, 1.0)
        norm = (arr + 1.0) / 2.0
        return (norm * 255.0).astype(np.uint8)

    pixel_L = to_uint8(y[0])
    pixel_R = to_uint8(y[1])
    
    coords, radii = _get_spiral_coords_forced(n_samples, center, radius_min, radius_max)
    
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < img_size) & \
                 (coords[:, 1] >= 0) & (coords[:, 1] < img_size)
    
    xs = coords[valid_mask, 0]
    ys = coords[valid_mask, 1]
    r_vals = radii[valid_mask]
    
    pixel_B = ((np.sin(r_vals * 0.1) + 1) * 0.5 * 255.0).astype(np.uint8)
    
    img[ys, xs, 2] = pixel_L[valid_mask]
    img[ys, xs, 1] = pixel_R[valid_mask]
    img[ys, xs, 0] = pixel_B

    # Header 8-bit (8 bytes = 8 pixels)
    header = struct.pack('II', n_samples, sr)
    header_pixels = np.frombuffer(header, dtype=np.uint8)
    img[0, 0:8, 0] = 0 
    img[0, 0:len(header_pixels), 0] = header_pixels
    
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"-> Guardado: {output_path}")

def save_16bits(input_path, output_path, duration=10.3, resolution=768):
    """Guarda audio como PNG 16-bit (0-65535)."""
    img_size = resolution
    center = (img_size // 2, img_size // 2)
    radius_min = 50
    radius_max = img_size // 2
    
    print(f"VinylNet: Codificando {input_path} -> 16-bit...")
    
    y, sr = _process_audio(input_path, duration)
    n_samples = y.shape[1]

    def to_uint16(arr):
        arr = np.clip(arr, -1.0, 1.0)
        norm = (arr + 1.0) / 2.0
        return (norm * 65535.0).astype(np.uint16)

    pixel_L = to_uint16(y[0])
    pixel_R = to_uint16(y[1])
    
    coords, radii = _get_spiral_coords_forced(n_samples, center, radius_min, radius_max)
    
    img = np.zeros((img_size, img_size, 3), dtype=np.uint16)
    
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < img_size) & \
                 (coords[:, 1] >= 0) & (coords[:, 1] < img_size)
    
    xs = coords[valid_mask, 0]
    ys = coords[valid_mask, 1]
    r_vals = radii[valid_mask]
    
    pixel_B = ((np.sin(r_vals * 0.1) + 1) * 0.5 * 65535.0).astype(np.uint16)
    
    img[ys, xs, 2] = pixel_L[valid_mask]
    img[ys, xs, 1] = pixel_R[valid_mask]
    img[ys, xs, 0] = pixel_B

    # Header 16-bit (8 bytes = 4 pixels)
    header = struct.pack('II', n_samples, sr)
    header_pixels = np.frombuffer(header, dtype=np.uint16)
    img[0, 0:4, 0] = 0 
    img[0, 0:len(header_pixels), 0] = header_pixels
    
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"-> Guardado: {output_path}")

# --- DECODERS (Imagen -> Audio) ---

def load_8bits(input_img_path, output_wav_path):
    """Reconstruye audio desde un PNG de 8 bits."""
    print(f"VinylNet: Decodificando {input_img_path} (8-bit)...")
    
    # Leer como UNCHANGED para obtener bytes crudos
    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {input_img_path}")
    
    if img.dtype != np.uint8:
        raise ValueError("La imagen no es de 8 bits (uint8).")

    img_size = img.shape[0]
    center = (img_size // 2, img_size // 2)
    radius_min = 50
    radius_max = img_size // 2

    # 1. Leer Header (8 pixels = 8 bytes)
    try:
        header_pixels = img[0, 0:8, 0]
        packed_data = header_pixels.tobytes()
        n_samples, sr = struct.unpack('II', packed_data)
        print(f"   Metadatos: {n_samples} muestras a {sr} Hz")
    except Exception as e:
        print(f"   Error leyendo metadatos: {e}")
        return

    # 2. Recalcular Geometría
    coords, _ = _get_spiral_coords_forced(n_samples, center, radius_min, radius_max)

    # 3. Filtrar coordenadas válidas
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < img_size) & \
                 (coords[:, 1] >= 0) & (coords[:, 1] < img_size)
    
    xs = coords[valid_mask, 0]
    ys = coords[valid_mask, 1]

    # 4. Extraer Audio (OpenCV es BGR: R=2, G=1)
    pixel_L = img[ys, xs, 2] 
    pixel_R = img[ys, xs, 1]

    # 5. Des-normalizar 8-bit
    def from_uint8(arr):
        norm = arr.astype(np.float32) / 255.0
        return (norm * 2.0) - 1.0

    audio_L = from_uint8(pixel_L)
    audio_R = from_uint8(pixel_R)
    
    stereo_audio = np.array([audio_L, audio_R]).T
    sf.write(output_wav_path, stereo_audio, sr)
    print(f"-> Audio reconstruido: {output_wav_path}")

def load_16bits(input_img_path, output_wav_path):
    """Reconstruye audio desde un PNG de 16 bits."""
    print(f"VinylNet: Decodificando {input_img_path} (16-bit)...")
    
    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {input_img_path}")

    if img.dtype != np.uint16:
        raise ValueError("La imagen no es de 16 bits (uint16).")

    img_size = img.shape[0]
    center = (img_size // 2, img_size // 2)
    radius_min = 50
    radius_max = img_size // 2

    # 1. Leer Header (4 pixels = 8 bytes en 16-bit)
    try:
        header_pixels = img[0, 0:4, 0]
        packed_data = header_pixels.tobytes()
        n_samples, sr = struct.unpack('II', packed_data)
        print(f"   Metadatos: {n_samples} muestras a {sr} Hz")
    except Exception as e:
        print(f"   Error leyendo metadatos: {e}")
        return

    # 2. Recalcular Geometría
    coords, _ = _get_spiral_coords_forced(n_samples, center, radius_min, radius_max)

    # 3. Filtrar
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < img_size) & \
                 (coords[:, 1] >= 0) & (coords[:, 1] < img_size)
    
    xs = coords[valid_mask, 0]
    ys = coords[valid_mask, 1]

    # 4. Extraer
    pixel_L = img[ys, xs, 2] 
    pixel_R = img[ys, xs, 1]

    # 5. Des-normalizar 16-bit
    def from_uint16(arr):
        norm = arr.astype(np.float32) / 65535.0
        return (norm * 2.0) - 1.0

    audio_L = from_uint16(pixel_L)
    audio_R = from_uint16(pixel_R)
    
    stereo_audio = np.array([audio_L, audio_R]).T
    sf.write(output_wav_path, stereo_audio, sr)
    print(f"-> Audio reconstruido: {output_wav_path}")