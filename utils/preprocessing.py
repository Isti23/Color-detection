import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Membaca gambar, mengubah ukurannya (opsional), dan mengonversinya ke RGB.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Gagal membaca gambar di path: {image_path}")
    
    # Resize gambar untuk konsistensi dimensi (jika diperlukan)
    if target_size:
        img_bgr = cv2.resize(img_bgr, target_size)
        
    # Konversi BGR ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
