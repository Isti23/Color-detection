import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_colors(image_rgb, k=3):
    """
    Ekstraksi k warna dominan menggunakan K-Means.
    Mengembalikan array 1D berukuran (k * 3).
    """
    # Ubah gambar menjadi list pixel (N, 3)
    pixels = image_rgb.reshape(-1, 3)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    # Ambil centroid (warna dominan)
    colors = kmeans.cluster_centers_  # bentuknya (k, 3)
    
    # Hitung jumlah pixel di setiap cluster untuk mengurutkan dominansi
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # Urutkan warna dari yang paling dominan ke yang kurang dominan
    dominant_indices = np.argsort(-counts)
    sorted_colors = colors[dominant_indices]
    
    # Ubah menjadi tipe data integer (nilai pixel 0-255)
    sorted_colors = np.round(sorted_colors).astype(int)
    
    # Ratakan (flatten) menjadi 1D array (9 fitur)
    features = sorted_colors.flatten()
    return features
