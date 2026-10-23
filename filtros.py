"""Filtragem espacial e detecção de bordas."""
import cv2
import numpy as np


def filtro_media(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Média (box filter): suaviza ruído, mas borra as bordas."""
    return cv2.blur(img, (k, k))


def filtro_mediana(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Mediana: excelente contra ruído sal-e-pimenta, preserva melhor as bordas."""
    return cv2.medianBlur(img, k)


def filtro_gaussiano(img: np.ndarray, k: int = 5, sigma: float = 0) -> np.ndarray:
    """Gaussiano: média ponderada (sigma=0 => calculado a partir de k)."""
    return cv2.GaussianBlur(img, (k, k), sigma)


def _cinza(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sobel(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Magnitude do gradiente de Sobel, normalizada para 0-255."""
    g = _cinza(img)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=k)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max() * 255
    return mag.astype(np.uint8)


def canny(img: np.ndarray, limiar1: int = 100, limiar2: int = 200) -> np.ndarray:
    """Canny com leve suavização gaussiana prévia."""
    g = cv2.GaussianBlur(_cinza(img), (5, 5), 0)
    return cv2.Canny(g, limiar1, limiar2)


def adicionar_ruido_sal_pimenta(img: np.ndarray, proporcao: float = 0.03,
                                semente: int = 0) -> np.ndarray:
    """Utilitário para demonstrar a diferença média x mediana."""
    rng = np.random.default_rng(semente)
    saida = img.copy()
    mascara = rng.random(img.shape[:2])
    saida[mascara < proporcao / 2] = 0
    saida[mascara > 1 - proporcao / 2] = 255
    return saida
