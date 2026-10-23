"""Operações básicas em imagens (funções puras: recebem array, devolvem array)."""
from typing import Tuple

import cv2
import numpy as np


def inverter_cores(img: np.ndarray) -> np.ndarray:
    """Negativo da imagem: s = 255 - r. (Implementação manual com NumPy.)"""
    return (255 - img.astype(np.int16)).astype(np.uint8)


def brilho_contraste(img: np.ndarray, alfa: float = 1.0, beta: int = 0) -> np.ndarray:
    """g(x,y) = alfa * f(x,y) + beta, com saturação em [0, 255].
    alfa controla o contraste (>1 aumenta); beta controla o brilho. (Manual com NumPy.)"""
    saida = alfa * img.astype(np.float32) + beta
    return np.clip(saida, 0, 255).astype(np.uint8)


def _igualar_tamanho(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Redimensiona b para o tamanho de a e iguala o número de canais."""
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if a.ndim == 3 and b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    elif a.ndim == 2 and b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return b


def somar_imagens(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Soma com saturação (evita overflow de uint8)."""
    b = _igualar_tamanho(a, b)
    return np.clip(a.astype(np.int16) + b.astype(np.int16), 0, 255).astype(np.uint8)


def subtrair_imagens(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Subtração com saturação em 0 (não gera valores negativos/wrap-around)."""
    b = _igualar_tamanho(a, b)
    return np.clip(a.astype(np.int16) - b.astype(np.int16), 0, 255).astype(np.uint8)


def para_cinza(img: np.ndarray) -> np.ndarray:
    """Conversão para níveis de cinza: Y = 0.299 R + 0.587 G + 0.114 B."""
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr_para_hsv(img: np.ndarray) -> np.ndarray:
    """BGR -> HSV (H em [0,179], S e V em [0,255] no OpenCV)."""
    return cv2.cvtColor(_garantir_bgr(img), cv2.COLOR_BGR2HSV)


def hsv_para_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def _garantir_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def redimensionar(img: np.ndarray, largura: int = None, altura: int = None,
                  escala: float = None) -> np.ndarray:
    """Redimensiona por escala ou por dimensões. Interpolação adequada ao sentido."""
    h, w = img.shape[:2]
    if escala is not None:
        largura, altura = max(1, int(w * escala)), max(1, int(h * escala))
    if not largura or not altura:
        raise ValueError("informe escala ou largura e altura")
    reduzindo = largura * altura < w * h
    interp = cv2.INTER_AREA if reduzindo else cv2.INTER_CUBIC
    return cv2.resize(img, (largura, altura), interpolation=interp)


def recortar_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Recorta a região (x, y, w, h), limitando às bordas da imagem."""
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI fora dos limites da imagem")
    return img[y0:y1, x0:x1].copy()


def separar_canais(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (B, G, R) em níveis de cinza."""
    b, g, r = cv2.split(_garantir_bgr(img))
    return b, g, r
