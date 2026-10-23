"""Segmentação por crescimento de regiões e morfologia matemática."""
from collections import deque
from typing import Tuple

import cv2
import numpy as np


def crescimento_regioes(img: np.ndarray, semente: Tuple[int, int], tolerancia: int = 20,
                        conectividade: int = 8) -> np.ndarray:
    """Crescimento de regiões a partir de uma semente (x, y). (Implementação manual, BFS.)

    Um vizinho é agregado se a diferença de intensidade em relação à MÉDIA ATUAL da
    região for <= tolerância. Em imagens coloridas usa-se a distância máxima entre canais.
    Retorna máscara binária (0/255) da região.
    """
    h, w = img.shape[:2]
    x0, y0 = semente
    if not (0 <= x0 < w and 0 <= y0 < h):
        raise ValueError("semente fora da imagem")

    ref = img.astype(np.float32)
    if ref.ndim == 2:
        ref = ref[:, :, None]

    visitado = np.zeros((h, w), bool)
    mascara = np.zeros((h, w), np.uint8)
    visitado[y0, x0] = True
    mascara[y0, x0] = 255
    soma = ref[y0, x0].copy()
    n = 1
    fila = deque([(x0, y0)])

    if conectividade == 4:
        vizinhos = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        vizinhos = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    while fila:
        x, y = fila.popleft()
        media = soma / n
        for dx, dy in vizinhos:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visitado[ny, nx]:
                visitado[ny, nx] = True
                if np.max(np.abs(ref[ny, nx] - media)) <= tolerancia:
                    mascara[ny, nx] = 255
                    soma += ref[ny, nx]
                    n += 1
                    fila.append((nx, ny))
    return mascara


def destacar_regiao(img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Sobrepõe a região segmentada em vermelho translúcido, com contorno."""
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cor = np.zeros_like(base)
    cor[:] = (0, 0, 255)
    mix = cv2.addWeighted(base, 0.6, cor, 0.4, 0)
    saida = np.where(mascara[:, :, None] > 0, mix, base)
    cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(saida, cont, -1, (0, 255, 255), 1)
    return saida


# ---------------------------------------------------------------- morfologia
FORMAS = {"retangulo": cv2.MORPH_RECT, "elipse": cv2.MORPH_ELLIPSE, "cruz": cv2.MORPH_CROSS}


def elemento_estruturante(k: int = 5, forma: str = "retangulo") -> np.ndarray:
    return cv2.getStructuringElement(FORMAS[forma], (k, k))


def erosao(img: np.ndarray, k: int = 5, forma: str = "retangulo", iteracoes: int = 1) -> np.ndarray:
    return cv2.erode(img, elemento_estruturante(k, forma), iterations=iteracoes)


def dilatacao(img: np.ndarray, k: int = 5, forma: str = "retangulo", iteracoes: int = 1) -> np.ndarray:
    return cv2.dilate(img, elemento_estruturante(k, forma), iterations=iteracoes)


def abertura(img: np.ndarray, k: int = 5, forma: str = "retangulo") -> np.ndarray:
    """Erosão seguida de dilatação: remove pequenos objetos/ruído claro."""
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, elemento_estruturante(k, forma))


def fechamento(img: np.ndarray, k: int = 5, forma: str = "retangulo") -> np.ndarray:
    """Dilatação seguida de erosão: fecha pequenos buracos escuros."""
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, elemento_estruturante(k, forma))
