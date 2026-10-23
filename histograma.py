"""Histogramas, equalização, binarização e quantização."""
from typing import Tuple

import cv2
import numpy as np


def _cinza(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calcular_histograma(canal: np.ndarray) -> np.ndarray:
    """Histograma de 256 bins de um canal uint8. (Manual com NumPy.)"""
    return np.bincount(canal.ravel(), minlength=256)


def desenhar_histogramas(img: np.ndarray) -> np.ndarray:
    """Gera uma imagem (BGR) com o histograma de cinza e dos canais B, G, R."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, 2 if img.ndim == 3 else 1, figsize=(11, 3.6), squeeze=False)
    eixos = eixos[0]
    h = calcular_histograma(_cinza(img))
    eixos[0].bar(range(256), h, color="gray", width=1.0)
    eixos[0].set_title("Histograma (cinza)")
    eixos[0].set_xlim(0, 255)
    if img.ndim == 3:
        for i, (cor, nome) in enumerate([("b", "B"), ("g", "G"), ("r", "R")]):
            eixos[1].plot(calcular_histograma(img[:, :, i]), color=cor, label=nome)
        eixos[1].set_title("Histograma por canal")
        eixos[1].set_xlim(0, 255)
        eixos[1].legend()
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def equalizar_cinza(g: np.ndarray) -> np.ndarray:
    """Equalização de histograma pela CDF. (Implementação manual.)
    s_k = round((L-1) * (cdf(k) - cdf_min) / (N - cdf_min))"""
    hist = calcular_histograma(g)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)][0]
    n = g.size
    if n == cdf_min:  # imagem de cor única
        return g.copy()
    tabela = np.round((cdf - cdf_min) / (n - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    return tabela[g]


def equalizar(img: np.ndarray) -> np.ndarray:
    """Em imagem colorida, equaliza só o canal V do HSV (preserva as cores)."""
    if img.ndim == 2:
        return equalizar_cinza(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = equalizar_cinza(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def analisar_canais(img: np.ndarray) -> str:
    """Resumo estatístico por canal (média, desvio, mín., máx.)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    linhas = ["canal   média   desvio  mín  máx"]
    for i, nome in enumerate("BGR"):
        c = img[:, :, i]
        linhas.append(f"  {nome}   {c.mean():6.1f}  {c.std():6.1f}  {c.min():3d}  {c.max():3d}")
    return "\n".join(linhas)


# ---------------------------------------------------------------- binarização
def limiar_fixo(img: np.ndarray, limiar: int = 127) -> np.ndarray:
    """Limiarização simples: 255 se pixel > limiar, senão 0. (Manual com NumPy.)"""
    return np.where(_cinza(img) > limiar, 255, 0).astype(np.uint8)


def limiar_adaptativo(img: np.ndarray, bloco: int = 31, c: int = 5) -> np.ndarray:
    """Limiar gaussiano local: bom para iluminação irregular. Bloco deve ser ímpar."""
    bloco = bloco if bloco % 2 == 1 else bloco + 1
    return cv2.adaptiveThreshold(_cinza(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, max(3, bloco), c)


def limiar_otsu(img: np.ndarray) -> Tuple[int, np.ndarray]:
    """Otsu: escolhe o limiar que maximiza a variância entre classes."""
    t, bw = cv2.threshold(_cinza(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), bw


# ---------------------------------------------------------------- quantização
def quantizar(img: np.ndarray, niveis: int = 4) -> np.ndarray:
    """Reduz cada canal a `niveis` valores igualmente espaçados. (Manual com NumPy.)
    Funciona em cinza e em cor."""
    if niveis < 2:
        raise ValueError("use pelo menos 2 níveis")
    passo = 256 / niveis
    idx = np.floor(img.astype(np.float32) / passo)
    return np.clip(np.round(idx * 255 / (niveis - 1)), 0, 255).astype(np.uint8)
