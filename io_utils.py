"""Entrada/saída, visualização e utilidades de interação (menu, mouse).

Convenção: todas as imagens são arrays NumPy uint8 no formato BGR (padrão do OpenCV)
ou em níveis de cinza (2D).
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

PASTA_SAIDA = "saida"


@dataclass
class Estado:
    """Estado do programa: imagem original (nunca alterada), imagem de trabalho
    e o último resultado produzido por uma operação."""
    original: Optional[np.ndarray] = None
    trabalho: Optional[np.ndarray] = None
    resultados: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    caminho: str = ""

    def tem_imagem(self) -> bool:
        return self.trabalho is not None

    def registrar(self, nome: str, img: np.ndarray) -> None:
        """Guarda um resultado (substitui um resultado de mesmo nome)."""
        self.resultados = [(n, i) for n, i in self.resultados if n != nome]
        self.resultados.append((nome, img.copy()))

    def ultimo(self) -> Optional[np.ndarray]:
        return self.resultados[-1][1] if self.resultados else None


# ---------------------------------------------------------------- leitura/escrita
def abrir_imagem(caminho: str) -> Optional[np.ndarray]:
    """Lê imagem do disco. Suporta caminhos com acentos (Windows). Retorna None em erro."""
    caminho = caminho.strip().strip('"').strip("'")
    if not os.path.isfile(caminho):
        print(f"[erro] Arquivo não encontrado: {caminho}")
        return None
    try:
        dados = np.fromfile(caminho, dtype=np.uint8)
        img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    except Exception as exc:  # arquivo corrompido, etc.
        print(f"[erro] Falha ao ler a imagem: {exc}")
        return None
    if img is None:
        print("[erro] O arquivo não é uma imagem válida.")
    return img


def salvar_imagem(img: np.ndarray, nome: str, pasta: str = PASTA_SAIDA) -> Optional[str]:
    """Salva imagem em `pasta`. Se o nome não tiver extensão, usa .png."""
    os.makedirs(pasta, exist_ok=True)
    if not os.path.splitext(nome)[1]:
        nome += ".png"
    caminho = os.path.join(pasta, nome)
    ext = os.path.splitext(caminho)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        print("[erro] Formato de saída não suportado.")
        return None
    buf.tofile(caminho)
    return caminho


def duplicar(img: np.ndarray) -> np.ndarray:
    """Cópia independente (processar a cópia não altera a original)."""
    return img.copy()


# ---------------------------------------------------------------- visualização
def _para_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def montar_lado_a_lado(imagens: List[Tuple[str, np.ndarray]], altura: int = 400) -> np.ndarray:
    """Cria um painel horizontal com as imagens (mesma altura) e legenda."""
    painel = []
    for titulo, img in imagens:
        img = _para_bgr(img)
        h, w = img.shape[:2]
        nova_w = max(1, int(w * altura / h))
        img = cv2.resize(img, (nova_w, altura))
        faixa = np.full((28, nova_w, 3), 30, np.uint8)
        cv2.putText(faixa, titulo, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        painel.append(np.vstack([faixa, img]))
    return np.hstack(painel)


def mostrar(titulo: str, img: np.ndarray) -> None:
    """Mostra uma janela e espera uma tecla."""
    cv2.imshow(titulo, img)
    print("(pressione qualquer tecla na janela da imagem para continuar)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def mostrar_original_e_processadas(estado: Estado) -> None:
    """Mostra simultaneamente a original e todos os resultados registrados."""
    if estado.original is None:
        print("[aviso] Nenhuma imagem carregada.")
        return
    itens = [("Original", estado.original)]
    if estado.trabalho is not None and not np.array_equal(estado.trabalho, estado.original):
        itens.append(("Trabalho", estado.trabalho))
    itens += estado.resultados[-3:]  # evita painéis gigantes
    mostrar("Original x Processadas", montar_lado_a_lado(itens))


# ---------------------------------------------------------------- entrada de dados
def ler_int(msg: str, padrao: int, minimo: int = None, maximo: int = None) -> int:
    """Lê inteiro com valor padrão (Enter) e validação de faixa."""
    while True:
        txt = input(f"{msg} [{padrao}]: ").strip()
        if not txt:
            return padrao
        try:
            v = int(txt)
        except ValueError:
            print("  valor inválido, digite um número inteiro.")
            continue
        if (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
            print(f"  valor fora da faixa [{minimo}, {maximo}].")
            continue
        return v


def ler_float(msg: str, padrao: float, minimo: float = None, maximo: float = None) -> float:
    while True:
        txt = input(f"{msg} [{padrao}]: ").strip().replace(",", ".")
        if not txt:
            return padrao
        try:
            v = float(txt)
        except ValueError:
            print("  valor inválido, digite um número.")
            continue
        if (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
            print(f"  valor fora da faixa [{minimo}, {maximo}].")
            continue
        return v


def ler_kernel(msg: str = "Tamanho do kernel (ímpar)", padrao: int = 5) -> int:
    """Lê tamanho de kernel e garante que seja ímpar e >= 3."""
    k = ler_int(msg, padrao, 3, 99)
    if k % 2 == 0:
        k += 1
        print(f"  kernel par não é permitido; usando {k}.")
    return k


def ler_sim_nao(msg: str, padrao: bool = True) -> bool:
    txt = input(f"{msg} [{'S/n' if padrao else 's/N'}]: ").strip().lower()
    if not txt:
        return padrao
    return txt.startswith("s")


# ---------------------------------------------------------------- interação com mouse
def clicar_ponto(img: np.ndarray, titulo: str = "Clique no ponto desejado") -> Optional[Tuple[int, int]]:
    """Mostra a imagem e devolve (x, y) do clique. ESC cancela."""
    ponto = {}

    def cb(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            ponto["p"] = (x, y)

    cv2.imshow(titulo, _para_bgr(img))
    cv2.setMouseCallback(titulo, cb)
    print("Clique na imagem (ESC cancela).")
    while "p" not in ponto:
        if cv2.waitKey(30) == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return ponto.get("p")


def selecionar_roi(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Seleção interativa de ROI. Retorna (x, y, w, h) ou None."""
    x, y, w, h = cv2.selectROI("Selecione a ROI (ENTER confirma, C cancela)", _para_bgr(img), False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return (x, y, w, h) if w > 0 and h > 0 else None
