"""Detecção e reconhecimento facial.

Abordagem: detecção com Haar Cascade (Viola-Jones) + reconhecimento com LBPH
(Local Binary Patterns Histograms, cv2.face). Escolhida por rodar em CPU, sem GPU e
com poucas fotos por pessoa. Limitações: sensível a iluminação, pose e expressão;
funciona melhor com 15-30 fotos por pessoa em condições variadas.
Em LBPH, o valor retornado é uma DISTÂNCIA: quanto menor, mais parecido.
"""
import json
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

PASTA_DATASET = "dataset"
ARQ_MODELO = "modelo_lbph.yml"
ARQ_ROTULOS = "rotulos.json"
TAMANHO_FACE = (200, 200)
LIMIAR_PADRAO = 70.0  # distância máxima aceita como "reconhecido"
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")

_detector = None


def _get_detector():
    """Carrega (uma única vez) o Haar Cascade frontal do OpenCV."""
    global _detector
    if _detector is None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("Esta versão do OpenCV não tem CascadeClassifier (OpenCV 5.x removeu). "
                               "Instale a versão 4.x: pip install \"opencv-contrib-python>=4.8,<5\"")
        _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _detector.empty():
            raise RuntimeError("Não foi possível carregar o Haar Cascade do OpenCV.")
    return _detector


def _tem_contrib() -> bool:
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def detectar_faces(img: np.ndarray, escala: float = 1.2, vizinhos: int = 5,
                   tamanho_min: int = 60) -> List[Tuple[int, int, int, int]]:
    """Retorna lista de caixas (x, y, w, h)."""
    cinza = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.equalizeHist(cinza)
    faces = _get_detector().detectMultiScale(cinza, scaleFactor=escala, minNeighbors=vizinhos,
                                             minSize=(tamanho_min, tamanho_min))
    return [tuple(int(v) for v in f) for f in faces]


def preparar_face(img: np.ndarray, caixa: Tuple[int, int, int, int]) -> np.ndarray:
    """Recorta, converte para cinza, redimensiona e equaliza (entrada padrão do LBPH)."""
    x, y, w, h = caixa
    cinza = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rosto = cv2.resize(cinza[y:y + h, x:x + w], TAMANHO_FACE)
    return cv2.equalizeHist(rosto)


def desenhar_deteccoes(img: np.ndarray, caixas, rotulos=None) -> np.ndarray:
    saida = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(caixas):
        cor = (0, 255, 0)
        texto = rotulos[i] if rotulos else None
        if texto and texto.startswith("Desconhecido"):
            cor = (0, 0, 255)
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 2)
        if texto:
            cv2.putText(saida, texto, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
    return saida


# ---------------------------------------------------------------- cadastro
def nome_seguro(nome: str) -> str:
    return "".join(c for c in nome.strip() if c.isalnum() or c in " _-").strip().replace(" ", "_")


def cadastrar_por_webcam(nome: str, quantidade: int = 20, camera: int = 0,
                         pasta: str = PASTA_DATASET) -> int:
    """Captura `quantidade` fotos de rosto pela webcam. Retorna quantas foram salvas."""
    nome = nome_seguro(nome)
    if not nome:
        print("[erro] Nome inválido.")
        return 0
    destino = os.path.join(pasta, nome)
    os.makedirs(destino, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("[erro] Webcam indisponível.")
        return 0
    salvas, ultimo = 0, 0.0
    print("Olhe para a câmera e varie levemente a pose. ESC interrompe.")
    try:
        while salvas < quantidade:
            ok, quadro = cap.read()
            if not ok:
                print("[erro] Falha ao ler a webcam.")
                break
            caixas = detectar_faces(quadro)
            agora = time.time()
            if len(caixas) == 1 and agora - ultimo > 0.4:
                x, y, w, h = caixas[0]
                existentes = len(os.listdir(destino))
                cv2.imwrite(os.path.join(destino, f"{existentes:03d}.png"), quadro[y:y + h, x:x + w])
                salvas += 1
                ultimo = agora
            vis = desenhar_deteccoes(quadro, caixas)
            cv2.putText(vis, f"{salvas}/{quantidade}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Cadastro de face", vis)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return salvas


def cadastrar_de_pasta(nome: str, pasta_origem: str, pasta: str = PASTA_DATASET) -> int:
    """Importa fotos de uma pasta: detecta o rosto em cada imagem e salva o recorte."""
    nome = nome_seguro(nome)
    if not nome or not os.path.isdir(pasta_origem):
        print("[erro] Nome ou pasta de origem inválidos.")
        return 0
    destino = os.path.join(pasta, nome)
    os.makedirs(destino, exist_ok=True)
    salvas = 0
    for arq in sorted(os.listdir(pasta_origem)):
        if not arq.lower().endswith(EXTENSOES):
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(pasta_origem, arq), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        caixas = detectar_faces(img)
        if len(caixas) != 1:
            print(f"  ignorada ({len(caixas)} faces): {arq}")
            continue
        x, y, w, h = caixas[0]
        existentes = len(os.listdir(destino))
        cv2.imwrite(os.path.join(destino, f"{existentes:03d}.png"), img[y:y + h, x:x + w])
        salvas += 1
    return salvas


# ---------------------------------------------------------------- treino
def treinar(pasta: str = PASTA_DATASET) -> Optional[int]:
    """Treina o LBPH com dataset/<aluno>/*.png e salva modelo + rótulos.
    Retorna o número de pessoas treinadas ou None em erro."""
    if not _tem_contrib():
        print("[erro] Instale opencv-contrib-python (pip install opencv-contrib-python).")
        return None
    if not os.path.isdir(pasta):
        print(f"[erro] Pasta '{pasta}' não existe. Cadastre pessoas primeiro.")
        return None
    rostos, ids, nomes = [], [], []
    for nome in sorted(os.listdir(pasta)):
        sub = os.path.join(pasta, nome)
        if not os.path.isdir(sub):
            continue
        n_antes = len(rostos)
        for arq in sorted(os.listdir(sub)):
            if not arq.lower().endswith(EXTENSOES):
                continue
            img = cv2.imdecode(np.fromfile(os.path.join(sub, arq), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Imagens já recortadas: usa direto; se houver uma face maior, recorta.
            if min(img.shape[:2]) > 250:
                caixas = detectar_faces(img)
                if caixas:
                    img = preparar_face(img, max(caixas, key=lambda c: c[2] * c[3]))
                    rostos.append(img)
                else:
                    rostos.append(cv2.equalizeHist(cv2.resize(img, TAMANHO_FACE)))
            else:
                rostos.append(cv2.equalizeHist(cv2.resize(img, TAMANHO_FACE)))
            ids.append(len(nomes))
        if len(rostos) > n_antes:
            nomes.append(nome)
    if len(set(ids)) < 1:
        print("[erro] Nenhuma imagem de treino encontrada.")
        return None
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(rostos, np.array(ids))
    rec.write(ARQ_MODELO)
    with open(ARQ_ROTULOS, "w", encoding="utf-8") as f:
        json.dump(nomes, f, ensure_ascii=False)
    print(f"Modelo treinado: {len(nomes)} pessoa(s), {len(rostos)} imagem(ns).")
    return len(nomes)


class Reconhecedor:
    """Carrega o modelo treinado e identifica rostos em quadros."""

    def __init__(self, limiar: float = LIMIAR_PADRAO):
        if not _tem_contrib():
            raise RuntimeError("opencv-contrib-python não instalado.")
        if not (os.path.isfile(ARQ_MODELO) and os.path.isfile(ARQ_ROTULOS)):
            raise FileNotFoundError("Modelo não treinado. Use a opção de treinar primeiro.")
        self.rec = cv2.face.LBPHFaceRecognizer_create()
        self.rec.read(ARQ_MODELO)
        with open(ARQ_ROTULOS, encoding="utf-8") as f:
            self.nomes = json.load(f)
        self.limiar = limiar

    def identificar(self, img: np.ndarray):
        """Retorna lista de (caixa, nome, distancia). nome='Desconhecido' se acima do limiar."""
        resultado = []
        for caixa in detectar_faces(img):
            id_, dist = self.rec.predict(preparar_face(img, caixa))
            nome = self.nomes[id_] if dist <= self.limiar and id_ < len(self.nomes) else "Desconhecido"
            resultado.append((caixa, nome, float(dist)))
        return resultado

    def anotar(self, img: np.ndarray):
        res = self.identificar(img)
        rotulos = [f"{n} ({d:.0f})" if n != "Desconhecido" else "Desconhecido" for _, n, d in res]
        return desenhar_deteccoes(img, [c for c, _, _ in res], rotulos), res


def reconhecer_em_video(fonte=0, limiar: float = LIMIAR_PADRAO) -> None:
    """Reconhecimento em tempo real (webcam = 0) ou em arquivo de vídeo. ESC sai."""
    try:
        rec = Reconhecedor(limiar)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[erro] {exc}")
        return
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print("[erro] Não foi possível abrir a fonte de vídeo.")
        return
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            vis, _ = rec.anotar(quadro)
            cv2.imshow("Reconhecimento facial (ESC sai)", vis)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
