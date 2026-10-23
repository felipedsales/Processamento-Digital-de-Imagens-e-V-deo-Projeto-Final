"""Chamada automatizada: reconhece alunos pela face e registra presença em CSV."""
import csv
import os
from datetime import datetime
from typing import Dict, Set

import cv2

import face

PASTA_SAIDA = "saida"
CONFIRMACOES = 5  # quadros consecutivos com o mesmo nome antes de registrar (evita falsos positivos)


def registrar_presenca(nome: str, arquivo: str) -> None:
    """Acrescenta (nome, data, hora) ao CSV, criando o cabeçalho se necessário."""
    novo = not os.path.isfile(arquivo)
    agora = datetime.now()
    with open(arquivo, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["nome", "data", "hora"])
        w.writerow([nome, agora.strftime("%Y-%m-%d"), agora.strftime("%H:%M:%S")])


def executar_chamada(fonte=0, limiar: float = face.LIMIAR_PADRAO) -> Set[str]:
    """Roda a chamada em webcam/vídeo. Cada aluno é registrado uma única vez por sessão.
    Rostos desconhecidos são rotulados e nunca registrados. ESC encerra."""
    try:
        rec = face.Reconhecedor(limiar)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[erro] {exc}")
        return set()
    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print("[erro] Não foi possível abrir a fonte de vídeo.")
        return set()

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    arquivo = os.path.join(PASTA_SAIDA, f"presenca_{datetime.now():%Y%m%d_%H%M%S}.csv")
    presentes: Set[str] = set()
    contagem: Dict[str, int] = {}
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            vis, res = rec.anotar(quadro)
            vistos = {n for _, n, _ in res if n != "Desconhecido"}
            for nome in vistos:
                contagem[nome] = contagem.get(nome, 0) + 1
                if contagem[nome] >= CONFIRMACOES and nome not in presentes:
                    presentes.add(nome)
                    registrar_presenca(nome, arquivo)
                    print(f"  presença registrada: {nome}")
            for nome in list(contagem):  # zera quem sumiu do quadro
                if nome not in vistos:
                    contagem[nome] = 0
            cv2.putText(vis, f"Presentes: {len(presentes)} / {len(rec.nomes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Chamada automatizada (ESC encerra)", vis)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    ausentes = sorted(set(rec.nomes) - presentes)
    print(f"\nPresentes ({len(presentes)}): {', '.join(sorted(presentes)) or '-'}")
    print(f"Ausentes  ({len(ausentes)}): {', '.join(ausentes) or '-'}")
    if presentes:
        print(f"CSV salvo em: {arquivo}")
    return presentes
