"""Software de Processamento Digital de Imagens e Vídeo — menu principal.

Execute:  python main.py
"""
import os

import cv2
import numpy as np

import basicas
import chamada
import face
import filtros
import histograma as hist
import io_utils as io
import segmentacao as seg
from io_utils import Estado


# ====================================================================== utilidades do menu
def exigir_imagem(estado: Estado) -> bool:
    if not estado.tem_imagem():
        print("[aviso] Carregue uma imagem primeiro (opção 1).")
        return False
    return True


def mostrar_resultado(estado: Estado, nome: str, resultado: np.ndarray) -> None:
    """Registra o resultado e mostra original x trabalho x resultado."""
    estado.registrar(nome, resultado)
    io.mostrar(nome, io.montar_lado_a_lado([("Trabalho", estado.trabalho), (nome, resultado)]))


def executar(estado: Estado, nome: str, funcao) -> None:
    """Aplica `funcao(img_trabalho)` com tratamento de erros e exibe o resultado."""
    if not exigir_imagem(estado):
        return
    try:
        resultado = funcao(estado.trabalho)
    except Exception as exc:
        print(f"[erro] {nome}: {exc}")
        return
    if resultado is not None:
        mostrar_resultado(estado, nome, resultado)


# ====================================================================== 1. arquivo e visualização
def op_abrir(estado: Estado) -> None:
    caminho = input("Caminho da imagem: ")
    img = io.abrir_imagem(caminho)
    if img is None:
        return
    estado.original = img
    estado.trabalho = io.duplicar(img)
    estado.resultados.clear()
    estado.caminho = caminho
    print(f"Imagem carregada: {img.shape[1]}x{img.shape[0]} px, {img.shape[2]} canais.")


def op_mostrar(estado: Estado) -> None:
    io.mostrar_original_e_processadas(estado)


def op_salvar(estado: Estado) -> None:
    if not estado.resultados:
        print("[aviso] Ainda não há imagem processada para salvar.")
        return
    print("Resultados disponíveis:")
    for i, (nome, _) in enumerate(estado.resultados, 1):
        print(f"  {i}) {nome}")
    idx = io.ler_int("Qual salvar", len(estado.resultados), 1, len(estado.resultados))
    nome, img = estado.resultados[idx - 1]
    arq = input("Nome do arquivo (sem extensão = .png): ").strip() or nome.replace(" ", "_")
    caminho = io.salvar_imagem(img, arq)
    if caminho:
        print(f"Salvo em: {caminho}")


def op_aplicar_resultado(estado: Estado) -> None:
    ult = estado.ultimo()
    if ult is None:
        print("[aviso] Não há resultado para aplicar.")
        return
    estado.trabalho = ult.copy()
    if estado.trabalho.ndim == 2:  # mantém o pipeline em BGR
        estado.trabalho = cv2.cvtColor(estado.trabalho, cv2.COLOR_GRAY2BGR)
    print("Último resultado agora é a imagem de trabalho (operações podem ser encadeadas).")


def op_resetar(estado: Estado) -> None:
    if estado.original is None:
        print("[aviso] Nenhuma imagem carregada.")
        return
    estado.trabalho = io.duplicar(estado.original)
    estado.resultados.clear()
    print("Imagem de trabalho restaurada para a original.")


# ====================================================================== 2. operações básicas
def op_inverter(e):
    executar(e, "Inversão", basicas.inverter_cores)


def op_brilho_contraste(e):
    alfa = io.ler_float("Contraste (alfa, 0.1 a 3.0)", 1.2, 0.1, 3.0)
    beta = io.ler_int("Brilho (beta, -255 a 255)", 30, -255, 255)
    executar(e, f"Brilho/Contraste a={alfa} b={beta}", lambda im: basicas.brilho_contraste(im, alfa, beta))


def _segunda_imagem() -> np.ndarray:
    return io.abrir_imagem(input("Caminho da segunda imagem: "))


def op_somar(e):
    if exigir_imagem(e):
        b = _segunda_imagem()
        if b is not None:
            executar(e, "Soma", lambda im: basicas.somar_imagens(im, b))


def op_subtrair(e):
    if exigir_imagem(e):
        b = _segunda_imagem()
        if b is not None:
            executar(e, "Subtração", lambda im: basicas.subtrair_imagens(im, b))


def op_cinza(e):
    executar(e, "Cinza", basicas.para_cinza)


def op_hsv(e):
    def f(im):
        hsv = basicas.bgr_para_hsv(im)
        h, s, v = cv2.split(hsv)
        print("Canais HSV exibidos: H (matiz), S (saturação), V (valor).")
        io.mostrar("HSV: H | S | V", io.montar_lado_a_lado([("H", h), ("S", s), ("V", v)], 300))
        volta = basicas.hsv_para_bgr(hsv)
        print(f"Conversão HSV->BGR reproduz a original? diferença máxima = "
              f"{int(np.abs(volta.astype(int) - im.astype(int)).max())}")
        return hsv
    executar(e, "HSV (visualização crua)", f)


def op_redimensionar(e):
    if not exigir_imagem(e):
        return
    modo = input("Redimensionar por (1) escala ou (2) largura x altura? [1]: ").strip() or "1"
    if modo == "2":
        h, w = e.trabalho.shape[:2]
        larg = io.ler_int("Largura", w, 1, 20000)
        alt = io.ler_int("Altura", h, 1, 20000)
        executar(e, f"Redimensionada {larg}x{alt}", lambda im: basicas.redimensionar(im, larg, alt))
    else:
        esc = io.ler_float("Escala (ex.: 0.5 metade, 2 dobro)", 0.5, 0.01, 10)
        executar(e, f"Redimensionada x{esc}", lambda im: basicas.redimensionar(im, escala=esc))


def op_recortar(e):
    if not exigir_imagem(e):
        return
    modo = input("ROI: (1) selecionar com o mouse ou (2) digitar coordenadas? [1]: ").strip() or "1"
    if modo == "2":
        h, w = e.trabalho.shape[:2]
        x = io.ler_int("x", 0, 0, w - 1)
        y = io.ler_int("y", 0, 0, h - 1)
        ww = io.ler_int("largura", w // 2, 1, w)
        hh = io.ler_int("altura", h // 2, 1, h)
        roi = (x, y, ww, hh)
    else:
        roi = io.selecionar_roi(e.trabalho)
        if roi is None:
            print("Seleção cancelada.")
            return
    executar(e, "ROI", lambda im: basicas.recortar_roi(im, *roi))


# ====================================================================== 3. filtros e bordas
def op_media(e):
    k = io.ler_kernel()
    executar(e, f"Média {k}x{k}", lambda im: filtros.filtro_media(im, k))


def op_mediana(e):
    k = io.ler_kernel()
    executar(e, f"Mediana {k}x{k}", lambda im: filtros.filtro_mediana(im, k))


def op_gaussiano(e):
    k = io.ler_kernel()
    sigma = io.ler_float("Sigma (0 = automático)", 0, 0, 50)
    executar(e, f"Gaussiano {k}x{k}", lambda im: filtros.filtro_gaussiano(im, k, sigma))


def op_ruido(e):
    p = io.ler_float("Proporção de ruído sal-e-pimenta (0 a 0.5)", 0.05, 0, 0.5)
    executar(e, "Ruído sal-pimenta", lambda im: filtros.adicionar_ruido_sal_pimenta(im, p))


def op_sobel(e):
    k = io.ler_int("Kernel Sobel (1,3,5,7)", 3, 1, 7)
    if k % 2 == 0:
        k += 1
    executar(e, "Sobel", lambda im: filtros.sobel(im, k))


def op_canny(e):
    t1 = io.ler_int("Limiar inferior", 100, 0, 255)
    t2 = io.ler_int("Limiar superior", 200, 0, 255)
    executar(e, f"Canny {t1}/{t2}", lambda im: filtros.canny(im, t1, t2))


# ====================================================================== 4. histograma e intensidade
def op_histograma(e):
    def f(im):
        painel = hist.desenhar_histogramas(im)
        io.mostrar("Histogramas", painel)
        return None
    executar(e, "Histograma", f)


def op_equalizar(e):
    def f(im):
        r = hist.equalizar(im)
        io.mostrar("Histogramas antes | depois",
                   io.montar_lado_a_lado([("Antes", hist.desenhar_histogramas(im)),
                                          ("Depois", hist.desenhar_histogramas(r))], 300))
        return r
    executar(e, "Equalização", f)


def op_canais(e):
    def f(im):
        print(hist.analisar_canais(im))
        b, g, r = basicas.separar_canais(im)
        io.mostrar("Canais B | G | R", io.montar_lado_a_lado([("B", b), ("G", g), ("R", r)], 300))
        return None
    executar(e, "Canais", f)


def op_limiar_fixo(e):
    t = io.ler_int("Limiar", 127, 0, 255)
    executar(e, f"Limiar fixo {t}", lambda im: hist.limiar_fixo(im, t))


def op_limiar_adaptativo(e):
    bloco = io.ler_kernel("Tamanho do bloco (ímpar)", 31)
    c = io.ler_int("Constante C", 5, -50, 50)
    executar(e, "Limiar adaptativo", lambda im: hist.limiar_adaptativo(im, bloco, c))


def op_otsu(e):
    def f(im):
        t, bw = hist.limiar_otsu(im)
        print(f"Limiar de Otsu calculado: {t}")
        return bw
    executar(e, "Otsu", f)


def op_quantizar(e):
    n = io.ler_int("Número de níveis por canal (2 a 128)", 4, 2, 128)
    executar(e, f"Quantização {n} níveis", lambda im: hist.quantizar(im, n))


# ====================================================================== 5. segmentação e morfologia
def op_crescimento(e):
    if not exigir_imagem(e):
        return
    tol = io.ler_int("Tolerância (0-255)", 20, 0, 255)
    conn = io.ler_int("Conectividade (4 ou 8)", 8, 4, 8)
    if conn not in (4, 8):
        conn = 8
    p = io.clicar_ponto(e.trabalho, "Clique na semente")
    if p is None:
        print("Cancelado.")
        return
    executar(e, f"Crescimento de regiões tol={tol}",
             lambda im: seg.destacar_regiao(im, seg.crescimento_regioes(im, p, tol, conn)))


def _morf(e, nome, funcao):
    k = io.ler_kernel("Tamanho do elemento estruturante (ímpar)", 5)
    forma = input("Forma (retangulo/elipse/cruz) [retangulo]: ").strip().lower() or "retangulo"
    if forma not in seg.FORMAS:
        print("Forma inválida; usando retangulo.")
        forma = "retangulo"
    if exigir_imagem(e) and e.trabalho.ndim == 3 and len(np.unique(e.trabalho)) > 2:
        if io.ler_sim_nao("Binarizar com Otsu antes (recomendado em morfologia)?", True):
            _, bw = hist.limiar_otsu(e.trabalho)
            e.trabalho = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            print("Imagem de trabalho binarizada (use 'Resetar' para voltar).")
    executar(e, f"{nome} {k}x{k} {forma}", lambda im: funcao(im, k, forma))


def op_erosao(e):
    _morf(e, "Erosão", seg.erosao)


def op_dilatacao(e):
    _morf(e, "Dilatação", seg.dilatacao)


def op_abertura(e):
    _morf(e, "Abertura", seg.abertura)


def op_fechamento(e):
    _morf(e, "Fechamento", seg.fechamento)


# ====================================================================== 6. reconhecimento facial
def op_detectar_faces(e):
    def f(im):
        caixas = face.detectar_faces(im)
        print(f"{len(caixas)} face(s) detectada(s).")
        return face.desenhar_deteccoes(im, caixas)
    executar(e, "Detecção de faces", f)


def op_cadastrar_webcam(_):
    nome = input("Nome do aluno: ")
    n = io.ler_int("Quantidade de fotos", 20, 5, 100)
    cam = io.ler_int("Índice da câmera", 0, 0, 10)
    print(f"{face.cadastrar_por_webcam(nome, n, cam)} foto(s) salva(s).")


def op_cadastrar_pasta(_):
    nome = input("Nome do aluno: ")
    origem = input("Pasta com as fotos: ").strip().strip('"')
    print(f"{face.cadastrar_de_pasta(nome, origem)} foto(s) importada(s).")


def op_treinar(_):
    face.treinar()


def op_reconhecer_imagem(e):
    def f(im):
        rec = face.Reconhecedor(io.ler_float("Limiar de distância", face.LIMIAR_PADRAO, 1, 300))
        vis, res = rec.anotar(im)
        for _, nome, d in res:
            print(f"  {nome} (distância {d:.1f})")
        return vis
    if exigir_imagem(e):
        try:
            executar(e, "Reconhecimento facial", f)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[erro] {exc}")


def _fonte_video():
    txt = input("Fonte: Enter = webcam, ou caminho de um vídeo: ").strip().strip('"')
    return int(io.ler_int("Índice da câmera", 0, 0, 10)) if not txt else txt


def op_reconhecer_video(_):
    face.reconhecer_em_video(_fonte_video(), io.ler_float("Limiar de distância", face.LIMIAR_PADRAO, 1, 300))


def op_chamada(_):
    chamada.executar_chamada(_fonte_video(), io.ler_float("Limiar de distância", face.LIMIAR_PADRAO, 1, 300))


# ====================================================================== menu
MENU = [
    ("ARQUIVO E VISUALIZAÇÃO", [
        ("1", "Abrir imagem", op_abrir),
        ("2", "Mostrar original x processadas", op_mostrar),
        ("3", "Salvar imagem processada", op_salvar),
        ("4", "Usar último resultado como imagem de trabalho (encadear)", op_aplicar_resultado),
        ("5", "Resetar para a original", op_resetar),
    ]),
    ("OPERAÇÕES BÁSICAS", [
        ("6", "Inverter cores", op_inverter),
        ("7", "Brilho e contraste", op_brilho_contraste),
        ("8", "Somar com outra imagem", op_somar),
        ("9", "Subtrair outra imagem", op_subtrair),
        ("10", "Converter para cinza", op_cinza),
        ("11", "Converter RGB <-> HSV", op_hsv),
        ("12", "Redimensionar", op_redimensionar),
        ("13", "Recortar ROI", op_recortar),
    ]),
    ("FILTRAGEM E REALCE", [
        ("14", "Filtro da média", op_media),
        ("15", "Filtro da mediana", op_mediana),
        ("16", "Filtro gaussiano", op_gaussiano),
        ("17", "Adicionar ruído sal-e-pimenta (para testar filtros)", op_ruido),
        ("18", "Bordas: Sobel", op_sobel),
        ("19", "Bordas: Canny", op_canny),
    ]),
    ("HISTOGRAMA E INTENSIDADE", [
        ("20", "Exibir histogramas", op_histograma),
        ("21", "Equalizar histograma", op_equalizar),
        ("22", "Splitting e análise de canais", op_canais),
        ("23", "Binarização: limiar fixo", op_limiar_fixo),
        ("24", "Binarização: adaptativa", op_limiar_adaptativo),
        ("25", "Binarização: Otsu", op_otsu),
        ("26", "Quantização", op_quantizar),
    ]),
    ("SEGMENTAÇÃO E MORFOLOGIA", [
        ("27", "Crescimento de regiões (clique na semente)", op_crescimento),
        ("28", "Erosão", op_erosao),
        ("29", "Dilatação", op_dilatacao),
        ("30", "Abertura", op_abertura),
        ("31", "Fechamento", op_fechamento),
    ]),
    ("RECONHECIMENTO FACIAL E CHAMADA", [
        ("32", "Detectar faces na imagem", op_detectar_faces),
        ("33", "Cadastrar aluno (webcam)", op_cadastrar_webcam),
        ("34", "Cadastrar aluno (pasta de fotos)", op_cadastrar_pasta),
        ("35", "Treinar modelo", op_treinar),
        ("36", "Reconhecer faces na imagem", op_reconhecer_imagem),
        ("37", "Reconhecer faces em vídeo/webcam", op_reconhecer_video),
        ("38", "CHAMADA AUTOMATIZADA (gera CSV)", op_chamada),
    ]),
]
ACOES = {cod: fn for _, itens in MENU for cod, _, fn in itens}


def imprimir_menu(estado: Estado) -> None:
    print("\n" + "=" * 60)
    print(" PROCESSAMENTO DE IMAGENS E VÍDEO — Projeto Final")
    if estado.tem_imagem():
        h, w = estado.trabalho.shape[:2]
        print(f" Imagem: {os.path.basename(estado.caminho)} ({w}x{h})")
    else:
        print(" Imagem: (nenhuma carregada)")
    print("=" * 60)
    for titulo, itens in MENU:
        print(f"\n [{titulo}]")
        for cod, desc, _ in itens:
            print(f"  {cod:>2}) {desc}")
    print("\n   0) Sair")


def main() -> None:
    estado = Estado()
    while True:
        imprimir_menu(estado)
        op = input("\nEscolha uma opção: ").strip()
        if op == "0":
            print("Até logo!")
            break
        acao = ACOES.get(op)
        if acao is None:
            print("[aviso] Opção inválida.")
            continue
        try:
            acao(estado)
        except KeyboardInterrupt:
            print("\n[aviso] Operação interrompida.")
            cv2.destroyAllWindows()
        except Exception as exc:  # o menu nunca deve derrubar o programa
            print(f"[erro] Falha inesperada: {exc}")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
