"""Testes automáticos (sem janelas e sem webcam) dos módulos do projeto.

Execute:  python teste_modulos.py
"""
import csv
import os
import shutil
import tempfile

import cv2
import numpy as np

import basicas
import chamada
import face
import filtros
import histograma as hist
import io_utils as io
import segmentacao as seg


def imagem_sintetica(h=240, w=320) -> np.ndarray:
    """Gradiente + retângulo vermelho + círculo azul + ruído leve."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 200, w, dtype=np.uint8)[None, :, None]
    cv2.rectangle(img, (40, 40), (140, 120), (0, 0, 220), -1)
    cv2.circle(img, (230, 150), 50, (220, 40, 0), -1)
    rng = np.random.default_rng(1)
    ruido = rng.integers(-8, 8, img.shape)
    return np.clip(img.astype(int) + ruido, 0, 255).astype(np.uint8)


def ok(nome: str) -> None:
    print(f"  OK  {nome}")


def testar_basicas(img):
    inv = basicas.inverter_cores(img)
    assert np.array_equal(basicas.inverter_cores(inv), img); ok("inversão (involução)")
    bc = basicas.brilho_contraste(img, 1.0, 0)
    assert np.array_equal(bc, img); ok("brilho/contraste identidade")
    assert basicas.brilho_contraste(img, 1.0, 255).min() >= 0
    assert basicas.brilho_contraste(img, 1.0, 255).max() == 255; ok("brilho satura em 255")
    claro = np.full((10, 10, 3), 200, np.uint8)
    assert basicas.somar_imagens(claro, claro).max() == 255; ok("soma sem overflow")
    assert basicas.subtrair_imagens(claro // 2, claro).max() == 0; ok("subtração sem wrap-around")
    pequena = cv2.resize(img, (50, 50))
    assert basicas.somar_imagens(img, pequena).shape == img.shape; ok("soma com tamanhos diferentes")
    assert basicas.para_cinza(img).ndim == 2; ok("cinza")
    hsv = basicas.bgr_para_hsv(img)
    volta = basicas.hsv_para_bgr(hsv)
    dif = np.abs(volta.astype(int) - img.astype(int))
    # H em 8 bits tem só 180 valores, então a ida-e-volta perde alguns níveis.
    assert dif.max() <= 8 and dif.mean() < 1; ok(f"BGR<->HSV (dif. máx {dif.max()}, média {dif.mean():.2f})")
    assert basicas.redimensionar(img, escala=0.5).shape[:2] == (120, 160); ok("redimensionar por escala")
    assert basicas.redimensionar(img, 100, 50).shape[:2] == (50, 100); ok("redimensionar por dimensões")
    assert basicas.recortar_roi(img, 10, 10, 50, 40).shape[:2] == (40, 50); ok("recorte ROI")
    assert basicas.recortar_roi(img, 300, 200, 100, 100).shape[:2] == (40, 20); ok("ROI limitada à borda")
    try:
        basicas.recortar_roi(img, 1000, 1000, 10, 10)
        raise AssertionError("deveria falhar")
    except ValueError:
        ok("ROI fora da imagem gera erro")
    assert len(basicas.separar_canais(img)) == 3; ok("split de canais")


def testar_filtros(img):
    ruidosa = filtros.adicionar_ruido_sal_pimenta(img, 0.05)
    base = img.astype(float)
    err_media = np.abs(filtros.filtro_media(ruidosa, 5) - base).mean()
    err_mediana = np.abs(filtros.filtro_mediana(ruidosa, 5) - base).mean()
    assert err_mediana < err_media; ok(f"mediana ({err_mediana:.1f}) melhor que média ({err_media:.1f}) em sal-e-pimenta")
    assert filtros.filtro_gaussiano(img, 5).shape == img.shape; ok("gaussiano")
    s = filtros.sobel(img)
    assert s.ndim == 2 and s.max() == 255; ok("Sobel")
    c = filtros.canny(img, 100, 200)
    assert set(np.unique(c)) <= {0, 255} and c.any(); ok("Canny")


def testar_histograma(img):
    g = basicas.para_cinza(img)
    h = hist.calcular_histograma(g)
    ref = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel().astype(int)
    assert np.array_equal(h, ref); ok("histograma manual = cv2.calcHist")
    baixo = (g // 4 + 60).astype(np.uint8)
    eq = hist.equalizar_cinza(baixo)
    assert eq.max() - eq.min() > baixo.max() - baixo.min(); ok("equalização amplia contraste")
    diff = np.abs(eq.astype(int) - cv2.equalizeHist(baixo).astype(int)).max()
    assert diff <= 1; ok(f"equalização manual ≈ cv2.equalizeHist (dif. máx {diff})")
    assert hist.equalizar(img).shape == img.shape; ok("equalização colorida (canal V)")
    assert hist.equalizar_cinza(np.full((5, 5), 7, np.uint8)).shape == (5, 5); ok("equalização de imagem constante")
    painel = hist.desenhar_histogramas(img)
    assert painel.ndim == 3 and painel.shape[2] == 3; ok("gráfico de histogramas")
    assert "B" in hist.analisar_canais(img); ok("análise de canais")
    fx = hist.limiar_fixo(img, 100)
    assert set(np.unique(fx)) <= {0, 255}; ok("limiar fixo")
    assert set(np.unique(hist.limiar_adaptativo(img, 30, 5))) <= {0, 255}; ok("limiar adaptativo (bloco par corrigido)")
    t, bw = hist.limiar_otsu(img)
    assert 0 < t < 255 and set(np.unique(bw)) <= {0, 255}; ok(f"Otsu (limiar={t})")
    q = hist.quantizar(g, 4)
    assert len(np.unique(q)) <= 4; ok("quantização gray 4 níveis")
    assert len(np.unique(hist.quantizar(img, 2))) <= 8; ok("quantização cor 2 níveis")
    try:
        hist.quantizar(g, 1)
        raise AssertionError
    except ValueError:
        ok("quantização com 1 nível gera erro")


def testar_segmentacao(img):
    m = seg.crescimento_regioes(img, (60, 60), tolerancia=30)
    assert m[60, 60] == 255 and m[200, 230] == 0; ok("crescimento de regiões isola o retângulo")
    area = (m > 0).sum()
    assert 70 * 90 < area < 90 * 100, area; ok(f"área segmentada coerente ({area} px)")
    m4 = seg.crescimento_regioes(img, (60, 60), 30, 4)
    assert (m4 > 0).sum() <= area + 50; ok("conectividade 4")
    assert seg.destacar_regiao(img, m).shape == img.shape; ok("destaque da região")
    try:
        seg.crescimento_regioes(img, (9999, 0))
        raise AssertionError
    except ValueError:
        ok("semente fora da imagem gera erro")

    bw = np.zeros((100, 100), np.uint8)
    cv2.rectangle(bw, (20, 20), (80, 80), 255, -1)
    bw[50, 50] = 0          # buraco pequeno
    bw[5, 5] = 255          # ruído
    assert seg.erosao(bw, 3).sum() < bw.sum(); ok("erosão reduz")
    assert seg.dilatacao(bw, 3).sum() > bw.sum(); ok("dilatação aumenta")
    assert seg.abertura(bw, 3)[5, 5] == 0; ok("abertura remove ruído")
    assert seg.fechamento(bw, 3)[50, 50] == 255; ok("fechamento preenche buraco")
    for forma in seg.FORMAS:
        seg.erosao(bw, 5, forma)
    ok("elementos estruturantes (retângulo/elipse/cruz)")


def testar_io(img, pasta):
    caminho = io.salvar_imagem(img, "teste_io", pasta)
    assert os.path.isfile(caminho); ok("salvar imagem")
    lida = io.abrir_imagem(caminho)
    assert np.array_equal(lida, img); ok("abrir imagem (PNG sem perdas)")
    assert io.abrir_imagem(os.path.join(pasta, "nao_existe.png")) is None; ok("arquivo inexistente tratado")
    lixo = os.path.join(pasta, "lixo.png")
    with open(lixo, "wb") as f:
        f.write(b"nao sou imagem")
    assert io.abrir_imagem(lixo) is None; ok("arquivo corrompido tratado")
    dup = io.duplicar(img)
    dup[:] = 0
    assert img.max() > 0; ok("duplicar é independente da original")
    painel = io.montar_lado_a_lado([("A", img), ("B", basicas.para_cinza(img))])
    assert painel.ndim == 3; ok("painel lado a lado (cor + cinza)")


def rosto_falso(semente: int) -> np.ndarray:
    """Textura sintética por 'pessoa' (apenas para testar o pipeline de treino/predição)."""
    rng = np.random.default_rng(semente)
    base = cv2.resize(rng.integers(0, 255, (8, 8), dtype=np.uint8), (200, 200), interpolation=cv2.INTER_CUBIC)
    return base


def testar_face_e_chamada(img, pasta):
    caixas = face.detectar_faces(img)
    assert isinstance(caixas, list); ok(f"detector Haar roda sem erro ({len(caixas)} faces na imagem sintética)")

    antigo = os.getcwd()
    os.chdir(pasta)
    try:
        for pessoa, semente in [("Ana", 1), ("Bruno", 2)]:
            sub = os.path.join("dataset", pessoa)
            os.makedirs(sub)
            base = rosto_falso(semente)
            for i in range(8):
                ruido = np.random.default_rng(100 + i).integers(-10, 10, base.shape)
                cv2.imwrite(os.path.join(sub, f"{i:03d}.png"), np.clip(base + ruido, 0, 255).astype(np.uint8))
        assert face.treinar() == 2; ok("treino LBPH com 2 pessoas")
        rec = face.Reconhecedor(limiar=100)
        for pessoa, semente in [("Ana", 1), ("Bruno", 2)]:
            base = rosto_falso(semente)
            ruido = np.random.default_rng(999).integers(-10, 10, base.shape)
            teste = cv2.equalizeHist(np.clip(base + ruido, 0, 255).astype(np.uint8))
            id_, dist = rec.rec.predict(teste)
            assert rec.nomes[id_] == pessoa, (pessoa, rec.nomes[id_]);
        ok("predição LBPH identifica cada pessoa")
        desconhecida = cv2.equalizeHist(rosto_falso(77))
        _, dist_desc = rec.rec.predict(desconhecida)
        print(f"      (distância de um rosto desconhecido: {dist_desc:.0f}; limiar padrão {face.LIMIAR_PADRAO:.0f})")

        arq = os.path.join("saida", "presenca_teste.csv")
        os.makedirs("saida")
        chamada.registrar_presenca("Ana", arq)
        chamada.registrar_presenca("Bruno", arq)
        with open(arq, encoding="utf-8") as f:
            linhas = list(csv.reader(f))
        assert linhas[0] == ["nome", "data", "hora"] and len(linhas) == 3; ok("CSV de presença (cabeçalho + 2 linhas)")

        # Erros esperados quando o modelo não existe
        os.remove(face.ARQ_MODELO)
        try:
            face.Reconhecedor()
            raise AssertionError
        except FileNotFoundError:
            ok("reconhecedor sem modelo gera erro claro")
    finally:
        os.chdir(antigo)


def main():
    img = imagem_sintetica()
    pasta = tempfile.mkdtemp(prefix="teste_pdi_")
    try:
        for titulo, fn, args in [
            ("Operações básicas", testar_basicas, (img,)),
            ("Filtros e bordas", testar_filtros, (img,)),
            ("Histograma e intensidade", testar_histograma, (img,)),
            ("Segmentação e morfologia", testar_segmentacao, (img,)),
            ("Entrada/saída", testar_io, (img, pasta)),
            ("Face e chamada", testar_face_e_chamada, (img, pasta)),
        ]:
            print(f"[{titulo}]")
            fn(*args)
        print("\nTodos os testes passaram.")
    finally:
        shutil.rmtree(pasta, ignore_errors=True)


if __name__ == "__main__":
    main()
