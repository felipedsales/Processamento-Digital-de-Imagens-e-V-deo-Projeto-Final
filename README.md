# Processamento Digital de Imagens e Vídeo — Projeto Final

Software em Python com menu de terminal que integra as técnicas de Processamento de
Imagens e Visão Computacional estudadas no semestre, incluindo reconhecimento facial e
chamada automatizada.

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
python main.py
```

> **Importante:** use OpenCV **4.x** (o `requirements.txt` já fixa `<5`). O OpenCV 5.0 removeu
> o `CascadeClassifier` (Haar Cascade), usado na detecção de faces.
> É necessário o pacote `opencv-contrib-python` (contém o reconhecedor LBPH). Não instale
> `opencv-python` junto com ele.

## Estrutura

```
main.py          menu e loop principal (funções op_* ligam o menu aos módulos)
io_utils.py      abrir/salvar/duplicar, painel lado a lado, leitura validada, mouse
basicas.py       inversão, brilho/contraste, soma/subtração, cinza, HSV, resize, ROI
filtros.py       média, mediana, gaussiano, Sobel, Canny, ruído sal-e-pimenta
histograma.py    histogramas, equalização, limiares (fixo/adaptativo/Otsu), quantização
segmentacao.py   crescimento de regiões e morfologia (erosão, dilatação, abertura, fechamento)
face.py          detecção (Haar), cadastro, treino e reconhecimento (LBPH)
chamada.py       chamada automatizada com registro em CSV
teste_modulos.py testes automáticos sem janelas/webcam
dataset/         uma subpasta por aluno (criada pelo cadastro)
saida/           imagens salvas e CSVs de presença
```

## Como o programa funciona

- Há sempre a **imagem original** (nunca alterada) e a **imagem de trabalho** (cópia).
- Cada operação mostra *trabalho × resultado* lado a lado e guarda o resultado.
- Opção **4** transforma o último resultado na nova imagem de trabalho, permitindo
  **encadear** operações (ex.: cinza → gaussiano → Canny). Opção **5** volta à original.
- Opção **3** salva qualquer resultado em `saida/`.
- Nas janelas de imagem, qualquer tecla continua; **ESC** cancela cliques e vídeos.

## Opções do menu

| Nº | Função | Observação |
|----|--------|-----------|
| 1 | Abrir imagem | aceita caminhos com acento e aspas |
| 2 | Mostrar original × processadas | até 3 últimos resultados |
| 3 | Salvar imagem processada | PNG por padrão |
| 4 | Usar último resultado como trabalho | encadeamento |
| 5 | Resetar para a original | |
| 6 | Inverter cores | manual (NumPy) |
| 7 | Brilho e contraste | `g = α·f + β`, manual |
| 8 / 9 | Somar / subtrair outra imagem | saturação e ajuste automático de tamanho |
| 10 | Converter para cinza | |
| 11 | RGB ↔ HSV | mostra H, S, V e confere a conversão de volta |
| 12 | Redimensionar | por escala ou largura × altura |
| 13 | Recortar ROI | mouse ou coordenadas |
| 14–16 | Filtros média, mediana, gaussiano | kernel ímpar validado |
| 17 | Ruído sal-e-pimenta | para comparar média × mediana |
| 18 / 19 | Sobel / Canny | limiares ajustáveis |
| 20 | Histogramas | cinza e por canal |
| 21 | Equalizar histograma | manual (CDF); em cor usa o canal V do HSV |
| 22 | Splitting e análise de canais | média, desvio, mín., máx. |
| 23–25 | Binarização fixa, adaptativa, Otsu | |
| 26 | Quantização | manual, níveis por canal |
| 27 | Crescimento de regiões | clique na semente, tolerância e conectividade 4/8 |
| 28–31 | Erosão, dilatação, abertura, fechamento | forma e tamanho do elemento estruturante |
| 32 | Detectar faces na imagem | |
| 33 / 34 | Cadastrar aluno (webcam / pasta) | |
| 35 | Treinar modelo | gera `modelo_lbph.yml` e `rotulos.json` |
| 36 / 37 | Reconhecer em imagem / vídeo / webcam | limiar de distância configurável |
| 38 | **Chamada automatizada** | gera `saida/presenca_AAAAMMDD_HHMMSS.csv` |

Algoritmos implementados manualmente com NumPy: inversão, brilho/contraste, histograma,
equalização, quantização, limiar fixo e crescimento de regiões. Os demais usam OpenCV.

## Reconhecimento facial e chamada

**Abordagem:** detecção com Haar Cascade e reconhecimento com LBPH (histogramas de padrões
binários locais). Roda em CPU, não exige GPU e funciona com poucas fotos por pessoa.

**Limitações:** sensível a iluminação, pose, óculos e expressão; o Haar falha com rosto
muito inclinado. Para melhor resultado, use 15 a 30 fotos por aluno com iluminação e
ângulos variados.

**Limiar de confiança:** o LBPH devolve uma *distância* (menor = mais parecido). Acima do
limiar (padrão 70) o rosto é rotulado como "Desconhecido" e **nunca** é registrado. Se
estiver reconhecendo pessoas erradas, diminua o limiar; se não reconhece ninguém, aumente.

### Passo a passo

1. Opção **33** (webcam) ou **34** (pasta com fotos) para cada aluno. As fotos são salvas
   em `dataset/Nome_do_Aluno/`. Você também pode criar essas pastas manualmente.
2. Opção **35** para treinar. Refaça sempre que cadastrar alguém novo.
3. Opção **38** para a chamada. Cada aluno só é registrado depois de ser reconhecido em
   5 quadros seguidos e **uma única vez por sessão**. O vídeo mostra o contador de
   presentes; ao sair (ESC), o terminal lista presentes e ausentes.

O CSV tem as colunas `nome, data, hora`.

## Testes

```bash
python teste_modulos.py
```

Verifica cada módulo com imagens sintéticas (inclui comparação das versões manuais com as
do OpenCV e o pipeline de treino/predição do LBPH). Não testa a detecção em rostos reais
nem a webcam; isso deve ser validado com fotos dos alunos.

## Roteiro de demonstração (8 a 10 passos)

1. **Abrir** uma foto (opção 1) e **mostrar** original e trabalho (2).
2. **Básicas:** inverter (6), brilho/contraste (7), cinza (10), HSV (11).
3. **Redimensionar** (12) e **recortar uma ROI com o mouse** (13).
4. **Filtros:** adicionar ruído (17), comparar média (14) e mediana (15) e mostrar que a mediana remove melhor o sal-e-pimenta; depois gaussiano (16).
5. **Bordas:** Sobel (18) e Canny (19).
6. **Histograma:** exibir (20), equalizar (21) e mostrar o antes/depois.
7. **Binarização e quantização:** fixo (23), adaptativo (24), Otsu (25), quantização com 4 níveis (26).
8. **Segmentação:** crescimento de regiões clicando em um objeto (27); ajustar a tolerância.
9. **Morfologia:** Otsu → abertura (30) e fechamento (31); mostrar erosão (28) e dilatação (29).
10. **Reconhecimento facial:** cadastrar dois colegas (33), treinar (35) e executar a **chamada automatizada** (38); abrir o CSV gerado.

## Melhorias futuras

- Trocar Haar + LBPH por detector DNN (YuNet) e embeddings (SFace/FaceNet), mais robustos a pose e iluminação.
- Interface gráfica (Tkinter ou Streamlit) no lugar do menu de terminal.
- Aumento de dados (espelhamento, variação de brilho) no cadastro das faces.
- Chamada com anti-spoofing (liveness) para evitar fraude com foto no celular.
- Exportar a lista de presença para Excel e consolidar várias aulas.
- Mais operações: filtros no domínio da frequência, transformada de Hough, k-means para segmentação.

## Referências

- Antonello, A. — *Introdução à Visão Computacional com Python e OpenCV 3*.
- Barelli, F. — *Introdução à Visão Computacional: uma abordagem prática com Python e OpenCV*. Casa do Código, 2018.
- Mollosi, M. et al. — Sistema de detecção e contagem de alunos em sala de aula utilizando IA e processamento de vídeo. *Iguazu Science*, v. 3, n. 7, 2025.
- Gonzalez, R. C.; Woods, R. E. — *Processamento de Imagens Digitais*. Blucher, 2000.
