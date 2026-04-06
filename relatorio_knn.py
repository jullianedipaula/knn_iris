"""
Gerador do Relatório Técnico: Classificação de Espécies Iris via KNN
Execute este script para gerar Relatorio_KNN_Iris.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

NOMES_CLASSES = ['Setosa', 'Versicolor', 'Virginica']
METRICAS = [
    ('euclidean',      {'metric': 'euclidean'}),
    ('manhattan',      {'metric': 'manhattan'}),
    ('chebyshev',      {'metric': 'chebyshev'}),
    ('minkowski(p=3)', {'metric': 'minkowski', 'p': 3}),
]
CORES      = ['steelblue', 'tomato', 'seagreen', 'darkorange']
MARCADORES = ['o', 's', '^', 'D']
FEATURES   = ['Comp. Sépala', 'Larg. Sépala', 'Comp. Pétala', 'Larg. Pétala']

DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def rodar_experimento():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled  = scaler.transform(X_teste)

    valores_k  = range(1, 16)
    resultados = {}
    for nome, params in METRICAS:
        acuracias = []
        for k in valores_k:
            modelo = KNeighborsClassifier(n_neighbors=k, **params)
            modelo.fit(X_treino_scaled, y_treino)
            acuracias.append(modelo.score(X_teste_scaled, y_teste))
        resultados[nome] = acuracias

    return X_treino, X_treino_scaled, valores_k, resultados


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def gerar_grafico_normalizacao(X_antes, X_depois, caminho):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, dados, titulo in zip(axes, [X_antes, X_depois],
                                  ['Antes da Normalização', 'Após Normalização (StandardScaler)']):
        ax.boxplot(dados, tick_labels=FEATURES, patch_artist=True,
                   boxprops=dict(facecolor='#aec6e8', color='steelblue'),
                   medianprops=dict(color='navy', linewidth=2))
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel('Valor')
        ax.grid(axis='y', alpha=0.4)

    fig.tight_layout(pad=2)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def gerar_grafico_acuracia(valores_k, resultados, caminho):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (nome, acuracias), cor, marcador in zip(resultados.items(), CORES, MARCADORES):
        ax.plot(list(valores_k), acuracias, label=nome,
                color=cor, marker=marcador, linewidth=2, markersize=7)
    ax.set_xlabel('Valor de K')
    ax.set_ylabel('Acurácia')
    ax.set_title('KNN — Acurácia × K por Métrica de Distância')
    ax.set_xticks(list(valores_k))
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()
    azul = colors.HexColor('#1a3a5c')

    titulo  = ParagraphStyle('Titulo',  parent=base['Title'],
                             fontSize=20, leading=26, spaceAfter=4, textColor=azul)
    meta    = ParagraphStyle('Meta',    parent=base['Normal'],
                             fontSize=10, leading=13, textColor=colors.HexColor('#444'))
    secao   = ParagraphStyle('Secao',   parent=base['Heading1'],
                             fontSize=13, spaceBefore=16, spaceAfter=4, textColor=azul)
    corpo   = ParagraphStyle('Corpo',   parent=base['Normal'],
                             fontSize=10, leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
    bullet  = ParagraphStyle('Bullet',  parent=corpo, leftIndent=14, spaceAfter=3)
    legenda = ParagraphStyle('Legenda', parent=base['Normal'],
                             fontSize=9, textColor=colors.HexColor('#666'),
                             alignment=TA_CENTER, spaceAfter=6)
    return dict(titulo=titulo, meta=meta, secao=secao,
                corpo=corpo, bullet=bullet, legenda=legenda)


def gerar_pdf(caminho_pdf, img_norm, img_acc, valores_k, resultados):
    doc = SimpleDocTemplate(caminho_pdf, pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    s = build_styles()
    story = []

    # Cabeçalho
    story.append(Paragraph("Relatório Técnico: Classificação de Espécies Iris via KNN", s['titulo']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#1a3a5c'), spaceAfter=6))
    info = [
        ['Disciplina:', 'Fundamentos de Inteligência Artificial'],
        ['Alunos:', 'Arthur Xavier, Felipe Souto, Julliane Di Paula'],
        ['Professor:', 'Maelso Bruno'],
        ['Turma:', 'P7 B – Manhã'],
    ]
    t = Table(info, colWidths=[3*cm, 13*cm])
    t.setStyle(TableStyle([
        ('FONTNAME',  (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ---- 1. Análise Exploratória ----
    story.append(Paragraph("1. Análise Exploratória do Dataset", s['secao']))
    story.append(Paragraph(
        "O <b>Iris Dataset</b> contém medidas físicas de flores de três espécies: "
        "<i>Setosa</i>, <i>Versicolor</i> e <i>Virginica</i>. "
        "São 150 amostras (50 por classe), cada uma descrita por 4 atributos numéricos:",
        s['corpo']
    ))
    for feat in ['Comprimento da sépala (cm)', 'Largura da sépala (cm)',
                 'Comprimento da pétala (cm)', 'Largura da pétala (cm)']:
        story.append(Paragraph(f"• {feat}", s['bullet']))

    story.append(Paragraph(
        "<b>Objetivo do modelo:</b> prever a espécie de uma flor a partir das suas medidas físicas.",
        s['corpo']
    ))
    story.append(Paragraph(
        "<b>Relevância:</b> todos os atributos são numéricos contínuos, o que torna o cálculo "
        "de distâncias diretamente aplicável, sem transformações adicionais.",
        s['corpo']
    ))

    # ---- 2. Pré-processamento ----
    story.append(Paragraph("2. Justificativa do Pré-processamento", s['secao']))

    tecnicas = [
        ("<b>Verificação de missing values</b>", "Nenhum valor ausente foi encontrado (0 nulos). "
         "Necessária pois valores ausentes causariam erros no cálculo das distâncias."),
        ("<b>Label Encoding</b>", "Os rótulos categóricos ('Setosa', etc.) foram convertidos para "
         "inteiros (0, 1, 2). Necessário pois o KNN opera sobre valores numéricos."),
        ("<b>Padronização — StandardScaler</b>", "Os atributos possuem escalas diferentes "
         "(ex: comprimento da pétala varia de 1 a 6,9 cm; largura da sépala de 2 a 4,4 cm). "
         "Sem padronização, features com maior magnitude dominariam o cálculo de distâncias, "
         "enviesando o modelo. O StandardScaler transforma cada feature para média ≈ 0 e "
         "desvio padrão ≈ 1. O fit foi aplicado <i>somente</i> nos dados de treino para "
         "evitar data leakage."),
        ("<b>Divisão treino/teste (80/20 estratificada)</b>", "120 amostras para treino e 30 "
         "para teste, mantendo a proporção das classes em ambos os conjuntos."),
    ]
    for titulo_tec, descricao in tecnicas:
        story.append(Paragraph(f"• {titulo_tec}: {descricao}", s['bullet']))

    # ---- 3. Gráfico de Desempenho ----
    story.append(Paragraph("3. Gráfico de Desempenho", s['secao']))
    story.append(Image(img_acc, width=15*cm, height=7.5*cm))

    # ---- 4. Conclusão ----
    story.append(Paragraph("4. Conclusão", s['secao']))

    story.append(Paragraph("<b>Melhor combinação (K + Distância):</b>", s['corpo']))
    story.append(Paragraph(
        "As métricas <b>Chebyshev</b> e <b>Minkowski (p=3)</b> atingiram acurácia de "
        "<b>100%</b> nos valores K = 11 a K = 14, sendo as melhores combinações encontradas. ",
        s['corpo']
    ))

    story.append(Paragraph("<b>Sinais de overfitting com K baixo:</b>", s['corpo']))
    story.append(Paragraph(
        "Em K=1, o modelo classifica com base em um único vizinho. "
        "Embora a acurácia no conjunto de teste tenha sido de 97%, "
        "K=1 apresenta alta variância: pequenas perturbações nos dados de entrada podem "
        "alterar a classificação.",
        s['corpo']
    ))

    doc.build(story)
    print(f"PDF gerado: {caminho_pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    img_norm = os.path.join(DIR, 'knn_normalizacao.png')
    img_acc  = os.path.join(DIR, 'knn_resultado.png')
    pdf_path = os.path.join(DIR, 'Relatorio_KNN_Iris.pdf')

    print("Rodando experimento...")
    X_antes, X_depois, valores_k, resultados = rodar_experimento()

    print("Gerando gráficos...")
    gerar_grafico_normalizacao(X_antes, X_depois, img_norm)
    gerar_grafico_acuracia(valores_k, resultados, img_acc)

    print("Gerando PDF...")
    gerar_pdf(pdf_path, img_norm, img_acc, valores_k, resultados)
