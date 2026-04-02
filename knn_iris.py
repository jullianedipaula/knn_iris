import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


SEPARADOR = "=" * 60

AMOSTRAS_TREINO = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
])

ROTULOS_TREINO = np.array([
    'Setosa', 'Setosa', 'Setosa',
    'Versicolor', 'Versicolor',
    'Virginica', 'Virginica', 'Virginica',
])

AMOSTRAS_TESTE = np.array([
    [5.0, 3.4, 1.5, 0.2],
    [6.1, 2.8, 4.7, 1.2],
])

ROTULOS_TESTE = np.array(['Setosa', 'Versicolor'])

MAPA_CLASSE_PARA_NUMERO = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
MAPA_NUMERO_PARA_CLASSE = {numero: classe for classe, numero in MAPA_CLASSE_PARA_NUMERO.items()}

METRICAS = [
    ('euclidean',      {'metric': 'euclidean'}),
    ('manhattan',      {'metric': 'manhattan'}),
    ('chebyshev',      {'metric': 'chebyshev'}),
    ('minkowski(p=3)', {'metric': 'minkowski', 'p': 3}),
]

CORES_GRAFICO     = ['steelblue', 'tomato', 'seagreen', 'darkorange']
MARCADORES_GRAFICO = ['o', 's', '^', 'D']


def converter_rotulos_para_numeros(rotulos):
    return np.array([MAPA_CLASSE_PARA_NUMERO[rotulo] for rotulo in rotulos])


def converter_numeros_para_rotulos(numeros):
    return [MAPA_NUMERO_PARA_CLASSE[numero] for numero in numeros]


def padronizar_dados(amostras_treino, amostras_teste):
    normalizador = StandardScaler()
    treino_normalizado = normalizador.fit_transform(amostras_treino)
    teste_normalizado  = normalizador.transform(amostras_teste)
    return treino_normalizado, teste_normalizado


def treinar_e_avaliar(amostras_treino, rotulos_treino, amostras_teste, rotulos_teste, k, params_metrica):
    modelo = KNeighborsClassifier(n_neighbors=k, **params_metrica)
    modelo.fit(amostras_treino, rotulos_treino)
    acuracia  = modelo.score(amostras_teste, rotulos_teste)
    predicoes = converter_numeros_para_rotulos(modelo.predict(amostras_teste))
    return acuracia, predicoes


def executar_experimento(amostras_treino, rotulos_treino, amostras_teste, rotulos_teste):
    valores_de_k = range(1, len(amostras_treino) + 1)
    resultados   = {}

    for nome_metrica, params_metrica in METRICAS:
        acuracias_por_k = []

        for k in valores_de_k:
            acuracia, predicoes = treinar_e_avaliar(
                amostras_treino, rotulos_treino,
                amostras_teste,  rotulos_teste,
                k, params_metrica
            )
            acuracias_por_k.append(acuracia)
            print(f"Métrica: {nome_metrica:14s} | K: {k} | Acurácia: {acuracia:.2f} | Predições: {predicoes}")

        resultados[nome_metrica] = acuracias_por_k

    return valores_de_k, resultados


def exibir_grafico(valores_de_k, resultados):
    plt.figure(figsize=(9, 5))

    for (nome_metrica, acuracias), cor, marcador in zip(resultados.items(), CORES_GRAFICO, MARCADORES_GRAFICO):
        plt.plot(list(valores_de_k), acuracias, label=nome_metrica,
                 color=cor, marker=marcador, linewidth=2, markersize=7)

    plt.xlabel('Valor de K')
    plt.ylabel('Acurácia')
    plt.title('KNN — Acurácia × K por Métrica de Distância')
    plt.xticks(list(valores_de_k))
    plt.ylim(-0.05, 1.15)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('knn_resultado.png', dpi=150)
    plt.show()
    print("Gráfico salvo em: knn_resultado.png")


def main():
    rotulos_treino = converter_rotulos_para_numeros(ROTULOS_TREINO)
    rotulos_teste  = converter_rotulos_para_numeros(ROTULOS_TESTE)

    amostras_treino, amostras_teste = padronizar_dados(AMOSTRAS_TREINO, AMOSTRAS_TESTE)

    print(SEPARADOR)
    print("KNN - Dataset Iris (amostra reduzida)")
    print(f"Treino: {len(amostras_treino)} amostras | Teste: {len(amostras_teste)} amostras")
    print(SEPARADOR)

    valores_de_k, resultados = executar_experimento(
        amostras_treino, rotulos_treino,
        amostras_teste,  rotulos_teste
    )

    print(SEPARADOR)
    exibir_grafico(valores_de_k, resultados)


if __name__ == '__main__':
    main()
