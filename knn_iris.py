import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


SEPARADOR = "=" * 60

NOMES_CLASSES = ['Setosa', 'Versicolor', 'Virginica']

MAPA_NUMERO_PARA_CLASSE = {i: nome for i, nome in enumerate(NOMES_CLASSES)}

METRICAS = [
    ('euclidean',      {'metric': 'euclidean'}),
    ('manhattan',      {'metric': 'manhattan'}),
    ('chebyshev',      {'metric': 'chebyshev'}),
    ('minkowski(p=3)', {'metric': 'minkowski', 'p': 3}),
]

CORES_GRAFICO      = ['steelblue', 'tomato', 'seagreen', 'darkorange']
MARCADORES_GRAFICO = ['o', 's', '^', 'D']


def carregar_e_verificar_dados():
    iris = load_iris()
    X = iris.data
    y = np.array([NOMES_CLASSES[rotulo] for rotulo in iris.target])

    missing = np.isnan(X).sum()
    print(f"Verificação de missing values: {missing} valor(es) ausente(s) encontrado(s).")

    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} atributos.")
    for nome in NOMES_CLASSES:
        print(f"  Classe '{nome}': {np.sum(y == nome)} amostras")

    return X, y


def dividir_dados(X, y):
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Divisão: {len(X_treino)} amostras de treino | {len(X_teste)} amostras de teste")
    return X_treino, X_teste, y_treino, y_teste


def converter_rotulos_para_numeros(rotulos):
    mapa = {nome: i for i, nome in enumerate(NOMES_CLASSES)}
    return np.array([mapa[r] for r in rotulos])


def converter_numeros_para_rotulos(numeros):
    return [MAPA_NUMERO_PARA_CLASSE[n] for n in numeros]


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
    valores_de_k = range(1, 16)
    resultados   = {}

    for nome_metrica, params_metrica in METRICAS:
        acuracias_por_k = []

        for k in valores_de_k:
            acuracia, _ = treinar_e_avaliar(
                amostras_treino, rotulos_treino,
                amostras_teste,  rotulos_teste,
                k, params_metrica
            )
            acuracias_por_k.append(acuracia)
            print(f"Métrica: {nome_metrica:14s} | K: {k:2d} | Acurácia: {acuracia:.2f}")

        resultados[nome_metrica] = acuracias_por_k

    return valores_de_k, resultados


def exibir_grafico(valores_de_k, resultados):
    plt.figure(figsize=(10, 5))

    for (nome_metrica, acuracias), cor, marcador in zip(resultados.items(), CORES_GRAFICO, MARCADORES_GRAFICO):
        plt.plot(list(valores_de_k), acuracias, label=nome_metrica,
                 color=cor, marker=marcador, linewidth=2, markersize=7)

    plt.xlabel('Valor de K')
    plt.ylabel('Acurácia')
    plt.title('KNN — Acurácia × K por Métrica de Distância')
    plt.xticks(list(valores_de_k))
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('knn_resultado.png', dpi=150)
    plt.show()
    print("Gráfico salvo em: knn_resultado.png")


def main():
    print(SEPARADOR)
    print("KNN - Dataset Iris Completo")
    print(SEPARADOR)

    X, y = carregar_e_verificar_dados()

    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

    rotulos_treino = converter_rotulos_para_numeros(y_treino)
    rotulos_teste  = converter_rotulos_para_numeros(y_teste)

    X_treino, X_teste = padronizar_dados(X_treino, X_teste)

    print(SEPARADOR)
    valores_de_k, resultados = executar_experimento(
        X_treino, rotulos_treino,
        X_teste,  rotulos_teste
    )

    print(SEPARADOR)
    exibir_grafico(valores_de_k, resultados)


if __name__ == '__main__':
    main()
