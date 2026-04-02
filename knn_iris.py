import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Dataset: Iris (amostra reduzida — 10 amostras)
# Features: [comprimento_sepala, largura_sepala, comprimento_petala, largura_petala]
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [4.9, 3.0, 1.4, 0.2],  # Setosa
    [4.7, 3.2, 1.3, 0.2],  # Setosa
    [7.0, 3.2, 4.7, 1.4],  # Versicolor
    [6.4, 3.2, 4.5, 1.5],  # Versicolor
    [6.3, 3.3, 6.0, 2.5],  # Virginica
    [5.8, 2.7, 5.1, 1.9],  # Virginica
    [7.1, 3.0, 5.9, 2.1],  # Virginica
])

y_train_str = np.array([
    'Setosa', 'Setosa', 'Setosa',
    'Versicolor', 'Versicolor',
    'Virginica', 'Virginica', 'Virginica'
])

# 20% de teste (2 amostras)
X_test = np.array([
    [5.0, 3.4, 1.5, 0.2],  # esperado: Setosa
    [6.1, 2.8, 4.7, 1.2],  # esperado: Versicolor
])
y_test_str = np.array(['Setosa', 'Versicolor'])

# Mapeamento texto <-> número
map_classes = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
map_inv     = {v: k for k, v in map_classes.items()}

y_train = np.array([map_classes[l] for l in y_train_str])
y_test  = np.array([map_classes[l] for l in y_test_str])

scaler       = StandardScaler()
X_train_s    = scaler.fit_transform(X_train)
X_test_s     = scaler.transform(X_test)

k_values = range(1, len(X_train) + 1)

metricas = [
    ('euclidean',       {'metric': 'euclidean'}),
    ('manhattan',       {'metric': 'manhattan'}),
    ('chebyshev',       {'metric': 'chebyshev'}),
    ('minkowski(p=3)',  {'metric': 'minkowski', 'p': 3}),
]

resultados = {} 

print("=" * 60)
print("KNN - Dataset Iris (amostra reduzida)")
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
print("=" * 60)

for nome, params in metricas:
    acuracias = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, **params)
        knn.fit(X_train_s, y_train)
        acc   = knn.score(X_test_s, y_test)
        preds = [map_inv[p] for p in knn.predict(X_test_s)]
        acuracias.append(acc)
        print(f"Métrica: {nome:14s} | K: {k} | Acurácia: {acc:.2f} | Predições: {preds}")
    resultados[nome] = acuracias

print("=" * 60)

plt.figure(figsize=(9, 5))
cores   = ['steelblue', 'tomato', 'seagreen', 'darkorange']
marcadores = ['o', 's', '^', 'D']

for (nome, acuracias), cor, marcador in zip(resultados.items(), cores, marcadores):
    plt.plot(list(k_values), acuracias, label=nome, color=cor,
             marker=marcador, linewidth=2, markersize=7)

plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.title('KNN — Acurácia × K por Métrica de Distância')
plt.xticks(list(k_values))
plt.ylim(-0.05, 1.15)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('knn_resultado.png', dpi=150)
plt.show()
print("Gráfico salvo em: knn_resultado.png")
