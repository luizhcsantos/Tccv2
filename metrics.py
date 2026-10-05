import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def calcular_stress(X_original, X_embedded):
    distancias_originais = pairwise_distances(X_original)
    distancias_embedded = pairwise_distances(X_embedded)

    numerador = np.sum((distancias_originais - distancias_embedded) ** 2)
    denominador = np.sum(distancias_originais ** 2)

    if denominador == 0:
        return 0.0


    return np.sqrt(numerador / denominador)

def calcular_trustworthiness(X_original, X_embedded, n_neighbors=10):
    return trustworthiness(
        X_original,
        X_embedded,
        n_neighbors=n_neighbors
    )

def calcular_continuity(X_original, X_embedded, n_neighbors=10):
    n_samples = X_original.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 2)

    dist_original = pairwise_distances(X_original)
    dist_embedded = pairwise_distances(X_embedded)

    ordem_original = np.argsort(dist_original, axis=1)
    ordem_embedded = np.argsort(dist_embedded, axis=1)

    vizinhos_original = ordem_original[:, 1:n_neighbors + 1]
    vizinhos_embedded = ordem_embedded[:, 1:n_neighbors + 1]

    ranks_embedded = np.zeros((n_samples, n_samples), dtype=int)

    for i in range(n_samples):
        for rank, index in enumerate(ordem_embedded[i]):
            ranks_embedded[i, index] = rank

    penalty = 0.0

    for i in range(n_samples):
        conjunto_original = set(vizinhos_original[i])
        conjunto_embedded = set(vizinhos_embedded[i])

        vizinhos_perdidos = conjunto_original - conjunto_embedded

        for vizinho in vizinhos_perdidos:
            penalty += ranks_embedded[i, vizinho] - n_neighbors

    denominador = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)

    if denominador == 0:
        return 1.0

    continuity = 1 - ((2 / denominador) * penalty)

    return continuity


def calcular_metricas(X_original, X_embedded, n_neighbors=10):
    return {
        "stress": calcular_stress(X_original, X_embedded),
        "trustworthiness": calcular_trustworthiness(
            X_original,
            X_embedded,
            n_neighbors=n_neighbors
        ),
        "continuity": calcular_continuity(
            X_original,
            X_embedded,
            n_neighbors=n_neighbors
        )
    }