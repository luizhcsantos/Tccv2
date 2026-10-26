import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

try:
    import umap
except ImportError:
    umap = None


def apply_pca(X, random_state=42):

    model = PCA(n_components=2, random_state=random_state)
    return model.fit_transform(X)


def apply_tsne(X, random_state=42):
    perplexity = min(30, max(5, (X.shape[0] - 1) // 3))

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state)
    return model.fit_transform(X)

def apply_mds(X, random_state=42):
    model = MDS(
        n_components=2,
        normalized_stress="auto",
        random_state=random_state,
        n_init=4,
        max_iter=300)
    return model.fit_transform(X)

def apply_umap(X, random_state=42):
    if umap is None:
        raise ImportError("O pacote 'umap-learn' não está instalado.")

    n_neighbors = min(15, max(2, X.shape[0] - 1))

    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=random_state)
    return model.fit_transform(X)


def run_reducer(name, X, random_state):
    start_time = time.perf_counter()

    if name == "PCA":
        embedding = apply_pca(X, random_state)
    elif name == "t-SNE":
        embedding = apply_tsne(X, random_state)
    elif name == "MDS":
        embedding = apply_mds(X, random_state)
    elif name == "UMAP":
        embedding = apply_umap(X, random_state)
    else:
        raise ValueError(f"Técnica desconhecida: {name}")

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return embedding, execution_time