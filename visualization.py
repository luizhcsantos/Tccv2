import os
import matplotlib.pyplot as plt
from numpy.ma.core import squeeze


def plot_embedding(embedding, labels, dataset_name, technique_name, output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=20,
        alpha=0.8
    )

    plt.title(f"{technique_name} - {dataset_name}")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.colorbar(scatter, label="Classe")
    plt.grid(True, alpha=0.3)

    filename = f"{dataset_name}_{technique_name}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_embeddings_matrix(embeddings_results, output_path="results/figures/comparacao_matriz.png"):

    if not embeddings_results:
        print("Nenhum resultado de embedding encontrado.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset_names = sorted({
        item["dataset"] for item in embeddings_results
    })

    technique_names = sorted({
        item["technique"] for item in embeddings_results
    })

    technique_order = ["PCA", "t-SNE", "MDS", "UMAP"]
    technique_names = [
        technique for technique in technique_order
        if technique in technique_names
    ]

    n_rows = len(dataset_names)
    n_cols = len(technique_names)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )

    for row_index, dataset_name in enumerate(dataset_names):
        for col_index, technique_name in enumerate(technique_names):
            ax = axes[row_index][col_index]

            result = next(
                (
                    item for item in embeddings_results
                    if item["dataset"] == dataset_name
                       and item["technique"] == technique_name
                ),
                None
            )

            if result is None:
                ax.axis("off")
                ax.set_title(f"{dataset_name} - {technique_name}\nSem resultado")
                continue

            embedding = result["embedding"]
            labels = result["labels"]

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="tab10",
                s=10,
                alpha=0.8
            )

            if row_index == 0:
                ax.set_title(technique_name, fontsize=14)

            if col_index == 0:
                ax.set_ylabel(dataset_name, fontsize=14)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)

    fig.suptitle("Comparação visual das técnicas de redução de dimensionalidade", fontsize=18)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Matriz de comparação salva em: {output_path}")