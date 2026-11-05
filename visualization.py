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
        cmap="nipy_spectral",
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

def plot_embeddings_matrix(
        embeddings_results,
        output_path="results/figures/comparacao_matriz.png",
        title="Comparação visual das técnicas de redução de dimensionalidade",
        dataset_order=None,
        technique_order=None):
    if not embeddings_results:
        print("Nenhum embedding disponível para gerar a matriz de comparação.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if dataset_order is None:
        dataset_names = sorted({
            item["dataset"] for item in embeddings_results
        })
    else:
        available_datasets = {
            item["dataset"] for item in embeddings_results
        }
        dataset_names = [
            dataset for dataset in dataset_order
            if dataset in available_datasets
        ]

    if technique_order is None:
        technique_order = ["PCA", "t-SNE", "MDS", "UMAP"]

    available_techniques = {
        item["technique"] for item in embeddings_results
    }

    technique_names = [
        technique for technique in technique_order
        if technique in available_techniques
    ]

    n_rows = len(dataset_names)
    n_cols = len(technique_names)

    if n_rows == 0 or n_cols == 0:
        print("Não há datasets ou técnicas suficientes para gerar a matriz.")
        return

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 5 * n_rows),
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

                if row_index == 0:
                    ax.set_title(technique_name, fontsize=15, fontweight="bold")

                if col_index == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        dataset_name,
                        transform=ax.transAxes,
                        fontsize=14,
                        fontweight="bold",
                        va="center",
                        ha="right",
                        rotation=90
                    )

                continue

            embedding = result["embedding"]
            labels = result["labels"]

            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="nipy_spectral",
                s=12,
                alpha=0.85,
                edgecolors="none"
            )

            if row_index == 0:
                ax.set_title(technique_name, fontsize=16, fontweight="bold")

            if col_index == 0:
                ax.text(
                    -0.08,
                    0.5,
                    dataset_name,
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.18)

    fig.suptitle(title, fontsize=20, fontweight="bold")

    plt.tight_layout(rect=(0.03, 0, 1, 0.96))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_output_path, bbox_inches="tight")

    plt.close()

    print(f"Matriz de comparação salva em: {output_path}")
    print(f"Matriz de comparação salva em: {pdf_output_path}")


def plot_embeddings_matrices_by_group(
        embeddings_results,
        groups,
        output_dir="results/figures"
):
    os.makedirs(output_dir, exist_ok=True)

    for group_name, dataset_names in groups.items():
        group_embeddings = [
            item for item in embeddings_results
            if item["dataset"] in dataset_names
        ]

        if not group_embeddings:
            print(f"Nenhum embedding disponível para o grupo: {group_name}")
            continue

        output_path = os.path.join(
            output_dir,
            f"comparacao_matriz_{group_name}.png"
        )

        title = f"Comparação visual - {group_name.replace('_', ' ').title()}"

        plot_embeddings_matrix(
            group_embeddings,
            output_path=output_path,
            title=title,
            dataset_order=dataset_names
        )