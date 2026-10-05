import os
import matplotlib.pyplot as plt


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
