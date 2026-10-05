import os
import pandas as pd

from datasets_loader import carregar_datasets
from reducers import run_reducer
from metrics import calcular_metricas
from visualization import plot_embedding

import sys

print(sys.executable)
    
def main():
    os.makedirs("results", exist_ok=True)

    random_state = 42
    n_neighbors = 10

    datasets = carregar_datasets()

    tecnicas = [
        "PCA",
        "t-SNE",
        "MDS",
        "UMAP"
    ]

    resultados = []

    for dataset_name, (X, y) in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print(f"Amostras: {X.shape[0]}")
        print(f"Variáveis: {X.shape[1]}")

        for nome_tecnicas in tecnicas:
            print(f"  Aplicando {nome_tecnicas}...")

            try:
                embedding, tempo_execucao = run_reducer(
                    nome_tecnicas,
                    X,
                    random_state=random_state
                )

                metrics = calcular_metricas(
                    X,
                    embedding,
                    n_neighbors=n_neighbors
                )

                plot_embedding(
                    embedding,
                    y,
                    dataset_name,
                    nome_tecnicas
                )

                result = {
                    "dataset": dataset_name,
                    "technique": nome_tecnicas,
                    "samples": X.shape[0],
                    "features": X.shape[1],
                    "execution_time": tempo_execucao,
                    "stress": metrics["stress"],
                    "trustworthiness": metrics["trustworthiness"],
                    "continuity": metrics["continuity"]
                }

                resultados.append(result)

                print(
                    f"    Tempo: {tempo_execucao:.4f}s | "
                    f"Stress: {metrics['stress']:.4f} | "
                    f"Trustworthiness: {metrics['trustworthiness']:.4f} | "
                    f"Continuity: {metrics['continuity']:.4f}"
                )

            except Exception as error:
                print(f"    Erro ao aplicar {nome_tecnicas} em {dataset_name}: {error}")

                resultados.append({
                    "dataset": dataset_name,
                    "technique": nome_tecnicas,
                    "samples": X.shape[0],
                    "features": X.shape[1],
                    "execution_time": None,
                    "stress": None,
                    "trustworthiness": None,
                    "continuity": None,
                    "error": str(error)
                })

    results_df = pd.DataFrame(resultados)
    results_df.to_csv("results/metrics_results.csv", index=False)

    print("\nExperimentos concluídos.")
    print("Resultados salvos em: results/metrics_results.csv")
    print("Gráficos salvos em: results/figures/")


if __name__ == "__main__":
    main()