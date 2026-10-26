import argparse
import os
import pandas as pd

from datasets_loader import carregar_dataset, carregar_datasets, listar_datasets
from reducers import run_reducer
from metrics import calcular_metricas
from visualization import plot_embedding

TECNICAS_DISPONIVEIS = [
    "PCA",
    "t-SNE",
    "MDS",
    "UMAP"
]

def executar_experimento(dataset_name, X, y, tecnicas, random_state, n_neighbors):
    resultados = []

    print(f"\nDataset: {dataset_name}")
    print(f"Amostras: {X.shape[0]}")
    print(f"Variáveis: {X.shape[1]}")

    if X.shape[0] < 4:
        print(
            f"  Dataset {dataset_name} possui apenas {X.shape[0]} amostras. "
            "Pulando execução por não haver amostras suficientes para as métricas."
        )

        for nome_tecnica in tecnicas:
            resultados.append({
                "dataset": dataset_name,
                "technique": nome_tecnica,
                "samples": X.shape[0],
                "features": X.shape[1],
                "execution_time": None,
                "stress": None,
                "trustworthiness": None,
                "continuity": None,
                "error": "Dataset com poucas amostras para cálculo das métricas."
            })

        return resultados

    for nome_tecnica in tecnicas:
        print(f"  Aplicando {nome_tecnica}...")

        try:
            embedding, tempo_execucao = run_reducer(
                nome_tecnica,
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
                nome_tecnica
            )

            result = {
                "dataset": dataset_name,
                "technique": nome_tecnica,
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
            print(f"    Erro ao aplicar {nome_tecnica} em {dataset_name}: {error}")

            resultados.append({
                "dataset": dataset_name,
                "technique": nome_tecnica,
                "samples": X.shape[0],
                "features": X.shape[1],
                "execution_time": None,
                "stress": None,
                "trustworthiness": None,
                "continuity": None,
                "error": str(error)
            })

    return resultados


def salvar_resultados(resultados, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results_df = pd.DataFrame(resultados)

    if os.path.exists(output_file):
        results_df.to_csv(
            output_file,
            mode="a",
            index=False,
            header=False
        )
    else:
        results_df.to_csv(
            output_file,
            index=False
        )


def criar_argumentos():
    parser = argparse.ArgumentParser(
        description="Executa técnicas de redução de dimensionalidade em datasets selecionados."
    )

    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all"] + listar_datasets(),
        help="Dataset que será usado. Use 'all' para executar todos."
    )

    parser.add_argument(
        "--technique",
        default="all",
        choices=["all"] + TECNICAS_DISPONIVEIS,
        help="Técnica de redução de dimensionalidade. Use 'all' para executar todas."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Quantidade máxima de amostras para datasets grandes."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade."
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Número de vizinhos usado nas métricas."
    )

    parser.add_argument(
        "--output",
        default="results/metrics_results.csv",
        help="Arquivo CSV de saída para salvar os resultados."
    )

    return parser.parse_args()


def main():
    os.makedirs("results", exist_ok=True)

    args = criar_argumentos()

    if args.technique == "all":
        tecnicas = TECNICAS_DISPONIVEIS
    else:
        tecnicas = [args.technique]

    resultados = []

    if args.dataset == "all":
        datasets = carregar_datasets(
            max_samples=args.max_samples,
            random_state=args.random_state
        )

        for dataset_name, (X, y) in datasets.items():
            resultados_dataset = executar_experimento(
                dataset_name,
                X,
                y,
                tecnicas,
                args.random_state,
                args.n_neighbors
            )

            resultados.extend(resultados_dataset)
            salvar_resultados(resultados_dataset, args.output)

    else:
        X, y = carregar_dataset(
            args.dataset,
            max_samples=args.max_samples,
            random_state=args.random_state
        )

        resultados = executar_experimento(
            args.dataset,
            X,
            y,
            tecnicas,
            args.random_state,
            args.n_neighbors
        )

        salvar_resultados(resultados, args.output)

    print("\nExperimentos concluídos.")
    print(f"Resultados salvos em: {args.output}")
    print("Gráficos salvos em: results/figures/")


if __name__ == "__main__":
    main()