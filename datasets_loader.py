import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

def prepare_dataset(data, target, feature_names=None,
                    max_samples=None, random_state=42):
    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(target)

    if max_samples is not None and len(X) > max_samples:
        sampled = X.copy()
        sampled["target"] = y.values

        sampled = sampled.groupby("target", group_keys=False).sample(
            n=min(max_samples // sampled["target"].nunique(),
                  sampled["target"].value_counts().min()),
            random_state=random_state
        )

        X = sampled.drop(columns=["target"])
        y = sampled["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


def carregar_datasets():

    datasets = {}

    iris = load_iris()
    datasets["iris"] = prepare_dataset(
        iris.data,
        iris.target,
        iris.feature_names
    )

    wine = load_wine()
    datasets["wine"] = prepare_dataset(
        wine.data,
        wine.target,
        wine.feature_names
    )

    breast_cancer = load_breast_cancer()
    datasets["breast_cancer"] = prepare_dataset(
        breast_cancer.data,
        breast_cancer.target,
        breast_cancer.feature_names
    )

    digits = load_digits()
    datasets["digits"] = prepare_dataset(
        digits.data,
        digits.target,
        digits.feature_names
    )

    return datasets