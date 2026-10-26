import os
import zipfile

import pandas as pd
import numpy as np
from PIL import Image

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASETS_DIR = "datasets_locais"

def prepare_dataset(data, target, feature_names=None,
                    max_samples=None, random_state=42):
    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(target)

    if max_samples is not None and len(X) > max_samples:
        X, y = sample_stratified(X, y, max_samples, random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values

def sample_stratified(X, y, max_samples, random_state):
    X = pd.DataFrame(X)
    y = pd.Series(y).reset_index(drop=True)

    sampled = X.copy()
    sampled["target"] = y.values

    n_classes = sampled["target"].nunique()
    samples_per_class = max(1, max_samples // n_classes)
    min_class_size = sampled["target"].value_counts().min()
    n = min(samples_per_class, min_class_size)

    sampled = sampled.groupby("target", group_keys=False).sample(
        n=n,
        random_state=random_state
    )

    X_sampled = sampled.drop(columns=["target"])
    y_sampled = sampled["target"]

    return X_sampled, y_sampled


def encode_labels(labels):
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

def prepare_numeric_dataframe(df, target_column,
                              drop_columns=None,
                               max_samples=None,
                              random_state=42):
    if drop_columns is None:
        drop_columns = []

    df = df.copy()
    df = df.drop(columns=drop_columns, errors="ignore")
    df = df.dropna(axis=0, how="any")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=False)
    y = encode_labels(y)

    return prepare_dataset(
        X,
        y,
        feature_names=X.columns,
        max_samples=max_samples,
        random_state=random_state
    )

def prepare_text_dataset(texts, labels,
                         max_features=1000,
                         max_samples=None,
                         random_state=42):

    texts = pd.Series(texts).fillna("").astype(str)
    labels = pd.Series(labels)

    if max_samples is not None and len(texts) > max_samples:
        X_temp = pd.DataFrame({"text": texts})
        X_sampled, y_sampled = sample_stratified(
            X_temp,
            labels,
            max_samples=max_samples,
            random_state=random_state
        )
        texts = X_sampled["text"]
        labels = y_sampled

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True
    )

    X = vectorizer.fit_transform(texts).toarray()
    y = encode_labels(labels)

    return X, y


def carregar_bank_marketing(max_samples=3000, random_state=42):
    path = os.path.join(
        DATASETS_DIR,
        "bank_marketing",
        "bank-full.csv"
    )

    df = pd.read_csv(path, sep=";")

    return prepare_numeric_dataframe(
        df,
        target_column="y",
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_fashion_mnist(max_samples=3000, random_state=42):
    path = os.path.join(
        DATASETS_DIR,
        "fashion_mnist",
        "fashionmnist.csv",
        "fashion-mnist_train.csv"
    )

    df = pd.read_csv(path)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    X = X / 255.0

    return prepare_dataset(
        X,
        y,
        feature_names=X.columns,
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_seizure(max_samples=3000, random_state=42):
    path = os.path.join(
        DATASETS_DIR,
        "seizure",
        "seizure_data",
        "Epileptic Seizure Recognition.csv"
    )

    df = pd.read_csv(path)

    target_column = "y" if "y" in df.columns else df.columns[-1]
    drop_columns = [df.columns[0]]

    return prepare_numeric_dataframe(
        df,
        target_column=target_column,
        drop_columns=drop_columns,
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_orl_faces(random_state=42):
    base_dir = os.path.join(
        DATASETS_DIR,
        "orl_faces",
        "ORL",
        "att-database-of-faces"
    )

    images = []
    labels = []

    for subject_dir in sorted(os.listdir(base_dir)):
        subject_path = os.path.join(base_dir, subject_dir)

        if not os.path.isdir(subject_path) or not subject_dir.startswith("s"):
            continue

        label = int(subject_dir.replace("s", ""))

        for filename in sorted(os.listdir(subject_path)):
            if not filename.lower().endswith(".pgm"):
                continue

            image_path = os.path.join(subject_path, filename)

            image = Image.open(image_path).convert("L")
            image_array = np.asarray(image, dtype=np.float32).flatten()

            images.append(image_array)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    return prepare_dataset(
        X,
        y,
        max_samples=None,
        random_state=random_state
    )


def carregar_spam(max_samples=3000, random_state=42):
    path = os.path.join(
        DATASETS_DIR,
        "spam",
        "spam",
        "spam.csv"
    )

    df = pd.read_csv(
        path,
        encoding="latin-1"
    )

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]

    df.columns = ["label", "text"]

    return prepare_text_dataset(
        df["text"],
        df["label"],
        max_features=1000,
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_hate_speech(max_samples=3000, random_state=42):
    path = os.path.join(
        DATASETS_DIR,
        "hate_speech",
        "labeled_data.csv"
    )

    df = pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip"
    )

    text_column = "tweet" if "tweet" in df.columns else df.columns[-1]
    target_column = "class" if "class" in df.columns else df.columns[-2]

    return prepare_text_dataset(
        df[text_column],
        df[target_column],
        max_features=1000,
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_sentiment(max_samples=3000, random_state=42):
    zip_path = os.path.join(
        DATASETS_DIR,
        "sentiment",
        "sentiment",
        "sentiment+labelled+sentences.zip"
    )

    texts = []
    labels = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        file_names = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".txt")
        ]

        for file_name in file_names:
            with zip_file.open(file_name) as file:
                for raw_line in file:
                    line = raw_line.decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    parts = line.rsplit("\t", 1)

                    if len(parts) != 2:
                        continue

                    text, label = parts

                    if label not in ["0", "1"]:
                        continue

                    texts.append(text)
                    labels.append(label)

    if len(texts) == 0:
        raise ValueError(
            "Nenhuma amostra foi carregada para o dataset sentiment. "
            "Verifique se o arquivo ZIP contém arquivos .txt com linhas no formato: texto<TAB>classe."
        )

    print(f"Sentiment carregado com {len(texts)} amostras.")

    return prepare_text_dataset(
        texts,
        labels,
        max_features=1000,
        max_samples=max_samples,
        random_state=random_state
    )
    zip_path = os.path.join(
        DATASETS_DIR,
        "sentiment",
        "sentiment",
        "sentiment+labelled+sentences.zip"
    )

    texts = []
    labels = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        file_names = [
            name for name in zip_file.namelist()
            if name.endswith(".txt")
        ]

        for file_name in file_names:
            with zip_file.open(file_name) as file:
                for raw_line in file:
                    line = raw_line.decode("utf-8", errors="ignore").strip()

                    if not line:
                        continue

                    parts = line.rsplit("\t", 1)

                    if len(parts) != 2:
                        continue

                    text, label = parts
                    texts.append(text)
                    labels.append(label)

    return prepare_text_dataset(
        texts,
        labels,
        max_features=1000,
        max_samples=max_samples,
        random_state=random_state
    )

def carregar_iris(random_state=42):
    iris = load_iris()
    return prepare_dataset(
        iris.data,
        iris.target,
        iris.feature_names,
        random_state=random_state
    )


def carregar_wine(random_state=42):
    wine = load_wine()
    return prepare_dataset(
        wine.data,
        wine.target,
        wine.feature_names,
        random_state=random_state
    )


def carregar_breast_cancer(random_state=42):
    breast_cancer = load_breast_cancer()
    return prepare_dataset(
        breast_cancer.data,
        breast_cancer.target,
        breast_cancer.feature_names,
        random_state=random_state
    )


def carregar_digits(random_state=42):
    digits = load_digits()
    return prepare_dataset(
        digits.data,
        digits.target,
        digits.feature_names,
        random_state=random_state
    )


DATASET_LOADERS = {
    "iris": carregar_iris,
    "wine": carregar_wine,
    "breast_cancer": carregar_breast_cancer,
    "digits": carregar_digits,
    "bank_marketing": carregar_bank_marketing,
    "fashion_mnist": carregar_fashion_mnist,
    "seizure": carregar_seizure,
    "orl_faces": carregar_orl_faces,
    "spam": carregar_spam,
    "hate_speech": carregar_hate_speech,
    "sentiment": carregar_sentiment,
}

def listar_datasets():
    return list(DATASET_LOADERS.keys())


def carregar_dataset(nome, max_samples=1000, random_state=42):
    if nome not in DATASET_LOADERS:
        datasets_disponiveis = ", ".join(listar_datasets())
        raise ValueError(
            f"Dataset desconhecido: {nome}."
            f"Datasets disponíveis: {datasets_disponiveis}"
        )

    loader = DATASET_LOADERS[nome]

    if nome in ["iris", "wine", "breast_cancer", "digits", "orl_faces"]:
        return loader(random_state=random_state)

    return loader(
        max_samples=max_samples,
        random_state=random_state
    )


def carregar_datasets(nomes=None, max_samples=1000, random_state=42):
    if nomes is None:
        nomes = listar_datasets()

    datasets = {}

    for nome in nomes:
        datasets[nome] = carregar_dataset(
            nome,
            max_samples,
            random_state)

    return datasets