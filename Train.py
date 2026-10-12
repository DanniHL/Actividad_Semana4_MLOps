"""Entrenamiento reproducible de Logistic Regression para South German Credit.

Ejemplo de ejecución desde la terminal de VS Code:
    python Train.py
    python Train.py --data-path "data/raw/SouthGermanCredit.asc" --C 5 --max-iter 1500 --random-state 21

Instalación de dependencias:
    pip install pandas numpy matplotlib seaborn scikit-learn mlflow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Permite guardar gráficas también en entornos sin interfaz gráfica.
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler


# Variables empleadas en el proyecto original.
NUMERIC_COLUMNS = ["duration", "amount", "age", "people_liable"]
ORDINAL_COLUMNS = [
    "employment_duration", "installment_rate", "present_residence",
    "property", "number_credits", "job",
]
CATEGORICAL_COLUMNS = [
    "status", "credit_history", "purpose", "savings", "personal_status_sex",
    "other_debtors", "other_installment_plans", "housing", "telephone",
    "foreign_worker",
]
RARE_COLUMNS = ["credit_history", "purpose", "other_debtors"]

COLUMN_MAPPING = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorías poco frecuentes, aprendiendo solo con los datos de fit."""

    def __init__(self, columns: Sequence[str] = RARE_COLUMNS, threshold: float = 0.05):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        self.frequent_categories_ = {}
        for column in self.columns:
            frequencies = X[column].value_counts(normalize=True, dropna=True)
            self.frequent_categories_[column] = set(frequencies[frequencies >= self.threshold].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()
        for column in self.columns:
            frequent = self.frequent_categories_[column]
            mask = (
                transformed[column].notna()
                & ~transformed[column].isin(frequent)
            )
            # Permite combinar valores numéricos y cadenas.
            transformed[column] = transformed[column].astype("object")
            transformed.loc[mask, column] = "other"
        # Evita conflictos entre valores numéricos y cadenas
        # en los codificadores de variables categóricas y ordinales.
        for column in CATEGORICAL_COLUMNS + ORDINAL_COLUMNS:
            transformed[column] = transformed[column].map(
                lambda value: str(value)
                if pd.notna(value)
                else np.nan
            )
        return transformed


def parse_arguments() -> argparse.Namespace:
    """Define los argumentos configurables desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa Logistic Regression para riesgo de crédito."
    )
    parser.add_argument(
        "--data-path", type=Path,
        default=Path("data/raw/SouthGermanCredit.asc"),
        help="Ruta al archivo SouthGermanCredit.asc (por defecto: data/raw/SouthGermanCredit.asc).",
    )
    parser.add_argument("--C", type=float, default=10.0, help="Inverso de la regularización (default: 10).")
    parser.add_argument("--max-iter", type=int, default=1000, help="Iteraciones máximas (default: 1000).")
    parser.add_argument("--random-state", type=int, default=6, help="Semilla aleatoria (default: 6).")
    parser.add_argument("--test-size", type=float, default=0.30, help="Fracción para prueba (default: 0.30).")
    parser.add_argument("--cv-splits", type=int, default=3, help="Pliegues de validación cruzada (default: 3).")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeticiones de validación cruzada (default: 3).")
    parser.add_argument("--rare-threshold", type=float, default=0.05, help="Umbral de categoría rara (default: 0.05).")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000", help="URI del servidor MLflow.")
    parser.add_argument("--experiment", default="Notebook-Baseline-Sem4", help="Nombre del experimento MLflow.")
    parser.add_argument("--eda", action="store_true", help="Genera gráficas descriptivas del conjunto de entrenamiento.")
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Carga los datos, renombra las columnas y define la etiqueta objetivo."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {data_path.resolve()}\n"
            "Coloca SouthGermanCredit.asc en data/raw/ o indica la ruta con --data-path."
        )
    df = pd.read_csv(data_path, sep=r"\s+", engine="python")
    df = df.rename(columns=COLUMN_MAPPING)
    if "credit_risk" not in df.columns:
        raise ValueError("No se encontró la columna objetivo 'kredit'/'credit_risk' en el dataset.")
    # Etiqueta original: 1 = buen cliente, 0 = mal cliente.
    # Etiqueta del proyecto: 1 = mal cliente, 0 = buen cliente.
    df["credit_risk"] = np.where(df["credit_risk"] == 1, 0, 1)
    missing_columns = set(NUMERIC_COLUMNS + ORDINAL_COLUMNS + CATEGORICAL_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {sorted(missing_columns)}")
    return df


def report_dataset(df: pd.DataFrame) -> None:
    """Muestra un resumen básico de calidad de datos."""
    print("\nDimensiones del dataset:", df.shape)
    print("\nValores por clase (0 = buen crédito, 1 = mal crédito):")
    print(df["credit_risk"].value_counts().sort_index())
    print("\nValores faltantes por columna:")
    print(df.isna().sum())
    print("\nFilas duplicadas:", df.duplicated().sum())


def split_dataset(df: pd.DataFrame, test_size: float, random_state: int):
    """Separa predictores y objetivo en conjuntos estratificados de train y test."""
    X = df.drop(columns="credit_risk")
    y = df["credit_risk"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=True, random_state=random_state
    )


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesamiento específico para variables numéricas y categóricas."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power_transform", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLUMNS),
        ("ord", ordinal_pipeline, ORDINAL_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def build_model_pipeline(args: argparse.Namespace) -> Pipeline:
    """Crea un pipeline completo para que el preprocesamiento ocurra dentro de cada fold."""
    return Pipeline([
        ("rare_categories", RareCategoryGrouper(threshold=args.rare_threshold)),
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            C=args.C, max_iter=args.max_iter, random_state=args.random_state
        )),
    ])


def evaluate_baseline(X_train, y_train, args: argparse.Namespace) -> None:
    """Evalúa un clasificador ingenuo que siempre predice la clase mayoritaria."""
    baseline = Pipeline([
        ("rare_categories", RareCategoryGrouper(threshold=args.rare_threshold)),
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.random_state
    )
    scores = cross_val_score(
        baseline, X_train, y_train, scoring=make_scorer(recall_score), cv=cv, n_jobs=-1
    )
    print("\nBaseline DummyClassifier, Recall por fold:", np.round(scores, 3))
    print(f"Recall promedio baseline: {scores.mean():.3f} (desv. estándar: {scores.std():.3f})")


def run_eda(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Genera gráficas exploratorias sencillas usando solo entrenamiento."""
    eda_df = X_train.copy()
    eda_df["credit_risk"] = y_train
    for column in NUMERIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data=X_train, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribución de {column}")
        fig.tight_layout()
        fig.savefig(f"eda_{column}.png", dpi=200)
        plt.close(fig)
    for column in CATEGORICAL_COLUMNS + ORDINAL_COLUMNS:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=X_train, x=column, ax=ax)
        ax.set_title(f"Frecuencia de {column}")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(f"eda_{column}.png", dpi=200)
        plt.close(fig)
    print("Gráficas EDA guardadas como eda_*.png")


def save_cv_plot(scores: np.ndarray, output_path: Path, C: float) -> None:
    """Guarda una gráfica de Recall por fold."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(scores) + 1), scores)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall por fold, Logistic Regression (C={C})")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """Guarda la matriz de confusión del conjunto de prueba."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred)).plot(ax=ax)
    ax.set_title("Matriz de confusión - Logistic Regression")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_evaluate_and_log(X_train, X_test, y_train, y_test, args: argparse.Namespace) -> None:
    """Ejecuta validación cruzada, entrena, evalúa en test y registra resultados en MLflow."""
    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.random_state
    )
    pipeline = build_model_pipeline(args)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        scoring=make_scorer(recall_score), cv=cv, n_jobs=-1,
    )
    print("\nRecall por fold:", np.round(scores, 3))
    print(f"Recall promedio CV: {scores.mean():.3f}")
    print(f"Desviación estándar CV: {scores.std():.3f}")

    pipeline.fit(X_train, y_train)
    predictions_train = pipeline.predict(X_train)
    predictions_test = pipeline.predict(X_test)
    train_precision = precision_score(y_train, predictions_train, zero_division=0)
    test_precision = precision_score(y_test, predictions_test, zero_division=0)
    test_recall = recall_score(y_test, predictions_test, zero_division=0)
    print(f"Precisión en train: {train_precision:.3f}")
    print(f"Recall en test: {test_recall:.3f}")
    print(f"Precisión en test: {test_precision:.3f}")

    cv_plot = Path("recall_cv.png")
    cm_plot = Path("confusion_matrix.png")
    save_cv_plot(scores, cv_plot, args.C)
    save_confusion_matrix(y_test, predictions_test, cm_plot)

    with mlflow.start_run(run_name=f"LogisticRegression_C_{args.C}"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("cv_splits", args.cv_splits)
        mlflow.log_param("cv_repeats", args.cv_repeats)
        mlflow.log_param("rare_threshold", args.rare_threshold)
        mlflow.log_param("metric", "Recall")
        mlflow.log_metric("recall_cv_mean", float(scores.mean()))
        mlflow.log_metric("recall_cv_std", float(scores.std()))
        mlflow.log_metric("precision_train", float(train_precision))
        mlflow.log_metric("recall_test", float(test_recall))
        mlflow.log_metric("precision_test", float(test_precision))
        mlflow.log_artifact(str(cv_plot))
        mlflow.log_artifact(str(cm_plot))
        mlflow.sklearn.log_model(sk_model=pipeline, name="model", skops_trusted_types=["__main__.RareCategoryGrouper","numpy.dtype",
                                                                                        ],
                                )
    print("\nExperimento y artefactos registrados en MLflow.")


def main() -> None:
    """Coordina las etapas principales del flujo de entrenamiento."""
    args = parse_arguments()
    if args.C <= 0:
        raise ValueError("--C debe ser mayor que cero.")
    if args.max_iter <= 0 or args.cv_splits < 2 or args.cv_repeats < 1:
        raise ValueError("--max-iter debe ser positivo, --cv-splits al menos 2 y --cv-repeats al menos 1.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe estar entre 0 y 1.")
    if not 0 <= args.rare_threshold <= 1:
        raise ValueError("--rare-threshold debe estar entre 0 y 1.")

    np.random.seed(args.random_state)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    df = load_dataset(args.data_path)
    report_dataset(df)
    X_train, X_test, y_train, y_test = split_dataset(
        df, test_size=args.test_size, random_state=args.random_state
    )
    print("\nDimensiones de train:", X_train.shape, "|", y_train.shape)
    print("Dimensiones de test:", X_test.shape, "|", y_test.shape)

    if args.eda:
        run_eda(X_train, y_train)
    evaluate_baseline(X_train, y_train, args)
    train_evaluate_and_log(X_train, X_test, y_train, y_test, args)


if __name__ == "__main__":
    main()
