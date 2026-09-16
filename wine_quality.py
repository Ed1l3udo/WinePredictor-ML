"""Wine Quality ML pipeline with from-scratch model implementations.

Run:
    python wine_quality.py

The script intentionally avoids importing ready-made ML models. NumPy is used
for linear algebra and vectorized math; Pandas/Matplotlib/Seaborn are used for
data loading and visualization.
"""

from __future__ import annotations

import os
import re
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CLASS_LABELS = ["ruim (<=5)", "mediano (=6)", "bom (>=7)"]
N_CLASSES = len(CLASS_LABELS)
MEDIAN_QUALITY_CENTER = 6.0

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RED_CSV = BASE_DIR / "winequality-red.csv"
WHITE_CSV = BASE_DIR / "winequality-white.csv"

RF_N_ESTIMATORS = 120
RF_TUNING_N_ESTIMATORS = 60
RF_TUNING_FOLDS = 3
RF_N_SPLIT_CANDIDATES = 32
RF_SEARCH_SPACE = [
    {"max_depth": 10, "min_samples_split": 16, "min_samples_leaf": 6, "max_features": "sqrt"},
    {"max_depth": 14, "min_samples_split": 12, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"max_depth": 18, "min_samples_split": 8, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"max_depth": 14, "min_samples_split": 12, "min_samples_leaf": 4, "max_features": "all"},
]
LOGISTIC_L2_STRENGTHS = np.logspace(-4, 0, 9)
K_VALUES = [3, 5, 7, 9, 11]
REPEATED_SPLIT_SEEDS = [11, 23, 42, 67, 101]
CI_Z_VALUE = 1.96


def quality_to_class(quality_values: np.ndarray | pd.Series) -> np.ndarray:
    """Map quality scores to ruim/mediano/bom classes.

    Integer labels use the project rule directly:
    quality <= 5 -> ruim, quality == 6 -> mediano, quality >= 7 -> bom.
    Continuous regressor outputs use the equivalent cut points 5.5 and 6.5.
    """
    values = np.asarray(quality_values, dtype=float)
    classes = np.ones(values.shape, dtype=int)
    classes[values <= 5.5] = 0
    classes[values >= 6.5] = 2
    return classes


def quality_scores_to_class_scores(quality_scores: np.ndarray) -> np.ndarray:
    """Build one-vs-rest ranking scores from continuous quality predictions."""
    scores = np.asarray(quality_scores, dtype=float)
    return np.column_stack(
        [
            -scores,
            -np.abs(scores - MEDIAN_QUALITY_CENTER),
            scores,
        ]
    )


def safe_name(name: str) -> str:
    """Return a filesystem-safe stem for plot filenames."""
    normalized = name.lower().replace("->", "to")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def save_current_plot(filename: str) -> None:
    """Save the current matplotlib figure under images/ and close it."""
    IMAGE_DIR.mkdir(exist_ok=True)
    path = IMAGE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def load_wine_data() -> pd.DataFrame:
    """Load red and white wine data, add is_red, and concatenate them."""
    if not RED_CSV.exists() or not WHITE_CSV.exists():
        raise FileNotFoundError("Expected winequality-red.csv and winequality-white.csv in the project root.")

    red = pd.read_csv(RED_CSV, sep=";")
    white = pd.read_csv(WHITE_CSV, sep=";")
    red["is_red"] = 1
    white["is_red"] = 0

    df = pd.concat([red, white], axis=0, ignore_index=True)
    df["quality"] = df["quality"].astype(float)
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Print exploratory analysis and save visual summaries."""
    print("\n" + "=" * 80)
    print("1. Exploratory Data Analysis")
    print("=" * 80)
    print(f"\nShape: {df.shape}\n")
    print("Dtypes:")
    print(df.dtypes)
    print("\nDescriptive statistics:")
    print(df.describe().T)

    quality_counts = df["quality"].value_counts().sort_index()
    class_target = quality_to_class(df["quality"])
    class_counts = pd.Series(class_target).value_counts().reindex(range(N_CLASSES), fill_value=0)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["quality"],
        bins=np.arange(df["quality"].min() - 0.5, df["quality"].max() + 1.5, 1),
        color="#2f6f73",
    )
    plt.title("Distribuicao das notas de qualidade")
    plt.xlabel("Nota de qualidade")
    plt.ylabel("Contagem")
    save_current_plot("quality_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=CLASS_LABELS,
        y=[class_counts.get(label, 0) for label in range(N_CLASSES)],
        palette=["#9f4c4c", "#c7a33a", "#2f6f73"],
        hue=CLASS_LABELS,
        legend=False,
    )
    plt.title("Distribuicao do alvo em tres classes")
    plt.xlabel("Classe")
    plt.ylabel("Contagem")
    save_current_plot("class_distribution_three_classes.png")

    corr_df = df.copy()
    corr_df["quality_class"] = class_target
    corr = corr_df.corr(method="pearson")

    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, square=False, linewidths=0.4)
    plt.title("Mapa de correlacao de Pearson")
    save_current_plot("correlation_heatmap.png")

    # EDA note: total sulfur dioxide includes the free form, so these two
    # measurements are expected to be positively correlated.
    free_total_corr = corr.loc["free sulfur dioxide", "total sulfur dioxide"]
    print(
        "\nCorrelation note - free sulfur dioxide vs total sulfur dioxide: "
        f"{free_total_corr:.3f}"
    )

    # EDA note: alcohol tends to reduce density, while residual sugar tends to
    # increase it; both relations are relevant for the wine chemistry context.
    density_alcohol_corr = corr.loc["density", "alcohol"]
    density_sugar_corr = corr.loc["density", "residual sugar"]
    print(
        "Correlation note - density vs alcohol: "
        f"{density_alcohol_corr:.3f}; density vs residual sugar: {density_sugar_corr:.3f}"
    )
    print("\nQuality counts:")
    print(quality_counts)
    print("\nThree-class counts:")
    print(class_counts.rename(index={index: label for index, label in enumerate(CLASS_LABELS)}))


def quality_type_strata(df: pd.DataFrame) -> np.ndarray:
    """Return a granular stratum for quality score and wine type."""
    quality = df["quality"].to_numpy(dtype=int)
    is_red = df["is_red"].to_numpy(dtype=int)
    return quality * 2 + is_red


def stratified_train_test_indices(strata: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified train/test indices using only NumPy."""
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []

    for label in np.unique(strata):
        label_indices = np.flatnonzero(strata == label)
        rng.shuffle(label_indices)
        n_test = int(round(len(label_indices) * TEST_SIZE))
        test_indices.extend(label_indices[:n_test].tolist())

    test_indices_array = np.array(test_indices, dtype=int)
    test_mask = np.zeros(len(strata), dtype=bool)
    test_mask[test_indices_array] = True
    train_indices = np.flatnonzero(~test_mask)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices_array)
    return train_indices, test_indices_array


def standardize_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit mean/std on train only, then transform train and test."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def preprocess_data(df: pd.DataFrame, seed: int = RANDOM_STATE, verbose: bool = True) -> dict[str, Any]:
    """Assert cleanliness, create targets, split, and standardize features."""
    if verbose:
        print("\n" + "=" * 80)
        print("2. Preprocessing")
        print("=" * 80)

    assert df.isna().sum().sum() == 0, "Unexpected missing values found in the dataset."
    if verbose:
        print("Missing-value assertion passed: no missing values found.")

    duplicate_rows = df.duplicated()
    duplicate_count = int(duplicate_rows.sum())
    duplicate_group_count = int(df[duplicate_rows].drop_duplicates().shape[0])
    if verbose:
        print(
            "Duplicate-row check before split: "
            f"{duplicate_count} repeated rows across {duplicate_group_count} duplicated row patterns."
        )

    feature_names = [column for column in df.columns if column != "quality"]
    X = df[feature_names].to_numpy(dtype=float)
    y_reg = df["quality"].to_numpy(dtype=float)
    y_cls = quality_to_class(df["quality"].to_numpy(dtype=float))
    strata = quality_type_strata(df)

    train_idx, test_idx = stratified_train_test_indices(strata, seed)
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    X_train, X_test, _, _ = standardize_from_train(X_train_raw, X_test_raw)

    if verbose:
        print("Train/test split was stratified by original quality score x wine type.")
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape:  {X_test.shape}")
        print("Manual standardization was fit only on the train split.")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg[train_idx],
        "y_reg_test": y_reg[test_idx],
        "y_cls_train": y_cls[train_idx],
        "y_cls_test": y_cls[test_idx],
        "cv_strata_train": strata[train_idx],
        "seed": seed,
        "duplicate_count": duplicate_count,
        "duplicate_group_count": duplicate_group_count,
        "feature_names": feature_names,
    }

