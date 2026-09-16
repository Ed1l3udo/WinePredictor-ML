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

def add_intercept(X: np.ndarray) -> np.ndarray:
    """Append a bias column to a feature matrix."""
    return np.column_stack([np.ones(X.shape[0]), X])


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Return inverse-frequency weights for a classification target."""
    weights = np.ones_like(y, dtype=float)
    n_samples = len(y)
    for label in np.unique(y):
        count = np.sum(y == label)
        weights[y == label] = n_samples / (len(np.unique(y)) * count)
    return weights


class SoftmaxRegressionScratch:
    """Multiclass logistic regression trained by gradient descent."""

    def __init__(
        self,
        l2_strength: float,
        learning_rate: float = 0.08,
        max_iter: int = 4000,
        tolerance: float = 1e-8,
    ) -> None:
        self.l2_strength = l2_strength
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegressionScratch":
        X_bias = add_intercept(X)
        weights = np.zeros((X_bias.shape[1], N_CLASSES), dtype=float)
        y_one_hot = np.eye(N_CLASSES)[y.astype(int)]
        sample_weights = balanced_sample_weights(y)
        total_weight = sample_weights.sum()

        for _ in range(self.max_iter):
            probabilities = softmax(X_bias @ weights)
            errors = (probabilities - y_one_hot) * sample_weights[:, None]
            gradient = (X_bias.T @ errors) / total_weight
            regularization = self.l2_strength * weights
            regularization[0, :] = 0.0
            update = self.learning_rate * (gradient + regularization)
            weights -= update
            if np.linalg.norm(update) < self.tolerance:
                break

        self.weights = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SoftmaxRegressionScratch must be fit before predict_proba.")
        return softmax(add_intercept(X) @ self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def coefficients(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SoftmaxRegressionScratch must be fit before coefficients.")
        return self.weights[1:]


class KNNScratch:
    """K-nearest neighbors for regression or multiclass classification."""

    def __init__(self, k: int, batch_size: int = 256) -> None:
        self.k = k
        self.batch_size = batch_size
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        self.X_train = X
        self.y_train = y
        return self

    def _nearest_values(self, X: np.ndarray) -> list[np.ndarray]:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNNScratch must be fit before prediction.")

        train_squared = np.sum(self.X_train**2, axis=1)
        nearest_batches: list[np.ndarray] = []

        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]
            distances = (
                np.sum(batch**2, axis=1)[:, None]
                + train_squared[None, :]
                - 2.0 * batch @ self.X_train.T
            )
            distances = np.maximum(distances, 0.0)
            nearest_idx = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]
            nearest_batches.append(self.y_train[nearest_idx])

        return nearest_batches

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        predictions = [values.mean(axis=1) for values in self._nearest_values(X)]
        return np.concatenate(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability_batches = []
        for values in self._nearest_values(X):
            batch_probabilities = np.zeros((values.shape[0], N_CLASSES), dtype=float)
            for row_index, neighbor_labels in enumerate(values.astype(int)):
                counts = np.bincount(neighbor_labels, minlength=N_CLASSES)
                batch_probabilities[row_index] = counts / self.k
            probability_batches.append(batch_probabilities)
        return np.vstack(probability_batches)

    def predict_classification(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

@dataclass
class RegressionTreeNode:
    """One node in a manually built regression tree."""

    value: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "RegressionTreeNode | None" = None
    right: "RegressionTreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeRegressorScratch:
    """CART-style regression tree used as the base learner for Random Forest."""

    def __init__(
        self,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str | int,
        n_split_candidates: int,
        random_state: int,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_split_candidates = n_split_candidates
        self.rng = np.random.default_rng(random_state)
        self.root: RegressionTreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._build_tree(X, y, depth=0)
        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTreeRegressorScratch must be fit before predict.")
        return np.array([self._predict_one(row, self.root) for row in X], dtype=float)

    def _predict_one(self, row: np.ndarray, node: RegressionTreeNode) -> float:
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if row[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.value

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> RegressionTreeNode:
        node_value = float(np.mean(y))
        node = RegressionTreeNode(value=node_value)

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, impurity_decrease = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node

        if self.feature_importances_ is not None:
            self.feature_importances_[feature_index] += impurity_decrease

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return float(np.var(y)) <= 1e-12

    def _feature_subset(self, n_features: int) -> np.ndarray:
        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "all":
            size = n_features
        elif isinstance(self.max_features, int):
            size = min(n_features, self.max_features)
        else:
            size = n_features
        return self.rng.choice(n_features, size=size, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None

        parent_sse = self._sse(y)
        best_feature: int | None = None
        best_threshold: float | None = None
        best_sse = parent_sse

        for feature_index in self._feature_subset(n_features):
            column = X[:, feature_index]
            order = np.argsort(column)
            x_sorted = column[order]
            y_sorted = y[order]

            if x_sorted[0] == x_sorted[-1]:
                continue

            candidate_count = min(self.n_split_candidates, n_samples - 2 * self.min_samples_leaf)
            split_positions = np.linspace(
                self.min_samples_leaf,
                n_samples - self.min_samples_leaf,
                num=candidate_count,
                dtype=int,
            )
            split_positions = np.unique(split_positions)

            cumulative_y = np.cumsum(y_sorted)
            cumulative_y2 = np.cumsum(y_sorted**2)
            total_y = cumulative_y[-1]
            total_y2 = cumulative_y2[-1]

            for position in split_positions:
                if x_sorted[position - 1] == x_sorted[position]:
                    continue

                left_count = position
                right_count = n_samples - position
                left_sum = cumulative_y[position - 1]
                left_sum2 = cumulative_y2[position - 1]
                right_sum = total_y - left_sum
                right_sum2 = total_y2 - left_sum2

                left_sse = left_sum2 - (left_sum * left_sum / left_count)
                right_sse = right_sum2 - (right_sum * right_sum / right_count)
                total_sse = left_sse + right_sse

                if total_sse < best_sse:
                    best_sse = total_sse
                    best_feature = int(feature_index)
                    best_threshold = float((x_sorted[position - 1] + x_sorted[position]) / 2.0)

        if best_feature is None or best_threshold is None:
            return None

        impurity_decrease = float(parent_sse - best_sse)
        if impurity_decrease <= 1e-12:
            return None
        return best_feature, best_threshold, impurity_decrease

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        return float(np.sum((y - np.mean(y)) ** 2))


class RandomForestRegressorScratch:
    """Random Forest regressor built from scratch with bootstrap aggregation."""

    def __init__(
        self,
        n_estimators: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str | int = "sqrt",
        n_split_candidates: int = 32,
        bootstrap_fraction: float = 1.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_split_candidates = n_split_candidates
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressorScratch] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorScratch":
        rng = np.random.default_rng(self.random_state)
        n_samples = len(y)
        bootstrap_size = max(1, int(round(n_samples * self.bootstrap_fraction)))
        self.trees = []
        importances = np.zeros(X.shape[1], dtype=float)

        for _ in range(self.n_estimators):
            sample_idx = rng.integers(0, n_samples, size=bootstrap_size)
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_split_candidates=self.n_split_candidates,
                random_state=tree_seed,
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total_importance = importances.sum()
        self.feature_importances_ = importances / total_importance if total_importance > 0 else importances
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("RandomForestRegressorScratch must be fit before predict.")
        predictions = np.vstack([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)


def kfold_indices(n_samples: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build shuffled K-fold train/validation indices."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    output: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_number in range(n_folds):
        validation_idx = folds[fold_number]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_number])
        output.append((train_idx, validation_idx))
    return output


def stratified_kfold_indices(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified K-fold train/validation indices for class labels."""
    rng = np.random.default_rng(seed)
    fold_lists: list[list[int]] = [[] for _ in range(n_folds)]

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        for fold_number, split in enumerate(np.array_split(label_indices, n_folds)):
            fold_lists[fold_number].extend(split.tolist())

    output: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(y))
    for fold in fold_lists:
        validation_idx = np.array(fold, dtype=int)
        validation_mask = np.zeros(len(y), dtype=bool)
        validation_mask[validation_idx] = True
        train_idx = all_indices[~validation_mask]
        rng.shuffle(train_idx)
        rng.shuffle(validation_idx)
        output.append((train_idx, validation_idx))
    return output


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum == 0:
        return float("nan")
    return float(1.0 - residual_sum / total_sum)


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for true_value, predicted_value in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        matrix[true_value, predicted_value] += 1
    return matrix


def auc_roc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary ROC-AUC with average ranks for tied scores."""
    y_true = y_true.astype(int)
    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[y_true == 1].sum()
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    return float(auc)


def multiclass_auc_roc_ovr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute macro one-vs-rest AUC-ROC for multiclass predictions."""
    auc_values = []
    for label in range(N_CLASSES):
        binary_true = (y_true.astype(int) == label).astype(int)
        auc_values.append(auc_roc_manual(binary_true, y_score[:, label]))
    return float(np.nanmean(auc_values))


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Compute multiclass classification metrics from scratch."""
    matrix = confusion_matrix_manual(y_true, y_pred)
    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []

    for label in range(N_CLASSES):
        true_positive = matrix[label, label]
        predicted_positive = matrix[:, label].sum()
        actual_positive = matrix[label, :].sum()
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_precision.append(float(precision))
        per_class_recall.append(float(recall))
        per_class_f1.append(float(f1))

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_macro": float(np.mean(per_class_f1)),
        "auc_roc_macro": multiclass_auc_roc_ovr(y_true, y_score),
        "precision_ruim": per_class_precision[0],
        "recall_ruim": per_class_recall[0],
        "precision_mediano": per_class_precision[1],
        "recall_mediano": per_class_recall[1],
        "precision_bom": per_class_precision[2],
        "recall_bom": per_class_recall[2],
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Save a confusion matrix for a classifier or binned regressor."""
    matrix = confusion_matrix_manual(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(f"Matriz de confusao - {model_name}")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    save_current_plot(f"confusion_matrix_{safe_name(model_name)}.png")


def tune_random_forest(X: np.ndarray, y: np.ndarray, cv_strata: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select a small Random Forest hyperparameter grid by CV RMSE."""
    folds = stratified_kfold_indices(cv_strata, RF_TUNING_FOLDS, RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    for config in RF_SEARCH_SPACE:
        fold_scores = []
        for fold_number, (train_idx, validation_idx) in enumerate(folds):
            X_fold_train, X_fold_validation, _, _ = standardize_from_train(X[train_idx], X[validation_idx])
            model = RandomForestRegressorScratch(
                n_estimators=RF_TUNING_N_ESTIMATORS,
                max_depth=int(config["max_depth"]),
                min_samples_split=int(config["min_samples_split"]),
                min_samples_leaf=int(config["min_samples_leaf"]),
                max_features=str(config["max_features"]),
                n_split_candidates=RF_N_SPLIT_CANDIDATES,
                random_state=RANDOM_STATE + fold_number,
            ).fit(X_fold_train, y[train_idx])
            predictions = model.predict(X_fold_validation)
            fold_scores.append(rmse(y[validation_idx], predictions))

        rows.append(
            {
                "max_depth": int(config["max_depth"]),
                "min_samples_split": int(config["min_samples_split"]),
                "min_samples_leaf": int(config["min_samples_leaf"]),
                "max_features": str(config["max_features"]),
                "cv_rmse": float(np.mean(fold_scores)),
            }
        )

    cv_results = pd.DataFrame(rows)
    best_row = cv_results.sort_values("cv_rmse").iloc[0]
    best_config = {
        "max_depth": int(best_row["max_depth"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_features": str(best_row["max_features"]),
    }
    return best_config, cv_results


def tune_logistic_l2(X: np.ndarray, y: np.ndarray, cv_strata: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select softmax L2 strength by stratified CV using macro AUC-ROC."""
    folds = stratified_kfold_indices(cv_strata, CV_FOLDS, RANDOM_STATE)
    rows: list[dict[str, float]] = []

    for strength in LOGISTIC_L2_STRENGTHS:
        fold_scores = []
        for train_idx, validation_idx in folds:
            X_fold_train, X_fold_validation, _, _ = standardize_from_train(X[train_idx], X[validation_idx])
            model = SoftmaxRegressionScratch(l2_strength=float(strength)).fit(X_fold_train, y[train_idx])
            probabilities = model.predict_proba(X_fold_validation)
            fold_scores.append(multiclass_auc_roc_ovr(y[validation_idx], probabilities))
        rows.append({"l2_strength": float(strength), "cv_auc_roc_macro": float(np.mean(fold_scores))})

    cv_results = pd.DataFrame(rows)
    best_strength = float(cv_results.sort_values("cv_auc_roc_macro", ascending=False).iloc[0]["l2_strength"])
    return best_strength, cv_results


def tune_knn_k(X: np.ndarray, y: np.ndarray, task: str, cv_strata: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Select KNN k by manual CV for regression RMSE or classification macro AUC."""
    folds = stratified_kfold_indices(cv_strata, CV_FOLDS, RANDOM_STATE)

    rows: list[dict[str, float]] = []
    for k in K_VALUES:
        fold_scores = []
        for train_idx, validation_idx in folds:
            X_fold_train, X_fold_validation, _, _ = standardize_from_train(X[train_idx], X[validation_idx])
            model = KNNScratch(k=k).fit(X_fold_train, y[train_idx])
            if task == "regression":
                predictions = model.predict_regression(X_fold_validation)
                fold_scores.append(rmse(y[validation_idx], predictions))
            else:
                probabilities = model.predict_proba(X_fold_validation)
                fold_scores.append(multiclass_auc_roc_ovr(y[validation_idx].astype(int), probabilities))

        metric_name = "cv_rmse" if task == "regression" else "cv_auc_roc_macro"
        rows.append({"k": float(k), metric_name: float(np.mean(fold_scores))})

    cv_results = pd.DataFrame(rows)
    if task == "regression":
        best_k = int(cv_results.sort_values("cv_rmse").iloc[0]["k"])
    else:
        best_k = int(cv_results.sort_values("cv_auc_roc_macro", ascending=False).iloc[0]["k"])
    return best_k, cv_results


def regression_row(
    model_name: str,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_cls: np.ndarray,
    *,
    y_train_reg: np.ndarray | None = None,
    y_train_pred_reg: np.ndarray | None = None,
    y_train_cls: np.ndarray | None = None,
    fit_seconds: float = float("nan"),
    predict_seconds: float = float("nan"),
    tuning_seconds: float = float("nan"),
    plot: bool = True,
) -> dict[str, float | str]:
    """Compute regression metrics plus three-class binned classification metrics."""
    y_pred_cls = quality_to_class(y_pred_reg)
    class_scores = quality_scores_to_class_scores(y_pred_reg)
    class_metrics = classification_scores(y_true_cls, y_pred_cls, class_scores)
    if plot:
        plot_confusion_matrix(y_true_cls, y_pred_cls, f"{model_name} class binned")

    train_rmse = float("nan")
    train_accuracy = float("nan")
    train_f1_macro = float("nan")
    train_auc_roc_macro = float("nan")
    if y_train_reg is not None and y_train_pred_reg is not None and y_train_cls is not None:
        train_rmse = rmse(y_train_reg, y_train_pred_reg)
        train_pred_cls = quality_to_class(y_train_pred_reg)
        train_scores = quality_scores_to_class_scores(y_train_pred_reg)
        train_class_metrics = classification_scores(y_train_cls, train_pred_cls, train_scores)
        train_accuracy = train_class_metrics["accuracy"]
        train_f1_macro = train_class_metrics["f1_macro"]
        train_auc_roc_macro = train_class_metrics["auc_roc_macro"]

    return {
        "model": model_name,
        "formulation": "regression",
        "prediction_type": "raw_quality_then_3_class_bins",
        "train_rmse": train_rmse,
        "train_accuracy": train_accuracy,
        "train_f1_macro": train_f1_macro,
        "train_auc_roc_macro": train_auc_roc_macro,
        "rmse": rmse(y_true_reg, y_pred_reg),
        "mae": mae(y_true_reg, y_pred_reg),
        "r2": r2_score_manual(y_true_reg, y_pred_reg),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "tuning_seconds": tuning_seconds,
        **class_metrics,
    }


def classification_row(
    model_name: str,
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_score: np.ndarray,
    *,
    y_train_cls: np.ndarray | None = None,
    y_train_pred_cls: np.ndarray | None = None,
    y_train_score: np.ndarray | None = None,
    fit_seconds: float = float("nan"),
    predict_seconds: float = float("nan"),
    tuning_seconds: float = float("nan"),
    plot: bool = True,
) -> dict[str, float | str]:
    """Compute three-class classification metrics."""
    class_metrics = classification_scores(y_true_cls, y_pred_cls, y_score)
    if plot:
        plot_confusion_matrix(y_true_cls, y_pred_cls, model_name)

    train_accuracy = float("nan")
    train_f1_macro = float("nan")
    train_auc_roc_macro = float("nan")
    if y_train_cls is not None and y_train_pred_cls is not None and y_train_score is not None:
        train_class_metrics = classification_scores(y_train_cls, y_train_pred_cls, y_train_score)
        train_accuracy = train_class_metrics["accuracy"]
        train_f1_macro = train_class_metrics["f1_macro"]
        train_auc_roc_macro = train_class_metrics["auc_roc_macro"]

    return {
        "model": model_name,
        "formulation": "classification",
        "prediction_type": "three_class",
        "train_rmse": np.nan,
        "train_accuracy": train_accuracy,
        "train_f1_macro": train_f1_macro,
        "train_auc_roc_macro": train_auc_roc_macro,
        "rmse": np.nan,
        "mae": np.nan,
        "r2": np.nan,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "tuning_seconds": tuning_seconds,
        **class_metrics,
    }

def plot_random_forest_importance(importances: np.ndarray, feature_names: list[str]) -> None:
    """Save feature importances learned by the manual Random Forest."""
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, y="feature", x="importance", color="#2f6f73")
    plt.title("Importancia de atributos - Random Forest")
    plt.xlabel("Importancia")
    plt.ylabel("Atributo")
    save_current_plot("rf_feature_importance_regressor.png")

def plot_softmax_coefficients(coefficients: np.ndarray, feature_names: list[str]) -> None:
    """Save standardized coefficients from the manual Softmax Regression."""
    rows = []
    for class_index, class_label in enumerate(CLASS_LABELS):
        for feature_name, coefficient in zip(feature_names, coefficients[:, class_index], strict=True):
            rows.append({"class": class_label, "feature": feature_name, "coefficient": coefficient})
    coef_df = pd.DataFrame(rows)
    feature_order = (
        coef_df.groupby("feature")["coefficient"]
        .apply(lambda values: values.abs().max())
        .sort_values(ascending=True)
        .index
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=coef_df, y="feature", x="coefficient", hue="class", order=feature_order, palette="Set2")
    plt.axvline(0, color="black", linewidth=0.9)
    plt.title("Coeficientes da Regressao Softmax")
    plt.xlabel("Coeficiente")
    plt.ylabel("Atributo")
    plt.legend(title="Classe")
    save_current_plot("softmax_coefficients.png")
