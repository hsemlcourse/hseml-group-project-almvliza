import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RANDOM_STATE = 42

# Загрузка данных

def load_splits(processed_dir: Path = Path("data/processed")) -> dict[str, pd.DataFrame]:
    """Загружает готовые сплиты из data/processed/."""
    splits = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = processed_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Файл {path} не найден. "
                "Сначала запустите: python src/preprocessing.py"
            )
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        splits[name] = df["target"].astype(int) if name.startswith("y_") else df

    logger.info(
        "Данные загружены: train=%d | val=%d | test=%d | фичей=%d",
        len(splits["X_train"]),
        len(splits["X_val"]),
        len(splits["X_test"]),
        splits["X_train"].shape[1],
    )
    return splits

# Метрики

def evaluate(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "val",
) -> dict:
    """Считает метрики для одной выборки."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    metrics = {
        "split":     split_name,
        "roc_auc":   round(roc_auc_score(y, y_proba), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y, y_pred), 4),
    }

    logger.info(
        "[%s] ROC-AUC=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f | Acc=%.4f",
        split_name,
        metrics["roc_auc"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["accuracy"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Обучение baseline
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Logistic Regression с дефолтными параметрами
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info("Baseline обучен: %s", model)
    return model


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------

def save_model(model: object, model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Сохраняет модель в models/{model_name}.joblib"""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    logger.info("Модель сохранена: %s", path)
    return path


def load_model(model_name: str, models_dir: Path = MODELS_DIR) -> object:
    """Загружает модель из models/{model_name}.joblib"""
    path = models_dir / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    model = joblib.load(path)
    logger.info("Модель загружена: %s", path)
    return model


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Загрузка данных
    splits = load_splits()
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    # Обучение
    model = train_baseline(X_train, y_train)

    # Метрики
    val_metrics  = evaluate(model, X_val,  y_val,  split_name="val")
    test_metrics = evaluate(model, X_test, y_test, split_name="test")

    # Сохранение
    save_model(model, "baseline_logistic_regression")

    # Итог
    print("\nBaseline results: ===")
    for metrics in [val_metrics, test_metrics]:
        print(f"\n[{metrics['split'].upper()}]")
        for k, v in metrics.items():
            if k != "split":
                print(f"  {k:12s}: {v}")
