"""
Очистка данных, корректный train/val/test сплит по времени,
масштабирование фичей.

Ключевой момент: используем TimeSeriesSplit — никакого random shuffle,
чтобы исключить data leakage (модель не должна видеть будущее).

Схема сплита:
  Train:       2018-01-01 — 2022-06-30  (~75%)
  Validation:  2022-07-01 — 2023-03-31  (~12.5%)
  Test:        2023-04-01 — 2023-12-31  (~12.5%)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")

TRAIN_END = "2022-06-30"
VAL_END   = "2023-03-31"

NON_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ticker",
    "target",
    "bb_mid",
    "volume_sma",
]


# ---------------------------------------------------------------------------
# Очистка
# ---------------------------------------------------------------------------

def remove_outliers(
    df: pd.DataFrame,
    cols: list[str],
    z_thresh: float = 5.0,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """
    Удаляет строки с экстремальными выбросами (|z-score| > z_thresh).

    ВАЖНО: статистики (mean, std) вычисляются ТОЛЬКО по train-периоду,
    чтобы исключить data leakage из будущего в val/test.
    Порог 5σ выбран намеренно — финансовые экстремумы (кризис, гэп)
    могут быть реальными событиями, поэтому не режем агрессивно.

    Parameters
    ----------
    df        : датафрейм с DatetimeIndex
    cols      : числовые колонки для проверки
    z_thresh  : порог по z-score
    train_end : граница train — статистики считаются только до неё
    """
    rows_before = len(df)

    # Статистики только по train-части — нет утечки из будущего
    train_mask = pd.DatetimeIndex(df.index) <= train_end
    train_part = df.loc[train_mask]

    mask_total = pd.Series(True, index=df.index)

    for col in cols:
        if col not in df.columns:
            continue

        std = train_part[col].std()
        if std == 0 or pd.isna(std):
            continue

        mean = train_part[col].mean()
        z = (df[col] - mean) / std
        mask_total &= z.abs() <= z_thresh

    df = df[mask_total]
    logger.info("Выбросы удалены: %d строк → %d строк", rows_before, len(df))
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """Логирует информацию о качестве данных."""
    na_pct    = df.isna().mean().sort_values(ascending=False)
    high_na   = na_pct[na_pct > 0.01]

    if not high_na.empty:
        logger.warning("Колонки с >1%% NaN:\n%s", high_na.to_string())
    else:
        logger.info("NaN: не обнаружено")

    dupes = df.duplicated(subset=["ticker"]).sum()
    logger.info("Дубликатов по (date, ticker): %d", dupes)
    logger.info("Строк: %d | Колонок: %d", *df.shape)
    logger.info(
        "Баланс target: %.1f%% / %.1f%%",
        (df["target"] == 1).mean() * 100,
        (df["target"] == 0).mean() * 100,
    )


# ---------------------------------------------------------------------------
# Сплит
# ---------------------------------------------------------------------------

def _fmt_index(idx: pd.DatetimeIndex) -> tuple[str, str]:
    """Возвращает (min_date, max_date) для непустого индекса."""
    if idx.empty:
        return "—", "—"
    return str(idx.min().date()), str(idx.max().date())


def time_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str   = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датафрейм на train / val / test по дате.

    Почему не random split:
    - Финансовые данные имеют временную зависимость.
    - Random split допустил бы утечку: модель видела бы данные из будущего
      при предсказании прошлого.
    - Строгий временной порядок: train < val < test.

    Parameters
    ----------
    df        : датафрейм с DatetimeIndex
    train_end : последняя дата train (включительно)
    val_end   : последняя дата val  (включительно)

    Returns
    -------
    (train, val, test) — три непересекающихся датафрейма
    """
    idx = pd.DatetimeIndex(df.index)

    train = df[idx <= train_end]
    val   = df[(idx > train_end) & (idx <= val_end)]
    test  = df[idx > val_end]

    # Проверка: ни одна выборка не должна быть пустой
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        if part.empty:
            raise ValueError(
                f"Выборка '{name}' пустая после сплита. "
                f"Проверьте границы дат и диапазон данных.\n"
                f"  train_end={train_end}, val_end={val_end}\n"
                f"  Данные: {idx.min().date()} — {idx.max().date()}"
            )

    t_min, t_max = _fmt_index(idx[idx <= train_end])
    v_min, v_max = _fmt_index(idx[(idx > train_end) & (idx <= val_end)])
    s_min, s_max = _fmt_index(idx[idx > val_end])

    logger.info(
        "Сплит:\n"
        "  Train: %s — %s (%d строк)\n"
        "  Val:   %s — %s (%d строк)\n"
        "  Test:  %s — %s (%d строк)",
        t_min, t_max, len(train),
        v_min, v_max, len(val),
        s_min, s_max, len(test),
    )

    return train, val, test


# ---------------------------------------------------------------------------
# Масштабирование
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    scaler_path: Path = PROCESSED_DIR / "scaler.joblib",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Масштабирует фичи с помощью RobustScaler.

    Почему RobustScaler:
    - Финансовые данные содержат выбросы (кризисы, гэпы).
    - RobustScaler использует медиану и IQR — устойчив к выбросам
      в отличие от StandardScaler (mean/std).

    fit() — ТОЛЬКО на X_train. transform() — на val и test.
    Это исключает утечку статистик из будущего.
    """
    # Дополнительная защита: падаем явно с понятным сообщением
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if X.empty:
            raise ValueError(
                f"{name} пустой (shape={X.shape}). "
                "Проверьте границы сплита и фильтрацию выбросов."
            )

    scaler = RobustScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler сохранён: %s", scaler_path)

    return X_train_scaled, X_val_scaled, X_test_scaled

def diagnose_inf_nan(df: pd.DataFrame, label: str) -> None:
    """Выводит детальный отчёт по inf/nan в датафрейме."""
    inf_cols = {}
    nan_cols = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        if n_inf > 0:
            inf_cols[col] = n_inf
        if n_nan > 0:
            nan_cols[col] = n_nan

    logger.info(
        "[%s] shape=%s | inf-колонок: %d | nan-колонок: %d",
        label, df.shape, len(inf_cols), len(nan_cols),
    )
    if inf_cols:
        logger.warning("[%s] INF:\n%s", label,
            pd.Series(inf_cols).sort_values(ascending=False).to_string())
    if nan_cols:
        logger.warning("[%s] NaN:\n%s", label,
            pd.Series(nan_cols).sort_values(ascending=False).to_string())

def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет +inf / -inf на NaN.

    Почему не просто дропаем строки:
    - inf возникает в конкретных колонках (деление на 0, log(0))
    - остальные фичи в этой строке могут быть валидными
    - заменяем на NaN, затем заполняем через forward fill (логика рынка:
      последнее известное значение — лучшая оценка при отсутствии данных)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[numeric_cols]).sum().sum()

    if n_inf > 0:
        logger.warning("Найдено %d inf-значений — заменяем на NaN", n_inf)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def fill_nan_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Заполняет NaN в фичах.

    Стратегия (в порядке применения):
    1. ffill  — последнее известное значение (основная стратегия)
    2. bfill  — для NaN в самом начале ряда (нет предыдущих значений)
    3. median — если всё равно остались NaN (изолированные колонки)

    ffill группируется по тикеру, если колонка ticker присутствует,
    иначе глобально.
    """
    has_ticker = "ticker" in df.columns
    numeric_feature_cols = (
        df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    if has_ticker:
        df[numeric_feature_cols] = (
            df.groupby("ticker")[numeric_feature_cols]
            .transform(lambda x: x.ffill().bfill())
        )
    else:
        df[numeric_feature_cols] = (
            df[numeric_feature_cols].ffill().bfill()
        )

    # Последний рубеж: медиана по колонке
    remaining_nan = df[numeric_feature_cols].isna().sum()
    still_nan_cols = remaining_nan[remaining_nan > 0].index.tolist()

    if still_nan_cols:
        logger.warning(
            "После ffill/bfill остались NaN в %d колонках — заполняем медианой",
            len(still_nan_cols),
        )
        for col in still_nan_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("  %s: медиана=%.4f", col, median_val)

    return df


def remove_outliers(
    df: pd.DataFrame,
    cols: list[str],
    z_thresh: float = 5.0,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """
    Удаляет строки с экстремальными выбросами (|z-score| > z_thresh).

    Статистики (mean, std) считаются ТОЛЬКО по train-периоду —
    исключает data leakage из будущего.
    Порог 5σ: финансовые экстремумы могут быть реальными событиями.
    """
    rows_before = len(df)
    train_mask  = pd.DatetimeIndex(df.index) <= train_end
    train_part  = df.loc[train_mask]
    mask_total  = pd.Series(True, index=df.index)

    for col in cols:
        if col not in df.columns:
            continue

        col_data = train_part[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col_data.empty:
            continue

        std = col_data.std()
        if std == 0 or pd.isna(std):
            continue

        mean = col_data.mean()
        z    = (df[col] - mean) / std

        # inf в самом df тоже помечаем как выброс
        mask_total &= z.abs().replace([np.inf, -np.inf], z_thresh + 1) <= z_thresh

    df = df[mask_total]
    logger.info("Выбросы удалены: %d строк → %d строк", rows_before, len(df))
    return df

# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def prepare_dataset(
    features_path: Path = PROCESSED_DIR / "features.csv",
    output_dir:    Path = PROCESSED_DIR,
    z_thresh:      float = 5.0,
) -> dict[str, pd.DataFrame]:
    """
    Полный пайплайн подготовки данных:
      1.  Загрузка фичей
      2.  Проверка качества
      3.  Определение списка фичей
      4.  Замена inf → NaN
      5.  Заполнение NaN (ffill → bfill → median)
      6.  Удаление выбросов (статистики только по train)
      7.  Финальная проверка — не должно остаться inf/nan
      8.  Time-based сплит
      9.  Масштабирование (fit только на train)
      10. Сохранение сплитов
    """
    # 1. Загрузка
    logger.info("Загрузка: %s", features_path)
    df = pd.read_csv(features_path, index_col="date", parse_dates=True)
    df = df.sort_index()

    # 2. Проверка качества
    check_data_quality(df)

    # 3. Список фичей
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info("Фичей для обучения: %d", len(feature_cols))
    logger.info("Список фичей: %s", feature_cols)

    # 4. inf → NaN
    df = replace_inf_with_nan(df)

    # 5. Заполнение NaN
    df = fill_nan_features(df, feature_cols)

    # 6. Удаление выбросов
    numeric_features = (
        df[feature_cols]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    df = remove_outliers(df, numeric_features, z_thresh=z_thresh, train_end=TRAIN_END)

    # 7. Финальная проверка перед скалированием
    diagnose_inf_nan(df[feature_cols], "after_cleaning")

    still_inf = np.isinf(df[numeric_features]).sum().sum()
    still_nan = df[numeric_features].isna().sum().sum()

    if still_inf > 0 or still_nan > 0:
        raise ValueError(
            f"После очистки остались невалидные значения: "
            f"inf={still_inf}, nan={still_nan}. "
            f"Проверьте логику генерации фичей."
        )

    # 8. Сплит
    train_df, val_df, test_df = time_split(df)

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val,   y_val   = val_df[feature_cols],   val_df["target"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target"]

    # 9. Масштабирование
    X_train_sc, X_val_sc, X_test_sc = scale_features(X_train, X_val, X_test)

    # 10. Сохранение
    splits = {
        "X_train": X_train_sc,
        "X_val":   X_val_sc,
        "X_test":  X_test_sc,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = output_dir / f"{name}.csv"
        data.to_csv(path)
        logger.info("Сохранено: %s (%d строк)", path, len(data))

    return splits


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    splits = prepare_dataset()

    print("\n=== Итог ===")
    for name, data in splits.items():
        print(f"{name:10s}: {data.shape}")
