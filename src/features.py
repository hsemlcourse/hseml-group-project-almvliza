"""
Расчёт технических индикаторов и таргета

Индикаторы:
- Трендовые: SMA, EMA
- Моментум: RSI, MACD, ROC
- Волатильность: Bollinger Bands, ATR
- Объём: OBV, Volume SMA ratio
- Свечные паттерны: Body size, Upper/Lower shadow
- Таргет: рост цены через 5 дней > 1%
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FORWARD_DAYS = 5
PRICE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Трендовые индикаторы
# ---------------------------------------------------------------------------

def add_sma(df: pd.DataFrame, windows: list[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Simple Moving Average.
    Добавляет sma_{window} и sma_{window}_ratio (close / sma).
    Ratio > 1 означает, что цена выше скользящей средней.
    """
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"sma_{w}_ratio"] = df["close"] / df[f"sma_{w}"]
    return df


def add_ema(df: pd.DataFrame, windows: list[int] = [12, 26]) -> pd.DataFrame:
    """
    Exponential Moving Average.
    Добавляет ema_{window} (напрямую не используется, нужно ддля MACD)
    """
    for w in windows:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# Моментум
# ---------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) — отношение среднего роста к среднему падению за 14 дней, в нормировке от 0 до 100:
      > 70 — перекупленность (возможный разворот вниз)
      < 30 — перепроданность (возможный разворот вверх)
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow) -- вспомогательная фича
    Signal line = EMA(MACD, signal) -- вспомогательная фича
    Histogram = MACD - Signal -- опорная фича
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_roc(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Rate of Change — процентное изменение цены за N дней.
    ROC = (close - close[N дней назад]) / close[N дней назад] * 100
    ROC и RSI частично дублируют друг друга, но RSI нелинейный и нормированный, ROC — линейный и не ограничен
    """
    for w in windows:
        df[f"roc_{w}"] = df["close"].pct_change(w) * 100
    return df


# ---------------------------------------------------------------------------
# Волатильность
# ---------------------------------------------------------------------------

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Добавляет:
    - bb_upper, bb_lower, bb_mid — полосы
    - bb_width — ширина канала (мера волатильности)
    - bb_pct — позиция цены внутри канала (0 = нижняя граница, 1 = верхняя)
    """
    rolling = df["close"].rolling(window)
    df["bb_mid"] = rolling.mean()
    std = rolling.std()

    df["bb_upper"] = df["bb_mid"] + num_std * std
    df["bb_lower"] = df["bb_mid"] - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    ).replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average True Range — средний истинный диапазон.
    Мера волатильности, не зависящая от направления движения.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(span=window, adjust=False).mean()
    df["atr_ratio"] = df["atr"] / df["close"]  # нормированный ATR
    return df


def add_historical_volatility(
    df: pd.DataFrame,
    windows: list[int] = [10, 20],
) -> pd.DataFrame:
    """
    Историческая волатильность — стандартное отклонение дневных логдоходностей
    за скользящее окно, умноженное на sqrt(252) для перевода в годовую.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for w in windows:
        df[f"hvol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
    return df


# ---------------------------------------------------------------------------
# Объёмные индикаторы
# ---------------------------------------------------------------------------

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    On-Balance Volume.
    Аккумулирует объём: прибавляет если цена выросла, вычитает если упала.
    Дивергенция OBV и цены — сигнал разворота.
    """
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (df["volume"] * direction).cumsum()
    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Объёмные фичи:
    - volume_sma_ratio: отношение текущего объёма к среднему (аномальный объём > 1.5)
    - volume_change: процентное изменение объёма
    """
    df["volume_sma"] = df["volume"].rolling(window).mean()
    df["volume_sma_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    df["volume_change"] = df["volume"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Свечные паттерны
# ---------------------------------------------------------------------------

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки на основе свечей:
    - body_size: размер тела свечи относительно ATR
    - upper_shadow: верхняя тень
    - lower_shadow: нижняя тень
    - is_bullish: 1 если цена закрытия выше открытия
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["body_size"] = (df["close"] - df["open"]).abs() / candle_range
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / candle_range
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / candle_range
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Таргет
# ---------------------------------------------------------------------------

def add_target(
    df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    threshold: float = PRICE_THRESHOLD,
) -> pd.DataFrame:
    """
    Целевая переменная: вырастет ли цена через forward_days торговых дней
    более чем на threshold.

    target = 1 если close[t + forward_days] > close[t] * (1 + threshold)
    target = 0 иначе

    shift(-forward_days) делается внутри группы по тикеру,
    чтобы не допустить смешивания данных разных акций.
    """
    future_close = df["close"].shift(-forward_days)
    df["target"] = (future_close > df["close"] * (1 + threshold)).astype(int)

    # Последние forward_days строк каждого тикера не имеют таргета — дропаем позже
    df["target"] = df["target"].where(
        df.groupby("ticker").cumcount(ascending=False) >= forward_days,
        other=np.nan,
    )
    return df


# ---------------------------------------------------------------------------
# Временные фичи
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Календарные признаки:
    - day_of_week: 0=пн ... 4=пт
    - month: 1-12
    - quarter: 1-4
    - is_month_end: флаг последнего торгового дня месяца
    """
    idx = pd.DatetimeIndex(df.index)
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_month_end"] = idx.is_month_end.astype(int)
    return df


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидает колонки: open, high, low, close, volume, ticker.
    Индекс по DatetimeIndex (date).

    Returns
    -------
    pd.DataFrame
        Датафрейм с исходными колонками + все фичи + target.
        Строки с NaN (из-за rolling-окон или таргета) удалены.
    """
    logger.info("Строим фичи для %d тикеров...", df["ticker"].nunique())

    result_frames = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_index().copy()

        # Трендовые
        group = add_sma(group)
        group = add_ema(group)

        # Моментум
        group = add_rsi(group)
        group = add_macd(group)
        group = add_roc(group)

        # Волатильность
        group = add_bollinger_bands(group)
        group = add_atr(group)
        group = add_historical_volatility(group)

        # Объём
        group = add_obv(group)
        group = add_volume_features(group)

        # Свечи
        group = add_candle_features(group)

        # Таргет
        group = add_target(group)

        # Время
        group = add_time_features(group)

        result_frames.append(group)

    result = pd.concat(result_frames).sort_values(["ticker", "date"])

    rows_before = len(result)
    result = result.dropna()
    logger.info(
        "Удалено строк с NaN: %d (rolling-окна + таргет последних %d дней)",
        rows_before - len(result),
        FORWARD_DAYS,
    )
    logger.info("Итоговый датасет: %d строк, %d колонок", *result.shape)

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    raw_path = Path("data/raw/all_tickers.csv")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Файл {raw_path} не найден. Сначала запустите src/parser.py"
        )

    df_raw = pd.read_csv(raw_path, index_col="date", parse_dates=True)
    df_features = build_features(df_raw)

    out_path = Path("data/processed/features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_path)

    print("\n=== Превью фичей ===")
    print(df_features.head(3).T)
    print(f"\nКолонок: {len(df_features.columns)}")
    print(f"Из них фичей: {len(df_features.columns) - 6}")  # -6 OHLCV + ticker
    print(f"Баланс классов:\n{df_features['target'].value_counts(normalize=True).round(3)}")
