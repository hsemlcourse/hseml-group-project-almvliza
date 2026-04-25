"""
Парсинг исторических OHLCV-данных с Yahoo Finance через yfinance.
Сохраняет даннные по каждому тикеру в data/raw/.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")

START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

#тикеры из разных секторов S&P 500
TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ORCL", "CSCO",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "AMAT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
    "USB", "PNC", "TFC", "SCHW", "COF", "MCO", "ICE",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "CVS", "CI",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "BKNG", "CMG", "MAR", "GM", "F", "ROST", "YUM",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL",
    "MDLZ", "KHC", "GIS", "K", "SYY", "KR", "TSN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    "PXD", "OXY", "BKR", "HAL", "DVN", "FANG", "HES",
    # Industrials
    "CAT", "HON", "UPS", "BA", "LMT", "RTX", "DE", "GE",
    "MMM", "FDX", "NSC", "UNP", "EMR", "ETN", "PH",
]

def fetch_ticker(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame | None:
    """
    Скачивает дневные OHLCV-данные для одного тикера.

    Parameters
    ----------
    ticker : str
        Тикер акции (например, "AAPL").
    start : str
        Начало периода в формате YYYY-MM-DD.
    end : str
        Конец периода в формате YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame | None
        DataFrame с колонками [Open, High, Low, Close, Volume] и DatetimeIndex,
        либо None если данные не удалось получить.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            logger.warning("Нет данных для тикера %s", ticker)
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df["ticker"] = ticker

        logger.info("✓ %s | строк: %d", ticker, len(df))
        return df

    except Exception as exc:
        logger.error("Ошибка при загрузке %s: %s", ticker, exc)
        return None


def save_ticker(df: pd.DataFrame, ticker: str, output_dir: Path) -> None:
    """Сохраняет DataFrame в CSV в указанную директорию."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}.csv"
    df.to_csv(path)
    logger.info("Сохранено: %s", path)


def parse_all(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = RAW_DATA_DIR,
    delay: float = 0.3,
) -> pd.DataFrame:
    """
    Скачивает данные по всем тикерам, сохраняет индивидуальные CSV
    и возвращает единый DataFrame со всеми акциями.

    Parameters
    ----------
    tickers : list[str]
        Список тикеров для загрузки.
    start : str
        Начало периода.
    end : str
        Конец периода.
    output_dir : Path
        Папка для сохранения сырых данных.
    delay : float
        Пауза между запросами (секунды)

    Returns
    -------
    pd.DataFrame
        Объединённый DataFrame по всем тикерам.
    """
    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Загружаем %s...", i, len(tickers), ticker)

        df = fetch_ticker(ticker, start=start, end=end)
        if df is not None:
            save_ticker(df, ticker, output_dir)
            all_frames.append(df)

        time.sleep(delay)

    if not all_frames:
        raise RuntimeError("Не удалось загрузить данные ни по одному тикеру.")

    combined = pd.concat(all_frames)
    combined = combined.sort_values(["ticker", "date"])

    combined_path = output_dir / "all_tickers.csv"
    combined.to_csv(combined_path)
    logger.info(
        "Итого строк: %d | тикеров: %d | сохранено в %s",
        len(combined),
        combined["ticker"].nunique(),
        combined_path,
    )

    return combined

if __name__ == "__main__":
    df = parse_all()
    print("\nПревью данных")
    print(df.head(10).to_string())
    print(f"\nФорма датасета: {df.shape}")
    print(f"Тикеров: {df['ticker'].nunique()}")
    print(f"Период: {df.index.min()} — {df.index.max()}")
