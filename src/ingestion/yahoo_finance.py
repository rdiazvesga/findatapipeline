"""
Módulo de ingesta de datos desde Yahoo Finance.

Responsabilidad única: descargar datos OHLCV históricos
y devolverlos en formato estandarizado. No limpia, no almacena.
"""
import logging
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceIngester:
    """
    Descarga datos OHLCV históricos desde Yahoo Finance.

    Maneja reintentos, rate limiting, y estandarización
    de columnas. No hace limpieza de calidad de datos
    (esa responsabilidad es de DataQualityChecker).

    Example:
        ingester = YahooFinanceIngester()
        df = ingester.fetch_single("SPY", "2023-01-01", "2024-01-01")
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Args:
            max_retries: Intentos máximos ante falla de red.
            retry_delay: Segundos de espera entre reintentos.
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_single(
        self,
        symbol: str,
        start_date: str,
        end_date:   str,
        interval:   str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Descarga datos OHLCV para un símbolo.

        Args:
            symbol:     Ticker (ej: 'SPY', 'EURUSD=X', 'BTC-USD').
            start_date: Fecha inicio 'YYYY-MM-DD'.
            end_date:   Fecha fin   'YYYY-MM-DD'.
            interval:   Frecuencia  '1d' | '1wk' | '1mo'.

        Returns:
            DataFrame con columnas [symbol, open, high, low, close, volume]
            e índice DatetimeIndex, o None si la descarga falla.
        """
        self._validate_dates(start_date, end_date)

        for attempt in range(1, self.max_retries + 1):
            try:
                df = self._download(symbol, start_date, end_date, interval)

                if df is None or df.empty:
                    logger.warning(
                        "Sin datos para %s en rango %s:%s",
                        symbol, start_date, end_date,
                    )
                    return None

                df = self._standardize(df, symbol)
                logger.info(
                    "Descargado %s: %d filas (%s → %s)",
                    symbol, len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                )
                return df

            except Exception as exc:
                logger.warning(
                    "Intento %d/%d fallido para %s: %s",
                    attempt, self.max_retries, symbol, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error("Descarga fallida definitivamente para %s", symbol)
        return None

    def fetch_multiple(
        self,
        symbols:    list[str],
        start_date: str,
        end_date:   str,
        interval:   str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga datos para una lista de símbolos.

        Args:
            symbols:    Lista de tickers.
            start_date: Fecha inicio 'YYYY-MM-DD'.
            end_date:   Fecha fin   'YYYY-MM-DD'.
            interval:   Frecuencia de datos.

        Returns:
            Diccionario {symbol: DataFrame} con descargas exitosas.
            Símbolos fallidos quedan excluidos (se loguean como error).
        """
        results: dict[str, pd.DataFrame] = {}
        failed:  list[str] = []

        for symbol in symbols:
            df = self.fetch_single(symbol, start_date, end_date, interval)
            if df is not None:
                results[symbol] = df
            else:
                failed.append(symbol)

        logger.info(
            "fetch_multiple: %d/%d exitosos. Fallidos: %s",
            len(results), len(symbols), failed or "ninguno",
        )
        return results

    # ── helpers privados ──────────────────────────────────────────────

    def _download(
        self,
        symbol:     str,
        start_date: str,
        end_date:   str,
        interval:   str,
    ) -> Optional[pd.DataFrame]:
        """Llama a yfinance y devuelve el DataFrame crudo."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            actions=False,       # sin dividendos ni splits en este dataset
        )
        return df if not df.empty else None

    def _standardize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Estandariza columnas, índice y tipos de datos.

        Salida garantizada:
          - índice: DatetimeIndex sin timezone, nombre 'date'
          - columnas: symbol, open, high, low, close, volume
          - tipos: float64 para OHLC, int64 para volume, str para symbol
        """
        # Normalizar nombres de columnas
        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume",
        })

        # Seleccionar solo las columnas que necesitamos
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in ohlcv_cols if c in df.columns]].copy()

        # Agregar símbolo
        df["symbol"] = symbol

        # Limpiar índice de timezone
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"

        # Tipos explícitos
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") \
                             .fillna(0).astype("int64")

        return df[["symbol"] + ohlcv_cols]

    @staticmethod
    def _validate_dates(start_date: str, end_date: str) -> None:
        """Valida formato y orden de fechas."""
        fmt = "%Y-%m-%d"
        try:
            start = datetime.strptime(start_date, fmt).date()
            end   = datetime.strptime(end_date,   fmt).date()
        except ValueError as exc:
            raise ValueError(
                f"Formato de fecha inválido. Use YYYY-MM-DD. Error: {exc}"
            ) from exc

        if start >= end:
            raise ValueError(
                f"start_date ({start_date}) debe ser anterior a end_date ({end_date})"
            )

        if end > date.today():
            raise ValueError(
                f"end_date ({end_date}) no puede ser una fecha futura"
            )