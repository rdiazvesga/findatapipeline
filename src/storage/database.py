"""
Módulo de almacenamiento en PostgreSQL.

Responsabilidad única: persistir datos limpios validados
y exponer queries para recuperarlos.
"""
import logging
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Gestiona la conexión y operaciones sobre PostgreSQL.

    Usa upsert (INSERT ... ON CONFLICT DO UPDATE) para que
    re-ejecuciones del pipeline sean idempotentes: no duplican datos.

    Example:
        db = DatabaseManager()
        db.upsert_prices(clean_df)
        df = db.get_prices("SPY", "2023-01-01", "2024-01-01")
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Args:
            db_url: URL de conexión PostgreSQL. Si es None, lee de settings.
        """
        if db_url is None:
            import sys
            sys.path.insert(0, ".")
            from config.settings import DB_URL
            db_url = DB_URL

        self._engine: Engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=2,
            pool_pre_ping=True,
        )

    @contextmanager
    def _get_connection(self):
        """Context manager para conexión con commit/rollback automático."""
        with self._engine.begin() as conn:
            yield conn

    # ── escritura ──────────────────────────────────────────────────────

    def upsert_prices(self, df: pd.DataFrame) -> int:
        """
        Inserta o actualiza precios en price_history.

        Usa ON CONFLICT DO UPDATE para ser idempotente:
        re-ejecutar no crea duplicados.

        Args:
            df: DataFrame con columnas [symbol, open, high, low, close, volume]
                e índice DatetimeIndex (date).

        Returns:
            Número de filas procesadas.
        """
        if df is None or df.empty:
            logger.warning("upsert_prices: DataFrame vacío, nada que guardar")
            return 0

        records = self._df_to_records(df)
        if not records:
            return 0

        upsert_sql = text("""
            INSERT INTO price_history
                (symbol, date, open, high, low, close, volume)
            VALUES
                (:symbol, :date, :open, :high, :low, :close, :volume)
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        with self._get_connection() as conn:
            conn.execute(upsert_sql, records)

        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "?"
        logger.info(
            "upsert_prices: %d filas guardadas para %s",
            len(records), symbol,
        )
        return len(records)

    def upsert_multiple(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> dict[str, int]:
        """
        Persiste múltiples activos.

        Returns:
            Diccionario {symbol: n_filas_guardadas}.
        """
        results: dict[str, int] = {}
        for symbol, df in datasets.items():
            try:
                n = self.upsert_prices(df)
                results[symbol] = n
            except Exception as exc:
                logger.error("Error guardando %s: %s", symbol, exc)
                results[symbol] = 0

        total = sum(results.values())
        logger.info(
            "upsert_multiple: %d filas totales en %d activos",
            total, len(datasets),
        )
        return results

    def save_quality_report(self, symbol: str, report) -> None:
        """Persiste el reporte de calidad en data_quality_log."""
        import json
        sql = text("""
            INSERT INTO data_quality_log
                (symbol, total_rows, null_rows, outlier_rows,
                 gap_count, quality_score, issues)
            VALUES
                (:symbol, :total_rows, :null_rows, :outlier_rows,
                 :gap_count, :quality_score, :issues)
        """)
        with self._get_connection() as conn:
            conn.execute(sql, {
                "symbol":        symbol,
                "total_rows":    report.total_rows,
                "null_rows":     report.null_rows,
                "outlier_rows":  report.outlier_rows,
                "gap_count":     report.gap_count,
                "quality_score": report.quality_score,
                "issues":        json.dumps(report.issues),
            })

    # ── lectura ────────────────────────────────────────────────────────

    def get_prices(
        self,
        symbol:     str,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Recupera precios históricos de PostgreSQL.

        Args:
            symbol:     Ticker del activo.
            start_date: Fecha inicio 'YYYY-MM-DD' (opcional).
            end_date:   Fecha fin   'YYYY-MM-DD' (opcional).

        Returns:
            DataFrame con columnas OHLCV e índice DatetimeIndex,
            o DataFrame vacío si no hay datos.
        """
        conditions = ["symbol = :symbol"]
        params: dict = {"symbol": symbol}

        if start_date:
            conditions.append("date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            conditions.append("date <= :end_date")
            params["end_date"] = end_date

        where = " AND ".join(conditions)
        sql   = text(f"""
            SELECT date, symbol, open, high, low, close, volume
            FROM   price_history
            WHERE  {where}
            ORDER  BY date ASC
        """)

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn, params=params, parse_dates=["date"])

        if df.empty:
            logger.warning("get_prices: sin datos para %s", symbol)
            return pd.DataFrame()

        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        logger.info(
            "get_prices: %d filas recuperadas para %s",
            len(df), symbol,
        )
        return df

    def get_available_symbols(self) -> list[str]:
        """Lista de símbolos con datos en price_history."""
        sql = text("""
            SELECT   symbol, COUNT(*) as rows,
                     MIN(date) as date_from,
                     MAX(date) as date_to
            FROM     price_history
            GROUP BY symbol
            ORDER BY symbol
        """)
        with self._get_connection() as conn:
            result = conn.execute(sql)
            rows   = result.fetchall()

        symbols = []
        for row in rows:
            symbols.append(row[0])
            logger.info(
                "  %s: %d filas (%s → %s)",
                row[0], row[1], row[2], row[3],
            )
        return symbols

    def get_row_count(self, symbol: Optional[str] = None) -> int:
        """Cuenta total de filas, opcionalmente filtrado por símbolo."""
        if symbol:
            sql    = text(
                "SELECT COUNT(*) FROM price_history WHERE symbol = :s"
            )
            params = {"s": symbol}
        else:
            sql    = text("SELECT COUNT(*) FROM price_history")
            params = {}

        with self._get_connection() as conn:
            result = conn.execute(sql, params)
            return int(result.scalar())

    # ── helper privado ────────────────────────────────────────────────

    def _df_to_records(self, df: pd.DataFrame) -> list[dict]:
        """Convierte DataFrame a lista de dicts para SQLAlchemy."""
        records = []
        for idx, row in df.iterrows():
            records.append({
                "symbol": str(row.get("symbol", "")),
                "date":   idx.date() if hasattr(idx, "date") else idx,
                "open":   float(row["open"])
                          if pd.notna(row["open"])   else None,
                "high":   float(row["high"])
                          if pd.notna(row["high"])   else None,
                "low":    float(row["low"])
                          if pd.notna(row["low"])    else None,
                "close":  float(row["close"])
                          if pd.notna(row["close"])  else None,
                "volume": int(row["volume"])
                          if pd.notna(row["volume"]) else None,
            })
        return records