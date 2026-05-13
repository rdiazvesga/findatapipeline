"""
Módulo de validación y reporte de calidad de datos financieros.

Responsabilidad única: recibir datos OHLCV crudos y producir
datos limpios + reporte de calidad con score numérico.
No descarga, no almacena.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """
    Reporte de calidad para un activo.

    Attributes:
        symbol:          Ticker del activo.
        total_rows:      Total de filas antes de limpieza.
        null_rows:       Filas con al menos un valor nulo en OHLCV.
        ohlc_violations: Filas con high < low o close fuera de rango.
        outlier_rows:    Filas con retorno diario > threshold (posibles errores).
        gap_count:       Gaps temporales inesperados (días hábiles faltantes).
        duplicate_rows:  Fechas duplicadas encontradas.
        rows_removed:    Total de filas eliminadas en limpieza.
        quality_score:   Score 0-100 (100 = datos perfectos).
        issues:          Lista de descripciones de problemas encontrados.
        is_usable:       True si quality_score >= min_score (default 70).
    """
    symbol:          str
    total_rows:      int        = 0
    null_rows:       int        = 0
    ohlc_violations: int        = 0
    outlier_rows:    int        = 0
    gap_count:       int        = 0
    duplicate_rows:  int        = 0
    rows_removed:    int        = 0
    quality_score:   float      = 0.0
    issues:          list[str]  = field(default_factory=list)
    is_usable:       bool       = True

    def summary(self) -> str:
        """Resumen legible del reporte."""
        status = "✓ USABLE" if self.is_usable else "✗ NO USABLE"
        return (
            f"{self.symbol:<12} score={self.quality_score:>5.1f}  "
            f"{status}  filas={self.total_rows}  "
            f"nulos={self.null_rows}  gaps={self.gap_count}  "
            f"eliminadas={self.rows_removed}"
        )


class DataQualityChecker:
    """
    Valida y limpia datos OHLCV financieros.

    Ejecuta cinco verificaciones en orden:
      1. Duplicados en fechas
      2. Valores nulos en columnas OHLCV
      3. Violaciones de integridad OHLCV (high >= low, close dentro rango)
      4. Outliers por retorno diario extremo
      5. Gaps temporales inesperados

    Example:
        checker = DataQualityChecker()
        clean_df, report = checker.validate(raw_df, "SPY")
    """

    def __init__(
        self,
        outlier_threshold:  float = 0.15,   # retorno diario > 15% = sospechoso
        min_quality_score:  float = 70.0,    # score mínimo para considerar usable
        max_gap_days:       int   = 5,       # gaps > 5 días hábiles = problema
    ):
        """
        Args:
            outlier_threshold: Retorno diario absoluto que activa alerta de outlier.
            min_quality_score: Score mínimo para marcar dataset como usable.
            max_gap_days:      Máximo de días hábiles consecutivos sin datos.
        """
        self.outlier_threshold = outlier_threshold
        self.min_quality_score = min_quality_score
        self.max_gap_days      = max_gap_days

    def validate(
        self,
        df:     pd.DataFrame,
        symbol: str,
    ) -> tuple[pd.DataFrame, QualityReport]:
        """
        Valida y limpia un DataFrame OHLCV.

        Args:
            df:     DataFrame con columnas [open, high, low, close, volume]
                    e índice DatetimeIndex.
            symbol: Ticker del activo (solo para logging y reporte).

        Returns:
            Tuple (clean_df, report):
              - clean_df: DataFrame limpio listo para almacenar.
              - report:   QualityReport con métricas y score.
        """
        if df is None or df.empty:
            report = QualityReport(symbol=symbol)
            report.issues.append("DataFrame vacío o None")
            report.is_usable = False
            return pd.DataFrame(), report

        report = QualityReport(symbol=symbol, total_rows=len(df))
        clean  = df.copy()

        # Ejecutar checks en orden
        clean = self._check_duplicates(clean, report)
        clean = self._check_nulls(clean, report)
        clean = self._check_ohlc_integrity(clean, report)
        clean = self._check_outliers(clean, report)
        self._check_gaps(clean, report)

        # Calcular filas eliminadas y score final
        report.rows_removed  = report.total_rows - len(clean)
        report.quality_score = self._compute_score(report)
        report.is_usable     = report.quality_score >= self.min_quality_score

        logger.info(report.summary())
        if not report.is_usable:
            logger.warning(
                "%s marcado como NO USABLE (score=%.1f)",
                symbol, report.quality_score,
            )

        return clean, report

    def validate_multiple(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, QualityReport]]:
        """
        Valida múltiples activos.

        Returns:
            Tuple (clean_datasets, reports) donde clean_datasets
            incluye solo activos con is_usable=True.
        """
        clean_datasets: dict[str, pd.DataFrame]   = {}
        reports:        dict[str, QualityReport]  = {}

        for symbol, df in datasets.items():
            clean_df, report = self.validate(df, symbol)
            reports[symbol] = report
            if report.is_usable:
                clean_datasets[symbol] = clean_df

        usable   = sum(1 for r in reports.values() if r.is_usable)
        unusable = len(reports) - usable
        logger.info(
            "validate_multiple: %d usables, %d descartados",
            usable, unusable,
        )
        return clean_datasets, reports

    # ── checks privados ────────────────────────────────────────────────

    def _check_duplicates(
        self, df: pd.DataFrame, report: QualityReport
    ) -> pd.DataFrame:
        """Elimina fechas duplicadas conservando la primera ocurrencia."""
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            report.duplicate_rows = int(dupes)
            report.issues.append(f"{dupes} fechas duplicadas eliminadas")
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _check_nulls(
        self, df: pd.DataFrame, report: QualityReport
    ) -> pd.DataFrame:
        """Elimina filas con nulos en columnas OHLCV críticas."""
        ohlcv_cols = [c for c in ["open","high","low","close","volume"]
                      if c in df.columns]
        null_mask  = df[ohlcv_cols].isnull().any(axis=1)
        null_count = int(null_mask.sum())

        if null_count > 0:
            pct = null_count / len(df) * 100
            report.null_rows = null_count
            report.issues.append(
                f"{null_count} filas con nulos en OHLCV ({pct:.1f}%)"
            )
            df = df[~null_mask]
        return df

    def _check_ohlc_integrity(
        self, df: pd.DataFrame, report: QualityReport
    ) -> pd.DataFrame:
        """
        Elimina filas que violan integridad OHLCV:
          - high < low
          - close fuera del rango [low, high]
          - open fuera del rango [low, high]
          - cualquier precio <= 0
        """
        if not all(c in df.columns for c in ["open","high","low","close"]):
            return df

        violations = (
            (df["high"] < df["low"])                    |
            (df["close"] < df["low"])                   |
            (df["close"] > df["high"])                  |
            (df["open"]  < df["low"])                   |
            (df["open"]  > df["high"])                  |
            (df[["open","high","low","close"]] <= 0).any(axis=1)
        )
        count = int(violations.sum())
        if count > 0:
            report.ohlc_violations = count
            report.issues.append(
                f"{count} filas con violaciones de integridad OHLCV"
            )
            df = df[~violations]
        return df

    def _check_outliers(
        self, df: pd.DataFrame, report: QualityReport
    ) -> pd.DataFrame:
        """
        Marca (pero NO elimina) filas con retornos diarios extremos.
        Los outliers se reportan pero se conservan: pueden ser
        movimientos reales (crisis, splits, events) que el analista
        debe decidir si descartar.
        """
        if "close" not in df.columns or len(df) < 2:
            return df

        returns      = df["close"].pct_change().abs()
        outlier_mask = returns > self.outlier_threshold
        count        = int(outlier_mask.sum())

        if count > 0:
            report.outlier_rows = count
            pct = count / len(df) * 100
            # Obtener fechas de outliers para el reporte
            outlier_dates = df.index[outlier_mask].strftime("%Y-%m-%d").tolist()
            report.issues.append(
                f"{count} retornos diarios > {self.outlier_threshold*100:.0f}% "
                f"({pct:.1f}%): {outlier_dates[:5]}"
                + (" ..." if len(outlier_dates) > 5 else "")
            )
        return df

    def _check_gaps(
        self, df: pd.DataFrame, report: QualityReport
    ) -> None:
        """
        Detecta gaps temporales inesperados en días hábiles.
        Solo reporta, no elimina (un gap puede ser festivo local).
        """
        if len(df) < 2:
            return

        # Calcular diferencia en días calendario entre fechas consecutivas
        date_diffs = pd.Series(df.index).diff().dt.days.dropna()

        # Un gap es sospechoso si > max_gap_days días calendario
        # (cubre fin de semana + algunos festivos)
        threshold  = self.max_gap_days
        large_gaps = date_diffs[date_diffs > threshold]

        if len(large_gaps) > 0:
            report.gap_count = len(large_gaps)
            report.issues.append(
                f"{len(large_gaps)} gaps temporales > {threshold} días"
            )

    def _compute_score(self, report: QualityReport) -> float:
        """
        Calcula score de calidad 0-100.

        Penalizaciones:
          - Nulos:            -30 puntos máximo (proporcional al %)
          - Violaciones OHLC: -25 puntos máximo
          - Outliers:         -15 puntos máximo
          - Gaps:             -20 puntos máximo
          - Duplicados:       -10 puntos máximo
        """
        if report.total_rows == 0:
            return 0.0

        score = 100.0

        # Penalización por nulos (proporcional)
        null_pct = report.null_rows / report.total_rows
        score   -= min(30.0, null_pct * 300)

        # Penalización por violaciones OHLC
        ohlc_pct = report.ohlc_violations / report.total_rows
        score   -= min(25.0, ohlc_pct * 250)

        # Penalización por outliers
        outlier_pct = report.outlier_rows / report.total_rows
        score      -= min(15.0, outlier_pct * 150)

        # Penalización por gaps (absoluta, por cantidad)
        score -= min(20.0, report.gap_count * 2.0)

        # Penalización por duplicados
        dup_pct = report.duplicate_rows / report.total_rows
        score  -= min(10.0, dup_pct * 100)

        return round(max(0.0, score), 2)