"""
Métricas de volatilidad para series de precios financieros.

Implementa tres estimadores distintos con propiedades
estadísticas diferentes según el uso.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class VolatilityCalculator:
    """
    Calcula estimadores de volatilidad sobre series de retornos.

    Tres estimadores disponibles:
      - Realized:  histórica rolling (simple, sin supuestos distribucionales)
      - EWMA:      da más peso a observaciones recientes (sensible a shocks)
      - Parkinson: usa high/low además de close (más eficiente estadísticamente)

    Example:
        calc = VolatilityCalculator()
        vol = calc.realized(df["close"], window=21)
        vol_regime = calc.regime(df["close"])
    """

    # ── estimadores de volatilidad ────────────────────────────────────

    def realized(
        self,
        prices: pd.Series,
        window: int = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """
        Volatilidad realizada histórica rolling.

        Desviación estándar de log-retornos en ventana de N períodos,
        opcionalmente anualizada multiplicando por sqrt(252).

        Es el estimador más simple y robusto. Usa como referencia base.

        Args:
            prices:           Serie de precios de cierre.
            window:           Períodos en la ventana rolling.
            annualize:        Si True, multiplica por sqrt(periods_per_year).
            periods_per_year: Factor de anualización.
        """
        log_ret = np.log(prices / prices.shift(1))
        vol = log_ret.rolling(window, min_periods=window).std()

        if annualize:
            vol = vol * np.sqrt(periods_per_year)

        return vol

    def ewma(
        self,
        prices:  pd.Series,
        span:    int  = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """
        Volatilidad EWMA (Exponentially Weighted Moving Average).

        Modelo RiskMetrics de JP Morgan (1994). Da más peso a
        retornos recientes: λ = 1 - 2/(span+1).

        Más reactivo que realized() ante shocks de volatilidad.
        Útil para risk management dinámico.

        Args:
            span: Controla el decay. span=21 ≈ λ=0.91 (RiskMetrics usa λ=0.94).
        """
        log_ret = np.log(prices / prices.shift(1))
        vol = log_ret.ewm(span=span, adjust=False).std()

        if annualize:
            vol = vol * np.sqrt(periods_per_year)

        return vol

    def parkinson(
        self,
        high:  pd.Series,
        low:   pd.Series,
        window: int  = 21,
        annualize: bool = True,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        """
        Estimador de volatilidad de Parkinson (1980).

        Usa el rango high-low en lugar de solo close-to-close.
        Es 5x más eficiente estadísticamente que el estimador
        de close-to-close cuando los precios siguen un proceso
        de Wiener (Parkinson 1980, Journal of Business).

        Formula: sigma = sqrt(1/(4*ln2) * E[ln(H/L)^2])

        Limitación: subestima volatilidad cuando hay gaps
        entre sesiones (overnight gaps).
        """
        log_hl = np.log(high / low)
        parkinson_daily = (log_hl ** 2) / (4 * np.log(2))
        vol = np.sqrt(
            parkinson_daily.rolling(window, min_periods=window).mean()
        )

        if annualize:
            vol = vol * np.sqrt(periods_per_year)

        return vol

    # ── análisis de régimen ───────────────────────────────────────────

    def regime(
        self,
        prices:       pd.Series,
        short_window: int = 21,
        long_window:  int = 63,
    ) -> pd.DataFrame:
        """
        Detecta régimen de volatilidad: HIGH / NORMAL / LOW.

        Compara volatilidad rolling de corto plazo contra largo plazo.
        Útil para ajustar estrategias según el régimen actual.

        Returns:
            DataFrame con columnas:
              - vol_short:  volatilidad 21d anualizada
              - vol_long:   volatilidad 63d anualizada
              - vol_ratio:  vol_short / vol_long
              - regime:     'HIGH' | 'NORMAL' | 'LOW'
        """
        vol_short = self.realized(prices, window=short_window)
        vol_long  = self.realized(prices, window=long_window)
        vol_ratio = vol_short / vol_long

        conditions = [
            vol_ratio > 1.25,
            vol_ratio < 0.75,
        ]
        choices = ["HIGH", "LOW"]
        regime_series = pd.Series(
            np.select(conditions, choices, default="NORMAL"),
            index=prices.index,
        )

        return pd.DataFrame({
            "vol_short": vol_short,
            "vol_long":  vol_long,
            "vol_ratio": vol_ratio,
            "regime":    regime_series,
        })

    def summary(
        self,
        prices: pd.Series,
        high:   pd.Series = None,
        low:    pd.Series = None,
    ) -> dict:
        """Resumen de métricas de volatilidad para un activo."""
        vol_21  = self.realized(prices, window=21).dropna()
        vol_63  = self.realized(prices, window=63).dropna()

        result = {
            "vol_realized_21d_current": round(
                float(vol_21.iloc[-1]) * 100, 2
            ) if not vol_21.empty else np.nan,
            "vol_realized_63d_current": round(
                float(vol_63.iloc[-1]) * 100, 2
            ) if not vol_63.empty else np.nan,
            "vol_realized_21d_mean": round(
                float(vol_21.mean()) * 100, 2
            ) if not vol_21.empty else np.nan,
            "vol_realized_21d_max": round(
                float(vol_21.max()) * 100, 2
            ) if not vol_21.empty else np.nan,
        }

        if high is not None and low is not None:
            vol_park = self.parkinson(high, low, window=21).dropna()
            result["vol_parkinson_21d_current"] = round(
                float(vol_park.iloc[-1]) * 100, 2
            ) if not vol_park.empty else np.nan

        return result