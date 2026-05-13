"""
Cálculo de métricas de retorno para series de precios financieros.

Todas las funciones reciben pd.Series de precios de cierre
y devuelven pd.Series de la misma longitud (con NaN donde
no hay suficiente historia).
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class ReturnsCalculator:
    """
    Calcula métricas de retorno sobre series de precios.

    Convención: las funciones reciben precios de cierre (close)
    como pd.Series con índice DatetimeIndex. Devuelven pd.Series
    con el mismo índice, con NaN en posiciones sin datos suficientes.

    Example:
        calc = ReturnsCalculator()
        log_ret = calc.log_returns(df["close"])
        cum_ret = calc.cumulative_return(log_ret)
    """

    # ── retornos básicos ──────────────────────────────────────────────

    def simple_returns(self, prices: pd.Series) -> pd.Series:
        """
        Retornos simples período a período.

        Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

        Uso: comparar rendimientos en el mismo período.
        Limitación: no son aditivos en el tiempo.
        """
        return prices.pct_change()

    def log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Retornos logarítmicos período a período.

        Formula: r_t = ln(P_t / P_{t-1})

        Ventajas sobre simple_returns:
          - Aditivos en el tiempo: r(t1→t3) = r(t1→t2) + r(t2→t3)
          - Aproximadamente normales para retornos pequeños
          - Útiles para modelos estadísticos
        """
        return np.log(prices / prices.shift(1))

    def rolling_return(
        self,
        prices: pd.Series,
        window: int,
    ) -> pd.Series:
        """
        Retorno acumulado log en ventana rolling de N períodos.

        Suma de log-retornos en la ventana = log(P_t / P_{t-N}).
        Útil para ver el momentum a distintas escalas temporales.

        Args:
            prices: Serie de precios de cierre.
            window: Número de períodos en la ventana.
        """
        log_ret = self.log_returns(prices)
        return log_ret.rolling(window, min_periods=window).sum()

    def cumulative_return(self, returns: pd.Series) -> pd.Series:
        """
        Retorno acumulado desde el inicio de la serie.

        Parte de 0.0 en la primera observación y acumula
        geométricamente. Útil para graficar equity curves.

        Formula: cum_r_t = prod(1 + r_i, i=1..t) - 1
        """
        return (1 + returns).cumprod() - 1

    # ── métricas anualizadas ──────────────────────────────────────────

    def annualized_return(
        self,
        returns: pd.Series,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """
        Retorno anualizado geométrico (CAGR).

        Formula: (prod(1+r_i))^(252/N) - 1

        Args:
            returns:          Serie de retornos simples.
            periods_per_year: 252 para datos diarios,
                              52 para semanales, 12 para mensuales.
        """
        if returns.dropna().empty:
            return np.nan

        total_return = (1 + returns.dropna()).prod()
        n_periods    = len(returns.dropna())
        n_years      = n_periods / periods_per_year

        if n_years <= 0:
            return np.nan

        return float(total_return ** (1.0 / n_years) - 1)

    # ── drawdown ─────────────────────────────────────────────────────

    def drawdown_series(self, prices: pd.Series) -> pd.Series:
        """
        Drawdown en cada punto como porcentaje del máximo anterior.

        Formula: DD_t = (P_t - max(P_0..P_t)) / max(P_0..P_t)

        Un valor de -0.15 significa que el precio está
        15% por debajo de su máximo histórico hasta ese momento.
        """
        rolling_max = prices.cummax()
        return (prices - rolling_max) / rolling_max

    def max_drawdown(self, prices: pd.Series) -> float:
        """
        Máximo drawdown en el período completo.

        Returns:
            Número negativo entre -1 y 0.
            Ej: -0.34 significa caída máxima del 34%.
        """
        dd = self.drawdown_series(prices)
        return float(dd.min()) if not dd.empty else np.nan

    def drawdown_duration(self, prices: pd.Series) -> int:
        """
        Duración del drawdown más largo en número de períodos.

        Útil para evaluar cuánto tiempo tarda el activo
        en recuperar un máximo anterior.
        """
        dd = self.drawdown_series(prices)
        in_drawdown = dd < 0

        max_duration = 0
        current_duration = 0

        for val in in_drawdown:
            if val:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def summary(self, prices: pd.Series) -> dict:
        """
        Resumen completo de métricas de retorno.

        Útil para reportes y comparaciones entre activos.
        """
        simple_ret = self.simple_returns(prices).dropna()
        log_ret    = self.log_returns(prices).dropna()

        return {
            "total_return_pct":    round(float(
                (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            ), 2) if len(prices) >= 2 else np.nan,
            "annualized_return":   round(
                self.annualized_return(simple_ret) * 100, 2
            ),
            "max_drawdown_pct":    round(
                self.max_drawdown(prices) * 100, 2
            ),
            "drawdown_duration_days": self.drawdown_duration(prices),
            "n_periods":           len(prices),
            "date_from":           str(prices.index.min().date()),
            "date_to":             str(prices.index.max().date()),
        }