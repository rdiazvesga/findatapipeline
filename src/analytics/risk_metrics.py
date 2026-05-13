"""
Métricas de riesgo ajustado para evaluación de performance.

Implementa ratios estándar de la industria de gestión de activos.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class RiskMetricsCalculator:
    """
    Calcula métricas de riesgo ajustado al retorno.

    References:
        Sharpe (1966): "Mutual Fund Performance"
        Sortino, van der Meer (1991): "Downside Risk"
        Rockafellar, Uryasev (2000): "Optimization of CVaR"

    Example:
        calc = RiskMetricsCalculator(risk_free_rate=0.04)
        sharpe = calc.sharpe_ratio(returns)
        var    = calc.value_at_risk(returns, confidence=0.95)
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Tasa libre de riesgo anual (default 4%).
                            Se usa como benchmark en Sharpe y Sortino.
        """
        self.risk_free_rate = risk_free_rate

    # ── ratios de performance ─────────────────────────────────────────

    def sharpe_ratio(
        self,
        returns:          pd.Series,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """
        Ratio de Sharpe anualizado.

        Formula: (E[r] - rf) / std(r) * sqrt(T)

        Mide el exceso de retorno por unidad de riesgo total.
        Limitación: penaliza igualmente la volatilidad al alza
        y a la baja, lo cual no es lo que los inversores perciben
        como riesgo real.

        Interpretación:
          < 0.5:  pobre
          0.5-1:  aceptable
          1-2:    bueno
          > 2:    excelente (posiblemente overfitted en backtest)
        """
        ret = returns.dropna()
        if len(ret) < 2:
            return np.nan

        daily_rf   = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess_ret = ret - daily_rf
        std        = ret.std()

        if std == 0:
            return np.nan

        return float(
            (excess_ret.mean() / std) * np.sqrt(periods_per_year)
        )

    def sortino_ratio(
        self,
        returns:          pd.Series,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """
        Ratio de Sortino anualizado.

        Como Sharpe, pero el denominador usa solo la desviación
        a la baja (downside deviation), no la volatilidad total.

        Más apropiado que Sharpe cuando la distribución de retornos
        es asimétrica (lo habitual en activos financieros reales).

        Formula: (E[r] - rf) / downside_std(r) * sqrt(T)
        """
        ret = returns.dropna()
        if len(ret) < 2:
            return np.nan

        daily_rf      = (1 + self.risk_free_rate) ** (1 / periods_per_year) - 1
        excess_ret    = ret - daily_rf
        downside_ret  = ret[ret < daily_rf] - daily_rf
        downside_std  = downside_ret.std()

        if downside_std == 0 or len(downside_ret) == 0:
            return np.nan

        return float(
            (excess_ret.mean() / downside_std) * np.sqrt(periods_per_year)
        )

    # ── value at risk ─────────────────────────────────────────────────

    def value_at_risk(
        self,
        returns:    pd.Series,
        confidence: float = 0.95,
        method:     str   = "historical",
    ) -> float:
        """
        Value at Risk (VaR) como pérdida máxima esperada.

        Responde: "¿Cuál es la pérdida máxima con X% de confianza
        en un período?"

        Args:
            returns:    Serie de retornos simples.
            confidence: Nivel de confianza (0.95 → percentil 5%).
            method:     'historical' | 'parametric'
                        - historical: no asume distribución
                        - parametric: asume normalidad

        Returns:
            Número negativo. Ej: -0.023 → pérdida máxima del 2.3%.

        Importante: VaR no dice nada sobre la magnitud de pérdidas
        más allá del threshold. Para eso usar CVaR (ver abajo).
        """
        ret = returns.dropna()
        if ret.empty:
            return np.nan

        if method == "historical":
            return float(ret.quantile(1 - confidence))
        elif method == "parametric":
            from scipy import stats
            z   = stats.norm.ppf(1 - confidence)
            return float(ret.mean() + z * ret.std())
        else:
            raise ValueError(f"method debe ser 'historical' o 'parametric'")

    def cvar(
        self,
        returns:    pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional Value at Risk (CVaR) o Expected Shortfall.

        Responde: "Dado que estamos en el peor X% de escenarios,
        ¿cuál es la pérdida promedio?"

        Es más informativo que VaR porque captura la magnitud
        de las pérdidas en la cola de la distribución.
        Preferido en risk management moderno (Basel III, Solvencia II).

        Returns:
            Número negativo. Ej: -0.041 → pérdida promedio
            en el peor 5% de los días fue -4.1%.
        """
        ret       = returns.dropna()
        var_level = self.value_at_risk(ret, confidence)
        tail      = ret[ret <= var_level]

        return float(tail.mean()) if not tail.empty else np.nan

    # ── resumen completo ──────────────────────────────────────────────

    def full_summary(
        self,
        prices:  pd.Series,
        returns: pd.Series = None,
    ) -> dict:
        """
        Resumen completo de métricas de riesgo para un activo.

        Args:
            prices:  Serie de precios de cierre.
            returns: Serie de retornos simples (se calcula si es None).
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        ret = returns.dropna()

        return {
            "sharpe_ratio":        round(self.sharpe_ratio(ret), 3),
            "sortino_ratio":       round(self.sortino_ratio(ret), 3),
            "var_95_historical":   round(
                self.value_at_risk(ret, 0.95) * 100, 3
            ),
            "cvar_95":             round(
                self.cvar(ret, 0.95) * 100, 3
            ),
            "var_99_historical":   round(
                self.value_at_risk(ret, 0.99) * 100, 3
            ),
            "best_day_pct":        round(float(ret.max()) * 100, 3),
            "worst_day_pct":       round(float(ret.min()) * 100, 3),
            "pct_positive_days":   round(
                float((ret > 0).mean()) * 100, 2
            ),
            "risk_free_rate_used": self.risk_free_rate,
        }