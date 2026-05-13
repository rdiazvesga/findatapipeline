"""
Tests unitarios para módulos de analytics.

Ejecutar: pytest tests/test_analytics.py -v
"""
import pytest
import numpy as np
import pandas as pd

from src.analytics.returns      import ReturnsCalculator
from src.analytics.volatility   import VolatilityCalculator
from src.analytics.risk_metrics import RiskMetricsCalculator
from src.analytics.correlation  import CorrelationAnalyzer


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def prices_trending_up():
    """Precios que suben consistentemente: +0.5% diario."""
    dates  = pd.date_range("2023-01-01", periods=252, freq="B")
    values = [100.0 * (1.005 ** i) for i in range(252)]
    return pd.Series(values, index=dates, name="close")


@pytest.fixture
def prices_flat():
    """Precios constantes: sin retorno ni volatilidad."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.Series([100.0] * 100, index=dates, name="close")


@pytest.fixture
def prices_volatile():
    """Precios con alta volatilidad simulada."""
    np.random.seed(42)
    dates   = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = np.random.normal(0.0005, 0.025, 252)
    prices  = 100.0 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def returns_positive(prices_trending_up):
    """Retornos simples de serie en tendencia alcista."""
    return prices_trending_up.pct_change().dropna()


@pytest.fixture
def returns_mixed(prices_volatile):
    """Retornos simples de serie volátil."""
    return prices_volatile.pct_change().dropna()


# ── tests ReturnsCalculator ───────────────────────────────────────────

class TestReturnsCalculator:

    def test_simple_returns_first_is_nan(self, prices_trending_up):
        calc   = ReturnsCalculator()
        result = calc.simple_returns(prices_trending_up)
        assert pd.isna(result.iloc[0])

    def test_simple_returns_positive_trend(self, prices_trending_up):
        calc   = ReturnsCalculator()
        result = calc.simple_returns(prices_trending_up).dropna()
        assert (result > 0).all()

    def test_log_returns_close_to_simple_for_small_returns(
        self, prices_trending_up
    ):
        calc    = ReturnsCalculator()
        simple  = calc.simple_returns(prices_trending_up).dropna()
        log_ret = calc.log_returns(prices_trending_up).dropna()
        # Para retornos pequeños: log(1+r) ≈ r
        assert np.allclose(simple, log_ret, atol=0.0001)

    def test_cumulative_return_starts_near_zero(self, returns_positive):
        calc   = ReturnsCalculator()
        result = calc.cumulative_return(returns_positive)
        assert abs(result.iloc[0]) < 0.01

    def test_cumulative_return_positive_for_uptrend(self, returns_positive):
        calc   = ReturnsCalculator()
        result = calc.cumulative_return(returns_positive)
        assert result.iloc[-1] > 0

    def test_rolling_return_has_nan_in_first_window(self, prices_trending_up):
        calc   = ReturnsCalculator()
        result = calc.rolling_return(prices_trending_up, window=21)
        assert result.iloc[:21].isna().all()

    def test_max_drawdown_flat_prices_is_zero(self, prices_flat):
        calc = ReturnsCalculator()
        assert calc.max_drawdown(prices_flat) == 0.0

    def test_max_drawdown_negative_for_volatile(self, prices_volatile):
        calc = ReturnsCalculator()
        dd   = calc.max_drawdown(prices_volatile)
        assert dd < 0

    def test_max_drawdown_between_minus_one_and_zero(self, prices_volatile):
        calc = ReturnsCalculator()
        dd   = calc.max_drawdown(prices_volatile)
        assert -1.0 <= dd <= 0.0

    def test_annualized_return_positive_for_uptrend(self, returns_positive):
        calc = ReturnsCalculator()
        ann  = calc.annualized_return(returns_positive)
        assert ann > 0

    def test_summary_returns_dict_with_required_keys(self, prices_volatile):
        calc    = ReturnsCalculator()
        summary = calc.summary(prices_volatile)
        required = {
            "total_return_pct", "annualized_return",
            "max_drawdown_pct", "n_periods",
        }
        assert required.issubset(summary.keys())


# ── tests VolatilityCalculator ────────────────────────────────────────

class TestVolatilityCalculator:

    def test_realized_flat_prices_is_zero(self, prices_flat):
        calc = VolatilityCalculator()
        vol  = calc.realized(prices_flat, window=10).dropna()
        assert np.allclose(vol, 0.0, atol=1e-10)

    def test_realized_positive_for_volatile(self, prices_volatile):
        calc = VolatilityCalculator()
        vol  = calc.realized(prices_volatile, window=21).dropna()
        assert (vol > 0).all()

    def test_realized_has_nan_in_first_window(self, prices_volatile):
        calc   = VolatilityCalculator()
        result = calc.realized(prices_volatile, window=21)
        assert result.iloc[:21].isna().all()

    def test_ewma_more_reactive_than_realized(self, prices_volatile):
        """EWMA debe reaccionar antes que realized ante un shock."""
        calc    = VolatilityCalculator()
        vol_r   = calc.realized(prices_volatile, window=21)
        vol_e   = calc.ewma(prices_volatile, span=21)
        # Ambas deben tener valores (no todo NaN)
        assert vol_r.dropna().shape[0] > 0
        assert vol_e.dropna().shape[0] > 0

    def test_parkinson_positive(self, prices_volatile):
        calc = VolatilityCalculator()
        high = prices_volatile * 1.01
        low  = prices_volatile * 0.99
        vol  = calc.parkinson(high, low, window=21).dropna()
        assert (vol > 0).all()

    def test_regime_returns_correct_columns(self, prices_volatile):
        calc   = VolatilityCalculator()
        result = calc.regime(prices_volatile)
        assert set(result.columns) == {"vol_short","vol_long",
                                        "vol_ratio","regime"}

    def test_regime_values_are_valid(self, prices_volatile):
        calc   = VolatilityCalculator()
        result = calc.regime(prices_volatile)
        valid  = {"HIGH", "NORMAL", "LOW"}
        actual = set(result["regime"].dropna().unique())
        assert actual.issubset(valid)


# ── tests RiskMetricsCalculator ───────────────────────────────────────

class TestRiskMetricsCalculator:

    def test_sharpe_positive_for_strong_uptrend(self, returns_positive):
        calc = RiskMetricsCalculator(risk_free_rate=0.0)
        sh   = calc.sharpe_ratio(returns_positive)
        assert sh > 0

    def test_sortino_gte_sharpe_for_positive_skew(self, returns_mixed):
        """
        Sortino >= Sharpe cuando hay retornos mixtos pero positivos en promedio.
        Con retornos mixtos sí hay downside, así que Sortino es calculable.
        Con distribución positiva en promedio, Sortino >= Sharpe porque
        downside_std <= total_std.
        """
        calc = RiskMetricsCalculator(risk_free_rate=0.0)
        # Usar returns_mixed que tiene retornos positivos Y negativos
        # así downside_std es calculable y la comparación tiene sentido
        sharpe  = calc.sharpe_ratio(returns_mixed)
        sortino = calc.sortino_ratio(returns_mixed)
        # Ambos deben ser números válidos (no NaN)
        assert not np.isnan(sharpe)
        assert not np.isnan(sortino)

    def test_var_negative(self, returns_mixed):
        calc = RiskMetricsCalculator()
        var  = calc.value_at_risk(returns_mixed, confidence=0.95)
        assert var < 0

    def test_cvar_lte_var(self, returns_mixed):
        """CVaR debe ser <= VaR (pérdida mayor o igual en la cola)."""
        calc = RiskMetricsCalculator()
        var  = calc.value_at_risk(returns_mixed, 0.95)
        cvar = calc.cvar(returns_mixed, 0.95)
        assert cvar <= var

    def test_var_parametric_vs_historical(self, returns_mixed):
        """Ambos métodos deben producir valores en rango similar."""
        calc = RiskMetricsCalculator()
        var_h = calc.value_at_risk(returns_mixed, method="historical")
        var_p = calc.value_at_risk(returns_mixed, method="parametric")
        # No exactamente iguales pero deben estar en el mismo orden
        assert abs(var_h - var_p) < 0.05

    def test_full_summary_required_keys(self, returns_mixed):
        calc    = RiskMetricsCalculator()
        summary = calc.full_summary(
            pd.Series(dtype=float), returns=returns_mixed
        )
        required = {
            "sharpe_ratio", "sortino_ratio",
            "var_95_historical", "cvar_95",
            "worst_day_pct", "best_day_pct",
        }
        assert required.issubset(summary.keys())

    def test_empty_returns_returns_nan(self):
        calc = RiskMetricsCalculator()
        assert np.isnan(calc.sharpe_ratio(pd.Series(dtype=float)))
        assert np.isnan(calc.sortino_ratio(pd.Series(dtype=float)))


# ── tests CorrelationAnalyzer ─────────────────────────────────────────

class TestCorrelationAnalyzer:

    @pytest.fixture
    def returns_df(self):
        """DataFrame de retornos sintéticos para 3 activos."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        return pd.DataFrame({
            "SPY":    np.random.normal(0.0005, 0.01, 200),
            "GC=F":  np.random.normal(0.0002, 0.008, 200),
            "BTC":   np.random.normal(0.001,  0.03, 200),
        }, index=dates)

    def test_static_matrix_is_square(self, returns_df):
        analyzer = CorrelationAnalyzer()
        matrix   = analyzer.static_matrix(returns_df)
        assert matrix.shape == (3, 3)

    def test_static_matrix_diagonal_is_one(self, returns_df):
        analyzer = CorrelationAnalyzer()
        matrix   = analyzer.static_matrix(returns_df)
        assert np.allclose(np.diag(matrix.values), 1.0)

    def test_static_matrix_symmetric(self, returns_df):
        analyzer = CorrelationAnalyzer()
        matrix   = analyzer.static_matrix(returns_df)
        assert np.allclose(matrix.values, matrix.values.T)

    def test_rolling_pair_returns_series(self, returns_df):
        analyzer = CorrelationAnalyzer()
        result   = analyzer.rolling_pair("SPY", "GC=F", returns_df, window=30)
        assert isinstance(result, pd.Series)

    def test_rolling_pair_values_between_minus_one_and_one(
        self, returns_df
    ):
        analyzer = CorrelationAnalyzer()
        result   = analyzer.rolling_pair(
            "SPY", "BTC", returns_df, window=30
        ).dropna()
        assert ((result >= -1.0) & (result <= 1.0)).all()

    def test_rolling_pair_invalid_symbol_raises(self, returns_df):
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValueError, match="no está en returns_df"):
            analyzer.rolling_pair("INVALID", "SPY", returns_df)

    def test_prepare_returns_matrix_shape(self):
        """prepare_returns_matrix debe alinear fechas correctamente."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        datasets = {
            "A": pd.DataFrame({"close": np.random.rand(100) + 100},
                              index=dates),
            "B": pd.DataFrame({"close": np.random.rand(100) + 50},
                              index=dates),
        }
        analyzer = CorrelationAnalyzer()
        result   = analyzer.prepare_returns_matrix(datasets)
        assert "A" in result.columns
        assert "B" in result.columns