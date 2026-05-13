"""
Tests unitarios para YahooFinanceIngester.

Ejecutar: pytest tests/test_ingestion.py -v
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ingestion.yahoo_finance import YahooFinanceIngester


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ingester():
    """Instancia de YahooFinanceIngester para tests."""
    return YahooFinanceIngester(max_retries=2, retry_delay=0.1)


@pytest.fixture
def sample_raw_df():
    """DataFrame con estructura que devuelve yfinance."""
    import numpy as np
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York")
    return pd.DataFrame({
        "Open":   [470.0, 471.5, 469.0, 472.0, 473.5],
        "High":   [472.0, 473.0, 471.5, 474.0, 475.0],
        "Low":    [468.5, 470.0, 468.0, 470.5, 472.0],
        "Close":  [471.0, 472.5, 470.0, 473.0, 474.5],
        "Volume": [80000000, 75000000, 90000000, 85000000, 70000000],
    }, index=dates)


# ── tests de validación de fechas ────────────────────────────────────

class TestDateValidation:

    def test_valid_dates_no_error(self, ingester):
        """Fechas válidas no levantan excepción."""
        ingester._validate_dates("2023-01-01", "2024-01-01")  # no debe fallar

    def test_invalid_format_raises(self, ingester):
        with pytest.raises(ValueError, match="Formato de fecha inválido"):
            ingester._validate_dates("01/01/2023", "2024-01-01")

    def test_start_after_end_raises(self, ingester):
        with pytest.raises(ValueError, match="debe ser anterior"):
            ingester._validate_dates("2024-06-01", "2024-01-01")

    def test_equal_dates_raises(self, ingester):
        with pytest.raises(ValueError, match="debe ser anterior"):
            ingester._validate_dates("2024-01-01", "2024-01-01")

    def test_future_end_date_raises(self, ingester):
        with pytest.raises(ValueError, match="no puede ser una fecha futura"):
            ingester._validate_dates("2024-01-01", "2099-12-31")


# ── tests de estandarización ─────────────────────────────────────────

class TestStandardize:

    def test_columns_renamed(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        expected_cols = {"symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(result.columns))

    def test_symbol_added(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        assert (result["symbol"] == "SPY").all()

    def test_index_has_no_timezone(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        assert result.index.tz is None

    def test_index_name_is_date(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        assert result.index.name == "date"

    def test_ohlc_are_float64(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        for col in ["open", "high", "low", "close"]:
            assert result[col].dtype == "float64", f"{col} no es float64"

    def test_volume_is_int64(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        assert result["volume"].dtype == "int64"

    def test_row_count_preserved(self, ingester, sample_raw_df):
        result = ingester._standardize(sample_raw_df, "SPY")
        assert len(result) == len(sample_raw_df)


# ── tests de fetch_single con mock ───────────────────────────────────

class TestFetchSingle:

    def test_returns_dataframe_on_success(self, ingester, sample_raw_df):
        with patch.object(ingester, "_download", return_value=sample_raw_df):
            result = ingester.fetch_single("SPY", "2024-01-01", "2024-06-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_returns_none_when_empty(self, ingester):
        with patch.object(ingester, "_download", return_value=pd.DataFrame()):
            result = ingester.fetch_single("SPY", "2024-01-01", "2024-06-01")
        assert result is None

    def test_returns_none_after_all_retries_fail(self, ingester):
        with patch.object(ingester, "_download", side_effect=Exception("Network error")):
            result = ingester.fetch_single("SPY", "2024-01-01", "2024-06-01")
        assert result is None

    def test_retries_on_failure_then_succeeds(self, ingester, sample_raw_df):
        """Falla en primer intento, éxito en segundo."""
        call_count = 0
        def flaky_download(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Timeout")
            return sample_raw_df

        with patch.object(ingester, "_download", side_effect=flaky_download):
            result = ingester.fetch_single("SPY", "2024-01-01", "2024-06-01")

        assert result is not None
        assert call_count == 2


# ── tests de fetch_multiple ───────────────────────────────────────────

class TestFetchMultiple:

    def test_returns_successful_symbols_only(self, ingester, sample_raw_df):
        def mock_fetch_single(symbol, *args, **kwargs):
            if symbol == "INVALID":
                return None
            return sample_raw_df

        with patch.object(ingester, "fetch_single", side_effect=mock_fetch_single):
            result = ingester.fetch_multiple(
                ["SPY", "INVALID", "QQQ"],
                "2024-01-01",
                "2024-06-01",
            )

        assert "SPY" in result
        assert "QQQ" in result
        assert "INVALID" not in result
        assert len(result) == 2

    def test_empty_list_returns_empty_dict(self, ingester):
        result = ingester.fetch_multiple([], "2024-01-01", "2024-06-01")
        assert result == {}