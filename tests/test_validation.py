"""
Tests unitarios para DataQualityChecker.

Ejecutar: pytest tests/test_validation.py -v
"""
import pytest
import pandas as pd
import numpy as np
from src.validation.data_quality import DataQualityChecker, QualityReport


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def checker():
    return DataQualityChecker(
        outlier_threshold=0.15,
        min_quality_score=70.0,
        max_gap_days=5,
    )


@pytest.fixture
def clean_df():
    """DataFrame OHLCV limpio sin problemas."""
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    base  = 100.0
    data  = {
        "symbol": "TEST",
        "open":   [base + i       for i in range(10)],
        "high":   [base + i + 1.0 for i in range(10)],
        "low":    [base + i - 1.0 for i in range(10)],
        "close":  [base + i + 0.5 for i in range(10)],
        "volume": [1_000_000] * 10,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def df_with_nulls(clean_df):
    """DataFrame con 2 filas nulas en close."""
    df = clean_df.copy()
    df.loc[df.index[2], "close"] = np.nan
    df.loc[df.index[5], "open"]  = np.nan
    return df


@pytest.fixture
def df_with_ohlc_violations(clean_df):
    """DataFrame con violación high < low."""
    df = clean_df.copy()
    df.loc[df.index[3], "high"] = 50.0   # high < low
    df.loc[df.index[3], "low"]  = 200.0
    return df


@pytest.fixture
def df_with_duplicates(clean_df):
    """DataFrame con una fecha duplicada."""
    return pd.concat([clean_df, clean_df.iloc[[0]]])


@pytest.fixture
def df_with_outlier(clean_df):
    """
    DataFrame con retorno diario extremo (> 15%) pero OHLC válido.

    El outlier debe tener high >= close para no ser eliminado
    por _check_ohlc_integrity antes de llegar a _check_outliers.
    """
    df = clean_df.copy()
    # Precio anterior (índice 3): close = 103.5
    # Outlier en índice 4: subida del 20% → close = 124.2
    outlier_close = df.loc[df.index[3], "close"] * 1.20  # 124.2
    df.loc[df.index[4], "open"]  = 104.0
    df.loc[df.index[4], "high"]  = round(outlier_close + 1.0, 2)  # 125.2
    df.loc[df.index[4], "low"]   = 103.0
    df.loc[df.index[4], "close"] = round(outlier_close, 2)         # 124.2
    return df

# ── tests sobre datos limpios ─────────────────────────────────────────

class TestCleanData:

    def test_clean_data_returns_same_rows(self, checker, clean_df):
        result, report = checker.validate(clean_df, "TEST")
        assert len(result) == len(clean_df)

    def test_clean_data_score_is_100(self, checker, clean_df):
        _, report = checker.validate(clean_df, "TEST")
        assert report.quality_score == 100.0

    def test_clean_data_is_usable(self, checker, clean_df):
        _, report = checker.validate(clean_df, "TEST")
        assert report.is_usable is True

    def test_clean_data_no_issues(self, checker, clean_df):
        _, report = checker.validate(clean_df, "TEST")
        assert len(report.issues) == 0


# ── tests de detección de nulos ───────────────────────────────────────

class TestNullDetection:

    def test_null_rows_detected(self, checker, df_with_nulls):
        _, report = checker.validate(df_with_nulls, "TEST")
        assert report.null_rows == 2

    def test_null_rows_removed(self, checker, df_with_nulls):
        result, _ = checker.validate(df_with_nulls, "TEST")
        assert result.isnull().sum().sum() == 0

    def test_null_penalizes_score(self, checker, df_with_nulls):
        _, report = checker.validate(df_with_nulls, "TEST")
        assert report.quality_score < 100.0

    def test_null_adds_issue_message(self, checker, df_with_nulls):
        _, report = checker.validate(df_with_nulls, "TEST")
        assert any("nulos" in issue for issue in report.issues)


# ── tests de integridad OHLC ──────────────────────────────────────────

class TestOHLCIntegrity:

    def test_violation_detected(self, checker, df_with_ohlc_violations):
        _, report = checker.validate(df_with_ohlc_violations, "TEST")
        assert report.ohlc_violations >= 1

    def test_violation_row_removed(self, checker, df_with_ohlc_violations):
        result, _ = checker.validate(df_with_ohlc_violations, "TEST")
        # La fila violación debe haber sido eliminada
        assert len(result) < len(df_with_ohlc_violations)

    def test_violation_penalizes_score(self, checker, df_with_ohlc_violations):
        _, report = checker.validate(df_with_ohlc_violations, "TEST")
        assert report.quality_score < 100.0


# ── tests de duplicados ───────────────────────────────────────────────

class TestDuplicates:

    def test_duplicates_detected(self, checker, df_with_duplicates):
        _, report = checker.validate(df_with_duplicates, "TEST")
        assert report.duplicate_rows >= 1

    def test_duplicates_removed(self, checker, df_with_duplicates):
        result, _ = checker.validate(df_with_duplicates, "TEST")
        assert result.index.duplicated().sum() == 0


# ── tests de outliers ─────────────────────────────────────────────────

class TestOutliers:

    def test_outlier_detected(self, checker, df_with_outlier):
        _, report = checker.validate(df_with_outlier, "TEST")
        assert report.outlier_rows >= 1

    def test_outlier_not_removed(self, checker, df_with_outlier):
        """Outliers se reportan pero no se eliminan automáticamente."""
        result, _ = checker.validate(df_with_outlier, "TEST")
        assert len(result) == len(df_with_outlier)

    def test_outlier_adds_issue_message(self, checker, df_with_outlier):
        _, report = checker.validate(df_with_outlier, "TEST")
        assert any("retornos" in issue for issue in report.issues)


# ── tests de datos vacíos ─────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_dataframe(self, checker):
        result, report = checker.validate(pd.DataFrame(), "TEST")
        assert result.empty
        assert report.is_usable is False

    def test_none_input(self, checker):
        result, report = checker.validate(None, "TEST")
        assert result.empty
        assert report.is_usable is False

    def test_single_row(self, checker, clean_df):
        """Un solo dato no debe crashear."""
        single = clean_df.iloc[[0]]
        result, report = checker.validate(single, "TEST")
        assert len(result) == 1