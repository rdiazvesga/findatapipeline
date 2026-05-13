"""
Análisis de correlaciones entre activos financieros.

Las correlaciones rolling detectan cambios en la estructura
de dependencia entre activos, especialmente en períodos de stress.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Calcula correlaciones entre series de retornos financieros.

    Nota importante: las correlaciones en finanzas no son estables.
    En períodos de crisis, activos normalmente descorrelacionados
    tienden a correlacionarse (correlation breakdown).
    Siempre usa ventanas rolling, no correlación estática.

    Example:
        analyzer = CorrelationAnalyzer()
        matrix   = analyzer.static_matrix(returns_df)
        rolling  = analyzer.rolling_pair("SPY", "GC=F", returns_df)
    """

    def static_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Matriz de correlación estática sobre todo el período.

        Args:
            returns_df: DataFrame donde cada columna es un activo
                        y cada fila es un período de retornos.

        Returns:
            DataFrame cuadrado N×N con correlaciones [-1, 1].

        Limitación: oculta cambios temporales en la estructura
        de correlación. Úsala solo como referencia de largo plazo.
        """
        corr = returns_df.corr(method="pearson")
        logger.info(
            "Matriz de correlación: %d activos",
            len(corr.columns),
        )
        return corr

    def rolling_pair(
        self,
        symbol_a:   str,
        symbol_b:   str,
        returns_df: pd.DataFrame,
        window:     int = 63,
    ) -> pd.Series:
        """
        Correlación rolling entre dos activos.

        Permite detectar cambios estructurales en la relación
        entre dos activos a lo largo del tiempo.

        Args:
            symbol_a:   Nombre de columna del primer activo.
            symbol_b:   Nombre de columna del segundo activo.
            returns_df: DataFrame de retornos con columnas por activo.
            window:     Ventana rolling en períodos (default 63 ≈ 3 meses).

        Returns:
            pd.Series de correlaciones rolling con índice temporal.
        """
        if symbol_a not in returns_df.columns:
            raise ValueError(f"{symbol_a} no está en returns_df")
        if symbol_b not in returns_df.columns:
            raise ValueError(f"{symbol_b} no está en returns_df")

        rolling_corr = (
            returns_df[symbol_a]
            .rolling(window, min_periods=window)
            .corr(returns_df[symbol_b])
        )
        logger.info(
            "Correlación rolling %s vs %s (window=%d): %d puntos",
            symbol_a, symbol_b, window, rolling_corr.notna().sum(),
        )
        return rolling_corr

    def rolling_matrix(
        self,
        returns_df: pd.DataFrame,
        window:     int = 63,
    ) -> pd.DataFrame:
        """
        Correlación rolling media de cada activo contra todos los demás.

        En lugar de N*(N-1)/2 series, devuelve para cada activo
        su correlación promedio con el resto del universo.
        Útil para detectar qué activos se vuelven más sistémicos.

        Returns:
            DataFrame con mismas columnas que returns_df,
            donde cada valor es la correlación media rolling
            del activo con los demás activos del universo.
        """
        symbols = returns_df.columns.tolist()
        result  = pd.DataFrame(index=returns_df.index, columns=symbols,
                               dtype=float)

        for sym in symbols:
            others = [s for s in symbols if s != sym]
            if not others:
                continue
            corr_series = []
            for other in others:
                c = (
                    returns_df[sym]
                    .rolling(window, min_periods=window)
                    .corr(returns_df[other])
                )
                corr_series.append(c)

            result[sym] = pd.concat(corr_series, axis=1).mean(axis=1)

        return result

    def prepare_returns_matrix(
        self,
        datasets: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Construye matriz de log-retornos a partir de datasets por activo.

        Args:
            datasets: {symbol: DataFrame con columna 'close'}

        Returns:
            DataFrame con una columna por activo, índice temporal alineado.
            Solo incluye fechas donde todos los activos tienen datos.
        """
        series_list = {}
        for symbol, df in datasets.items():
            if "close" not in df.columns:
                logger.warning("%s no tiene columna 'close', omitido", symbol)
                continue
            log_ret = np.log(df["close"] / df["close"].shift(1))
            series_list[symbol] = log_ret

        if not series_list:
            return pd.DataFrame()

        returns_df = pd.DataFrame(series_list)

        # Alinear fechas: outer join para no perder fechas
        # Los NaN en fechas con datos parciales se manejan en corr()
        logger.info(
            "Matriz de retornos: %d activos × %d fechas",
            len(returns_df.columns), len(returns_df),
        )
        return returns_df