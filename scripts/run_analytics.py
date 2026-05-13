"""
Demuestra el módulo de analytics sobre datos reales de PostgreSQL.

Uso:
    python scripts/run_analytics.py
    python scripts/run_analytics.py --symbol SPY --start 2022-01-01
"""
import sys
import argparse
import logging

sys.path.insert(0, ".")

from src.storage.database       import DatabaseManager
from src.analytics.returns      import ReturnsCalculator
from src.analytics.volatility   import VolatilityCalculator
from src.analytics.risk_metrics import RiskMetricsCalculator
from src.analytics.correlation  import CorrelationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="SPY")
    parser.add_argument("--start",    default="2022-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    return parser.parse_args()


def main():
    args = parse_args()
    db   = DatabaseManager()

    # ── Cargar datos desde PostgreSQL ────────────────────────────────
    print("\n" + "="*60)
    print(f"ANALYTICS - {args.symbol}  ({args.start} → {args.end})")
    print("="*60)

    df = db.get_prices(args.symbol, args.start, args.end)
    if df.empty:
        print(f"Sin datos para {args.symbol}. Ejecuta run_pipeline.py primero.")
        return

    prices  = df["close"]
    returns = prices.pct_change().dropna()
    print(f"\nDatos: {len(df)} filas  "
          f"({df.index.min().date()} → {df.index.max().date()})")

    # ── Retornos ─────────────────────────────────────────────────────
    print("\n[1] MÉTRICAS DE RETORNO")
    ret_calc = ReturnsCalculator()
    ret_sum  = ret_calc.summary(prices)
    for k, v in ret_sum.items():
        print(f"  {k:<30} {v}")

    # ── Volatilidad ───────────────────────────────────────────────────
    print("\n[2] MÉTRICAS DE VOLATILIDAD")
    vol_calc = VolatilityCalculator()
    vol_sum  = vol_calc.summary(prices, df.get("high"), df.get("low"))
    for k, v in vol_sum.items():
        print(f"  {k:<30} {v}%")

    regime_df = vol_calc.regime(prices)
    regime_counts = regime_df["regime"].value_counts()
    print(f"\n  Régimen de volatilidad (distribución):")
    for regime, count in regime_counts.items():
        pct = count / len(regime_df.dropna()) * 100
        print(f"    {regime:<8} {count:>4} días  ({pct:.1f}%)")
    print(f"  Régimen actual: "
          f"{regime_df['regime'].dropna().iloc[-1]}")

    # ── Risk Metrics ──────────────────────────────────────────────────
    print("\n[3] MÉTRICAS DE RIESGO")
    risk_calc = RiskMetricsCalculator(risk_free_rate=0.04)
    risk_sum  = risk_calc.full_summary(prices, returns)
    for k, v in risk_sum.items():
        print(f"  {k:<30} {v}")

    # ── Correlaciones multi-activo ────────────────────────────────────
    print("\n[4] CORRELACIONES (todos los activos en DB)")
    symbols  = db.get_available_symbols()
    datasets = {}
    for sym in symbols:
        df_sym = db.get_prices(sym, args.start, args.end)
        if not df_sym.empty:
            datasets[sym] = df_sym

    if len(datasets) >= 2:
        analyzer   = CorrelationAnalyzer()
        returns_df = analyzer.prepare_returns_matrix(datasets)
        matrix     = analyzer.static_matrix(returns_df.dropna())
        print(f"\n  Matriz de correlación ({args.start} → {args.end}):")
        print(matrix.round(3).to_string())

    print("\n" + "="*60)
    print("Analytics completado")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()