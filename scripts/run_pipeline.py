"""
Pipeline completo: ingesta → validación → almacenamiento.

Uso:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --symbols SPY QQQ --start 2023-01-01
"""
import sys
import argparse
import logging

sys.path.insert(0, ".")

from src.ingestion.yahoo_finance  import YahooFinanceIngester
from src.validation.data_quality  import DataQualityChecker
from src.storage.database         import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline completo de datos")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["SPY", "QQQ", "EURUSD=X", "GC=F", "BTC-USD"],
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("PIPELINE COMPLETO - FindDataPipeline")
    print("="*60)
    print(f"Símbolos : {args.symbols}")
    print(f"Período  : {args.start} → {args.end}")
    print("="*60)

    # ── Etapa 1: Ingesta ─────────────────────────────────────
    print("\n[1/3] Ingesta de datos...")
    ingester  = YahooFinanceIngester()
    raw_data  = ingester.fetch_multiple(args.symbols, args.start, args.end)
    print(f"      Descargados: {len(raw_data)}/{len(args.symbols)} activos")

    # ── Etapa 2: Validación ───────────────────────────────────
    print("\n[2/3] Validación de calidad...")
    checker       = DataQualityChecker()
    clean_data, reports = checker.validate_multiple(raw_data)

    print("\n      Reporte de calidad:")
    for symbol, report in reports.items():
        print(f"      {report.summary()}")
        if report.issues:
            for issue in report.issues:
                print(f"        ↳ {issue}")

    unusable = [s for s, r in reports.items() if not r.is_usable]
    if unusable:
        print(f"\n      ⚠ Descartados por baja calidad: {unusable}")

    # ── Etapa 3: Almacenamiento ───────────────────────────────
    print("\n[3/3] Almacenando en PostgreSQL...")
    db      = DatabaseManager()
    results = db.upsert_multiple(clean_data)

    for symbol, n_rows in results.items():
        db.save_quality_report(symbol, reports[symbol])
        print(f"      {symbol:<12} {n_rows:>5} filas guardadas")

    # ── Resumen final ─────────────────────────────────────────
    total_stored = sum(results.values())
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Activos procesados  : {len(raw_data)}")
    print(f"  Activos usables     : {len(clean_data)}")
    print(f"  Filas en PostgreSQL : {total_stored:,}")
    print(f"  Total en DB         : {db.get_row_count():,} filas")

    print("\n  Activos disponibles en DB:")
    db.get_available_symbols()

    print("\n  Verificación de recuperación (SPY, últimas 3 filas):")
    df_check = db.get_prices("SPY", start_date="2024-12-01")
    if not df_check.empty:
        print(df_check.tail(3)[["open","high","low","close","volume"]])

    print("\n✓ Pipeline completado exitosamente")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()