"""
Script de ingesta de datos.
Descarga datos históricos y los muestra en consola.

Uso:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --symbols SPY QQQ GC=F --start 2022-01-01
"""
import sys
import argparse
import logging

# Para imports relativos funcionen desde scripts/
sys.path.insert(0, ".")

from src.ingestion.yahoo_finance import YahooFinanceIngester

# Configurar logging visible en consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingestar datos financieros")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["SPY", "QQQ", "EURUSD=X", "GC=F", "BTC-USD"],
        help="Lista de tickers a descargar"
    )
    parser.add_argument(
        "--start", default="2022-01-01",
        help="Fecha inicio YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="Fecha fin YYYY-MM-DD"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("INGESTA DE DATOS - FindDataPipeline")
    print("="*60)
    print(f"Símbolos:  {args.symbols}")
    print(f"Período:   {args.start} → {args.end}")
    print("="*60 + "\n")

    ingester = YahooFinanceIngester()
    datasets = ingester.fetch_multiple(args.symbols, args.start, args.end)

    print("\n" + "="*60)
    print("RESUMEN DE DESCARGA")
    print("="*60)

    total_rows = 0
    for symbol, df in datasets.items():
        rows = len(df)
        total_rows += rows
        date_min = df.index.min().date()
        date_max = df.index.max().date()
        close_last = df["close"].iloc[-1]
        print(
            f"  {symbol:<12} {rows:>5} filas  "
            f"{date_min} → {date_max}  "
            f"último close: {close_last:>10.4f}"
        )

    failed = set(args.symbols) - set(datasets.keys())
    if failed:
        print(f"\n  FALLIDOS: {list(failed)}")

    print(f"\n  TOTAL: {total_rows:,} filas en {len(datasets)} activos")
    print("="*60 + "\n")

    # Mostrar muestra del primer dataset
    if datasets:
        first_symbol = list(datasets.keys())[0]
        df = datasets[first_symbol]
        print(f"Muestra de {first_symbol} (últimas 5 filas):")
        print(df.tail())
        print(f"\nTipos de datos:")
        print(df.dtypes)


if __name__ == "__main__":
    main()