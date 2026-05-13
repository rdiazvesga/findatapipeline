"""
Script de verificación del entorno completo.
Ejecutar: python scripts/verify_setup.py
"""
import sys


def check_imports():
    """Verifica que todas las dependencias están instaladas."""
    required = [
        "yfinance", "pandas", "numpy", "sqlalchemy",
        "psycopg2", "dotenv", "dash", "plotly", "pytest"
    ]
    failed = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NO INSTALADO")
            failed.append(package)
    return len(failed) == 0


def check_database():
    """Verifica conexión a PostgreSQL."""
    try:
        import sys
        sys.path.insert(0, ".")
        from config.settings import DB_URL
        from sqlalchemy import create_engine, text

        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM assets"))
            count = result.scalar()
            print(f"  ✓ PostgreSQL conectado - {count} activos en DB")
        return True
    except Exception as e:
        print(f"  ✗ PostgreSQL - ERROR: {e}")
        return False


def check_yahoo_finance():
    """Verifica descarga de datos desde Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="5d")
        if df.empty:
            print("  ✗ Yahoo Finance - Sin datos")
            return False
        print(f"  ✓ Yahoo Finance - {len(df)} filas descargadas para SPY")
        return True
    except Exception as e:
        print(f"  ✗ Yahoo Finance - ERROR: {e}")
        return False


def main():
    print("\n" + "="*50)
    print("VERIFICACIÓN DEL ENTORNO - FindDataPipeline")
    print("="*50)

    print("\n[1] Dependencias Python:")
    deps_ok = check_imports()

    print("\n[2] Conexión PostgreSQL:")
    db_ok = check_database()

    print("\n[3] API Yahoo Finance:")
    api_ok = check_yahoo_finance()

    print("\n" + "="*50)
    if deps_ok and db_ok and api_ok:
        print("✓ ENTORNO LISTO - Puedes empezar a desarrollar")
    else:
        print("✗ HAY PROBLEMAS - Revisar los errores arriba")
    print("="*50 + "\n")

    return 0 if (deps_ok and db_ok and api_ok) else 1


if __name__ == "__main__":
    sys.exit(main())