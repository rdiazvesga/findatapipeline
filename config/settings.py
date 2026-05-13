"""
Configuración centralizada del proyecto.
Lee variables de entorno desde archivo .env
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Ruta raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent

# Cargar variables de entorno
load_dotenv(ROOT_DIR / ".env")

# ── Base de datos ──────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "findatapipeline"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ── Ingesta de datos ───────────────────────────────────────
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_INTERVAL   = "1d"
MAX_RETRIES        = 3
REQUEST_TIMEOUT    = 30

# Universo de activos por defecto
DEFAULT_SYMBOLS = {
    "indices":    ["SPY", "QQQ", "IWM", "EEM"],
    "sectors":    ["XLF", "XLE", "XLK", "XLV"],
    "fx":         ["EURUSD=X", "JPYUSD=X", "GBPUSD=X"],
    "commodities":["GC=F", "CL=F"],
    "crypto":     ["BTC-USD", "ETH-USD"],
    "colombia":   ["CIB", "GMEXICOB.MX"],
}

# ── Analytics ──────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE        = 0.04   # 4% anual referencia
VAR_CONFIDENCE_LEVEL  = 0.95

# ── Dashboard ─────────────────────────────────────────────
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "localhost")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "True") == "True"