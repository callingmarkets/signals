"""
CallingMarkets Signal Engine
Matches CallingMarkets Indicator 2 (Pine Script v5) exactly.
Signal = BUY if 2 or more of: EMA20 > EMA55, RSI14 > RSI_EMA14, MACD > Signal
Price: uses latest quote (bid/ask midpoint), falls back to last daily close.
"""

import json
import os
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

# ── API KEYS ───────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

STOCKS_URL       = "https://data.alpaca.markets/v2/stocks/bars"
CRYPTO_URL       = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
STOCKS_QUOTE_URL = "https://data.alpaca.markets/v2/stocks/quotes/latest"
CRYPTO_TRADE_URL = "https://data.alpaca.markets/v1beta3/crypto/us/latest/trades"

# ── TICKERS ────────────────────────────────────────────────────────────────────
TICKERS = [

    # ═══════════════════════════════════════════════════════════════════
    # BROAD MARKET
    # Phase ETF: SPY
    # ═══════════════════════════════════════════════════════════════════
    ("SPY",  "Broad Market"),
    ("QQQ",  "Broad Market"),
    ("IWM",  "Broad Market"),
    ("DIA",  "Broad Market"),
    ("MDY",  "Broad Market"),
    ("VTI",  "Broad Market"),
    ("VOO",  "Broad Market"),
    ("RSP",  "Broad Market"),
    ("SPLG", "Broad Market"),
    ("SCHB", "Broad Market"),
    ("ITOT", "Broad Market"),

    # ═══════════════════════════════════════════════════════════════════
    # TIER 1 — GICS CORE SECTORS
    # ═══════════════════════════════════════════════════════════════════

    # ── TECHNOLOGY  Phase ETF: XLK ──────────────────────────────────
    ("XLK",  "Technology"),
    ("AAPL",  "Technology"),
    ("MSFT",  "Technology"),
    ("NVDA",  "Technology"),
    ("AVGO",  "Technology"),
    ("ORCL",  "Technology"),
    ("CRM",   "Technology"),
    ("AMD",   "Technology"),
    ("INTC",  "Technology"),
    ("QCOM",  "Technology"),
    ("TXN",   "Technology"),
    ("MU",    "Technology"),
    ("AMAT",  "Technology"),
    ("LRCX",  "Technology"),
    ("KLAC",  "Technology"),
    ("MRVL",  "Technology"),
    ("ADBE",  "Technology"),
    ("NOW",   "Technology"),
    ("INTU",  "Technology"),
    ("WDAY",  "Technology"),
    ("SNOW",  "Technology"),
    ("PLTR",  "Technology"),
    ("CRWD",  "Technology"),
    ("PANW",  "Technology"),
    ("ZS",    "Technology"),
    ("NET",   "Technology"),
    ("DDOG",  "Technology"),
    ("MDB",   "Technology"),
    ("SHOP",  "Technology"),
    ("UBER",  "Technology"),
    ("HPQ",   "Technology"),
    ("DELL",  "Technology"),

    # ── COMMUNICATIONS  Phase ETF: XLC ──────────────────────────────
    ("XLC",   "Communications"),
    ("GOOGL", "Communications"),
    ("META",  "Communications"),
    ("DIS",   "Communications"),
    ("NFLX",  "Communications"),
    ("SPOT",  "Communications"),
    ("APP",   "Communications"),
    ("RBLX",  "Communications"),
    ("U",     "Communications"),
    ("T",     "Communications"),
    ("VZ",    "Communications"),
    ("CMCSA", "Communications"),
    ("CHTR",  "Communications"),
    ("TMUS",  "Communications"),
    ("EA",    "Communications"),
    ("TTWO",  "Communications"),
    ("PARA",  "Communications"),
    ("WBD",   "Communications"),
    ("FOXA",  "Communications"),
    ("LBRDA", "Communications"),

    # ── CONSUMER DISCRETIONARY  Phase ETF: XLY ──────────────────────
    ("XLY",   "Consumer Discretionary"),
    ("AMZN",  "Consumer Discretionary"),
    ("TSLA",  "Consumer Discretionary"),
    ("HD",    "Consumer Discretionary"),
    ("LOW",   "Consumer Discretionary"),
    ("MCD",   "Consumer Discretionary"),
    ("SBUX",  "Consumer Discretionary"),
    ("NKE",   "Consumer Discretionary"),
    ("LULU",  "Consumer Discretionary"),
    ("TJX",   "Consumer Discretionary"),
    ("BKNG",  "Consumer Discretionary"),
    ("MAR",   "Consumer Discretionary"),
    ("HLT",   "Consumer Discretionary"),
    ("TGT",   "Consumer Discretionary"),
    ("ABNB",  "Consumer Discretionary"),
    ("LYFT",  "Consumer Discretionary"),
    ("RCL",   "Consumer Discretionary"),
    ("CCL",   "Consumer Discretionary"),
    ("F",     "Consumer Discretionary"),
    ("GM",    "Consumer Discretionary"),

    # ── CONSUMER STAPLES  Phase ETF: XLP ────────────────────────────
    ("XLP",   "Consumer Staples"),
    ("COST",  "Consumer Staples"),
    ("WMT",   "Consumer Staples"),
    ("PG",    "Consumer Staples"),
    ("KO",    "Consumer Staples"),
    ("PEP",   "Consumer Staples"),
    ("PM",    "Consumer Staples"),
    ("MO",    "Consumer Staples"),
    ("CL",    "Consumer Staples"),
    ("GIS",   "Consumer Staples"),
    ("K",     "Consumer Staples"),
    ("MDLZ",  "Consumer Staples"),
    ("HSY",   "Consumer Staples"),
    ("TSN",   "Consumer Staples"),
    ("SJM",   "Consumer Staples"),
    ("CAG",   "Consumer Staples"),

    # ── FINANCIALS  Phase ETF: XLF ───────────────────────────────────
    ("XLF",   "Financials"),
    ("JPM",   "Financials"),
    ("BAC",   "Financials"),
    ("GS",    "Financials"),
    ("MS",    "Financials"),
    ("WFC",   "Financials"),
    ("C",     "Financials"),
    ("BX",    "Financials"),
    ("KKR",   "Financials"),
    ("APO",   "Financials"),
    ("BLK",   "Financials"),
    ("SCHW",  "Financials"),
    ("V",     "Financials"),
    ("MA",    "Financials"),
    ("AXP",   "Financials"),
    ("PYPL",  "Financials"),
    ("SQ",    "Financials"),
    ("NU",    "Financials"),
    ("SOFI",  "Financials"),
    ("COF",   "Financials"),
    ("USB",   "Financials"),
    ("PNC",   "Financials"),
    ("TFC",   "Financials"),
    ("AIG",   "Financials"),
    ("MET",   "Financials"),

    # ── HEALTH CARE  Phase ETF: XLV ──────────────────────────────────
    ("XLV",   "Health Care"),
    ("UNH",   "Health Care"),
    ("JNJ",   "Health Care"),
    ("LLY",   "Health Care"),
    ("ABBV",  "Health Care"),
    ("MRK",   "Health Care"),
    ("PFE",   "Health Care"),
    ("AMGN",  "Health Care"),
    ("GILD",  "Health Care"),
    ("BIIB",  "Health Care"),
    ("REGN",  "Health Care"),
    ("VRTX",  "Health Care"),
    ("ISRG",  "Health Care"),
    ("BSX",   "Health Care"),
    ("MDT",   "Health Care"),
    ("CVS",   "Health Care"),
    ("HUM",   "Health Care"),
    ("ELV",   "Health Care"),
    ("CI",    "Health Care"),
    ("ZTS",   "Health Care"),
    ("DXCM",  "Health Care"),
    ("IDXX",  "Health Care"),

    # ── ENERGY  Phase ETF: XLE ───────────────────────────────────────
    ("XLE",   "Energy"),
    ("XOM",   "Energy"),
    ("CVX",   "Energy"),
    ("COP",   "Energy"),
    ("EOG",   "Energy"),
    ("SLB",   "Energy"),
    ("MPC",   "Energy"),
    ("PSX",   "Energy"),
    ("VLO",   "Energy"),
    ("HAL",   "Energy"),
    ("DVN",   "Energy"),
    ("OXY",   "Energy"),
    ("HES",   "Energy"),
    ("BKR",   "Energy"),
    ("FANG",  "Energy"),
    ("MRO",   "Energy"),
    ("APA",   "Energy"),

    # ── INDUSTRIALS  Phase ETF: XLI ──────────────────────────────────
    ("XLI",   "Industrials"),
    ("CAT",   "Industrials"),
    ("DE",    "Industrials"),
    ("RTX",   "Industrials"),
    ("LMT",   "Industrials"),
    ("NOC",   "Industrials"),
    ("GE",    "Industrials"),
    ("HON",   "Industrials"),
    ("BA",    "Industrials"),
    ("UNP",   "Industrials"),
    ("FDX",   "Industrials"),
    ("UPS",   "Industrials"),
    ("WM",    "Industrials"),
    ("MMM",   "Industrials"),
    ("EMR",   "Industrials"),
    ("ETN",   "Industrials"),
    ("PH",    "Industrials"),
    ("ROK",   "Industrials"),
    ("CMI",   "Industrials"),
    ("PCAR",  "Industrials"),

    # ── MATERIALS  Phase ETF: XLB ────────────────────────────────────
    ("XLB",   "Materials"),
    ("NEM",   "Materials"),
    ("FCX",   "Materials"),
    ("AA",    "Materials"),
    ("NUE",   "Materials"),
    ("LIN",   "Materials"),
    ("APD",   "Materials"),
    ("ECL",   "Materials"),
    ("DD",    "Materials"),
    ("DOW",   "Materials"),
    ("PPG",   "Materials"),
    ("VMC",   "Materials"),
    ("MLM",   "Materials"),
    ("IP",    "Materials"),
    ("PKG",   "Materials"),
    ("ALB",   "Materials"),

    # ── UTILITIES  Phase ETF: XLU ────────────────────────────────────
    ("XLU",   "Utilities"),
    ("NEE",   "Utilities"),
    ("DUK",   "Utilities"),
    ("SO",    "Utilities"),
    ("D",     "Utilities"),
    ("AEP",   "Utilities"),
    ("EXC",   "Utilities"),
    ("SRE",   "Utilities"),
    ("PCG",   "Utilities"),
    ("ES",    "Utilities"),
    ("AWK",   "Utilities"),
    ("ETR",   "Utilities"),
    ("FE",    "Utilities"),
    ("PPL",   "Utilities"),
    ("NI",    "Utilities"),
    ("CMS",   "Utilities"),

    # ── REAL ESTATE  Phase ETF: XLRE ─────────────────────────────────
    ("XLRE",  "Real Estate"),
    ("PLD",   "Real Estate"),
    ("AMT",   "Real Estate"),
    ("EQIX",  "Real Estate"),
    ("SPG",   "Real Estate"),
    ("O",     "Real Estate"),
    ("WELL",  "Real Estate"),
    ("PSA",   "Real Estate"),
    ("VTR",   "Real Estate"),
    ("EQR",   "Real Estate"),
    ("AVB",   "Real Estate"),
    ("DLR",   "Real Estate"),
    ("CCI",   "Real Estate"),
    ("SBAC",  "Real Estate"),
    ("WY",    "Real Estate"),
    ("HST",   "Real Estate"),

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2 — SUB-SECTORS (independent universes, duplication intentional)
    # ═══════════════════════════════════════════════════════════════════

    # ── SEMICONDUCTORS  Phase ETF: SMH ───────────────────────────────
    ("SMH",   "Semiconductors"),
    ("SOXX",  "Semiconductors"),
    ("NVDA",  "Semiconductors"),
    ("AMD",   "Semiconductors"),
    ("INTC",  "Semiconductors"),
    ("QCOM",  "Semiconductors"),
    ("MU",    "Semiconductors"),
    ("AVGO",  "Semiconductors"),
    ("AMAT",  "Semiconductors"),
    ("LRCX",  "Semiconductors"),
    ("KLAC",  "Semiconductors"),
    ("MRVL",  "Semiconductors"),
    ("ASML",  "Semiconductors"),
    ("TSM",   "Semiconductors"),
    ("MCHP",  "Semiconductors"),
    ("ON",    "Semiconductors"),
    ("MPWR",  "Semiconductors"),
    ("ENTG",  "Semiconductors"),
    ("ONTO",  "Semiconductors"),
    ("TER",   "Semiconductors"),

    # ── BIOTECH  Phase ETF: XBI ──────────────────────────────────────
    ("XBI",   "Biotech"),
    ("ARKG",  "Biotech"),
    ("MRNA",  "Biotech"),
    ("BNTX",  "Biotech"),
    ("REGN",  "Biotech"),
    ("VRTX",  "Biotech"),
    ("BIIB",  "Biotech"),
    ("AMGN",  "Biotech"),
    ("GILD",  "Biotech"),
    ("ILMN",  "Biotech"),
    ("EXAS",  "Biotech"),
    ("BMRN",  "Biotech"),
    ("ALNY",  "Biotech"),
    ("INCY",  "Biotech"),
    ("RARE",  "Biotech"),

    # ── CYBERSECURITY  Phase ETF: CIBR ───────────────────────────────
    ("CIBR",  "Cybersecurity"),
    ("HACK",  "Cybersecurity"),
    ("CRWD",  "Cybersecurity"),
    ("PANW",  "Cybersecurity"),
    ("ZS",    "Cybersecurity"),
    ("NET",   "Cybersecurity"),
    ("FTNT",  "Cybersecurity"),
    ("OKTA",  "Cybersecurity"),
    ("S",     "Cybersecurity"),
    ("TENB",  "Cybersecurity"),
    ("QLYS",  "Cybersecurity"),
    ("VRNS",  "Cybersecurity"),
    ("CSCO",  "Cybersecurity"),
    ("CHKP",  "Cybersecurity"),
    ("RPD",   "Cybersecurity"),

    # ── AI & ROBOTICS  Phase ETF: BOTZ ───────────────────────────────
    ("BOTZ",  "AI & Robotics"),
    ("ROBO",  "AI & Robotics"),
    ("NVDA",  "AI & Robotics"),
    ("MSFT",  "AI & Robotics"),
    ("GOOGL", "AI & Robotics"),
    ("META",  "AI & Robotics"),
    ("PATH",  "AI & Robotics"),
    ("AI",    "AI & Robotics"),
    ("BBAI",  "AI & Robotics"),
    ("SOUN",  "AI & Robotics"),
    ("CFLT",  "AI & Robotics"),
    ("GTLB",  "AI & Robotics"),
    ("AMBA",  "AI & Robotics"),
    ("TER",   "AI & Robotics"),
    ("ISRG",  "AI & Robotics"),
    ("PLTR",  "AI & Robotics"),
    ("QBTS",  "AI & Robotics"),  # quantum computing — next frontier of AI infrastructure

    # ── FINTECH  Phase ETF: FINX ─────────────────────────────────────
    ("FINX",  "Fintech"),
    ("V",     "Fintech"),
    ("MA",    "Fintech"),
    ("PYPL",  "Fintech"),
    ("SQ",    "Fintech"),
    ("NU",    "Fintech"),
    ("SOFI",  "Fintech"),
    ("AFRM",  "Fintech"),
    ("UPST",  "Fintech"),
    ("LC",    "Fintech"),
    ("WEX",   "Fintech"),
    ("FLYW",  "Fintech"),
    ("CWAN",  "Fintech"),
    ("COIN",  "Fintech"),
    ("HOOD",  "Fintech"),

    # ── CRYPTO  Phase ETF: IBIT ──────────────────────────────────────
    # ETF wrappers (TradFi access)
    ("IBIT",      "Crypto"),
    ("FBTC",      "Crypto"),
    ("ETHA",      "Crypto"),
    # Top 25 non-stablecoin crypto by market cap
    ("BTC/USD",   "Crypto"),   # 1  Bitcoin
    ("ETH/USD",   "Crypto"),   # 2  Ethereum
    ("SOL/USD",   "Crypto"),   # 4  Solana
    ("XRP/USD",   "Crypto"),   # 5  XRP
    ("ADA/USD",   "Crypto"),   # 6  Cardano
    ("AVAX/USD",  "Crypto"),   # 7  Avalanche
    ("DOGE/USD",  "Crypto"),   # 8  Dogecoin
    ("DOT/USD",   "Crypto"),   # 9  Polkadot
    ("MATIC/USD", "Crypto"),   # 10 Polygon
    ("LINK/USD",  "Crypto"),   # 12 Chainlink
    ("UNI/USD",   "Crypto"),   # 13 Uniswap
    ("LTC/USD",   "Crypto"),   # 14 Litecoin
    ("ALGO/USD",  "Crypto"),   # 17 Algorand
    ("NEAR/USD",  "Crypto"),   # 18 NEAR Protocol
    ("FIL/USD",   "Crypto"),   # 20 Filecoin
    ("ARB/USD",   "Crypto"),   # 22 Arbitrum
    ("AAVE/USD",  "Crypto"),   # 25 Aave

    # ── COMMODITIES  Phase ETF: PDBC ─────────────────────────────────
    ("PDBC",  "Commodities"),
    ("GLD",   "Commodities"),
    ("IAU",   "Commodities"),
    ("SLV",   "Commodities"),
    ("PPLT",  "Commodities"),
    ("USO",   "Commodities"),
    ("BNO",   "Commodities"),
    ("UNG",   "Commodities"),
    ("DBA",   "Commodities"),
    ("CORN",  "Commodities"),
    ("WEAT",  "Commodities"),
    ("SOYB",  "Commodities"),
    ("DJP",   "Commodities"),
    ("GDX",   "Commodities"),
    ("GDXJ",  "Commodities"),
    ("COPX",  "Commodities"),
    ("URA",   "Commodities"),

    # ── VOLATILITY  Phase ETF: VXX ───────────────────────────────────
    # VIX futures ETFs + CBOE (exchange that runs VIX) + related products
    ("VXX",   "Volatility"),
    ("UVXY",  "Volatility"),
    ("SVXY",  "Volatility"),
    ("VIXY",  "Volatility"),
    ("VIXM",  "Volatility"),
    ("CBOE",  "Volatility"),  # Chicago Board Options Exchange — the VIX company
    ("SPXS",  "Volatility"),  # 3x inverse S&P — hedging instrument
    ("SQQQ",  "Volatility"),  # 3x inverse Nasdaq — hedging instrument
    ("SH",    "Volatility"),  # simple inverse S&P
    ("PSQ",   "Volatility"),  # simple inverse Nasdaq
    ("TAIL",  "Volatility"),  # tail risk hedge ETF
    ("BTAL",  "Volatility"),  # anti-beta — moves opposite market

    # ── FIXED INCOME  Phase ETF: TLT ─────────────────────────────────
    ("TLT",   "Fixed Income"),
    ("IEF",   "Fixed Income"),
    ("SHY",   "Fixed Income"),
    ("BND",   "Fixed Income"),
    ("AGG",   "Fixed Income"),
    ("HYG",   "Fixed Income"),
    ("JNK",   "Fixed Income"),
    ("LQD",   "Fixed Income"),
    ("EMB",   "Fixed Income"),
    ("TIP",   "Fixed Income"),
    ("MUB",   "Fixed Income"),

    # ── INTERNATIONAL DEVELOPED  Phase ETF: EFA ──────────────────────
    ("EFA",   "International Developed"),
    ("VEA",   "International Developed"),
    ("EWJ",   "International Developed"),
    ("EWG",   "International Developed"),
    ("EWU",   "International Developed"),
    ("EWC",   "International Developed"),
    ("EWA",   "International Developed"),
    ("EWL",   "International Developed"),
    ("EWQ",   "International Developed"),
    ("HEZU",  "International Developed"),
    ("DXJ",   "International Developed"),

    # ── INTERNATIONAL EMERGING  Phase ETF: EEM ───────────────────────
    ("EEM",   "International Emerging"),
    ("VWO",   "International Emerging"),
    ("FXI",   "International Emerging"),
    ("MCHI",  "International Emerging"),
    ("EWZ",   "International Emerging"),
    ("INDA",  "International Emerging"),
    ("EWY",   "International Emerging"),
    ("EWT",   "International Emerging"),
    ("KWEB",  "International Emerging"),
    ("GXC",   "International Emerging"),
    ("INDY",  "International Emerging"),
]


# ── INDICATOR PARAMS ───────────────────────────────────────────────────────────
EMA_FAST   = 20
EMA_SLOW   = 55
RSI_LEN    = 14
RSI_MA_LEN = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9

START_DATES = {
    "1Day":   (datetime.utcnow() - timedelta(days=500)).strftime("%Y-%m-%d"),
    "1Week":  (datetime.utcnow() - timedelta(weeks=350)).strftime("%Y-%m-%d"),
    "1Month": (datetime.utcnow() - timedelta(days=365*30)).strftime("%Y-%m-%d"),
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

def fetch_latest_prices(symbols: list) -> dict:
    """
    Fetch latest price for all tickers.
    Stocks: latest quote (bid/ask midpoint).
    Crypto: latest trade price.
    """
    prices = {}
    stocks  = [s for s in symbols if "/" not in s]
    cryptos = [s for s in symbols if "/" in s]

    # Stocks — latest quote
    if stocks:
        try:
            r = requests.get(STOCKS_QUOTE_URL, headers=get_headers(),
                             params={"symbols": ",".join(stocks)}, timeout=10)
            r.raise_for_status()
            for sym, data in r.json().get("quotes", {}).items():
                bid = data.get("bp", 0)
                ask = data.get("ap", 0)
                if bid and ask:
                    prices[sym] = round((bid + ask) / 2, 2)
                elif ask:
                    prices[sym] = round(ask, 2)
                elif bid:
                    prices[sym] = round(bid, 2)
            print(f"  Stock quotes fetched: {list(prices.keys())}")
        except Exception as e:
            print(f"  WARNING stock quotes: {e}")

    # Crypto — latest trade price
    if cryptos:
        try:
            r = requests.get(CRYPTO_TRADE_URL, headers=get_headers(),
                             params={"symbols": ",".join(cryptos)}, timeout=10)
            r.raise_for_status()
            data = r.json()
            # Response format: {"trades": {"BTC/USD": {"p": 87000, ...}}}
            for sym, trade in data.get("trades", {}).items():
                price = trade.get("p")
                if price:
                    prices[sym] = round(float(price), 2)
            print(f"  Crypto trades fetched: {[s for s in cryptos if s in prices]}")
        except Exception as e:
            print(f"  WARNING crypto trades: {e}")

    return prices

def fetch_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    is_crypto = "/" in symbol
    url = CRYPTO_URL if is_crypto else STOCKS_URL
    params = {
        "symbols":   symbol,
        "timeframe": timeframe,
        "start":     START_DATES[timeframe],
        "limit":     1000,
        "sort":      "asc",
    }
    all_bars = []
    while True:
        r = requests.get(url, headers=get_headers(), params=params, timeout=15)
        r.raise_for_status()
        data     = r.json()
        bars_raw = data.get("bars", {}).get(symbol, [])
        all_bars.extend(bars_raw)
        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

# ── INDICATOR MATH ─────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def calc_rma(series: pd.Series, length: int) -> pd.Series:
    alpha  = 1.0 / length
    values = series.values.astype(float)
    result = np.full(len(values), np.nan)
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            result[i] = np.mean(window)
            for j in range(i + 1, len(values)):
                result[j] = alpha * values[j] + (1 - alpha) * result[j - 1]
            break
    return pd.Series(result, index=series.index)

def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = calc_rma(gain, length)
    avg_l = calc_rma(loss, length)
    rs    = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int, slow: int, sig: int):
    macd_line   = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line

# ── SIGNAL LOGIC ───────────────────────────────────────────────────────────────
def compute_signal(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < EMA_SLOW + 10:
        return {"signal": "N/A", "last_close": None}

    src                    = df["close"]
    ema20                  = calc_ema(src, EMA_FAST)
    ema55                  = calc_ema(src, EMA_SLOW)
    rsi14                  = calc_rsi(src, RSI_LEN)
    rsi_ma                 = calc_ema(rsi14, RSI_MA_LEN)
    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)

    bull1  = bool(ema20.iloc[-1]     > ema55.iloc[-1])
    bull2  = bool(rsi14.iloc[-1]     > rsi_ma.iloc[-1])
    bull3  = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    is_buy = (int(bull1) + int(bull2) + int(bull3)) >= 2

    return {
        "signal":     "BUY" if is_buy else "SELL",
        "last_close": round(float(src.iloc[-1]), 2),
    }

# ── DEBUG MODE ────────────────────────────────────────────────────────────────
def debug_ticker(symbol: str, timeframe: str = "1Month"):
    # Prints detailed indicator values for a ticker to compare with TradingView
    print(f"\n{'='*60}")
    print(f"DEBUG: {symbol} — {timeframe}")
    print(f"{'='*60}")

    df = fetch_bars(symbol, timeframe)
    if df.empty:
        print("ERROR: No bars returned")
        return

    print(f"Bars fetched: {len(df)}")
    print(f"Date range:   {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Last close:   {df['close'].iloc[-1]:.4f}")
    print(f"Prev close:   {df['close'].iloc[-2]:.4f}")

    src = df["close"]

    ema20 = calc_ema(src, EMA_FAST)
    ema55 = calc_ema(src, EMA_SLOW)
    print(f"\nEMA20:  {ema20.iloc[-1]:.4f}")
    print(f"EMA55:  {ema55.iloc[-1]:.4f}")
    print(f"bull1 (EMA20 > EMA55): {ema20.iloc[-1] > ema55.iloc[-1]}")

    rsi14  = calc_rsi(src, RSI_LEN)
    rsi_ma = calc_ema(rsi14, RSI_MA_LEN)
    print(f"\nRSI14:   {rsi14.iloc[-1]:.4f}")
    print(f"RSI EMA: {rsi_ma.iloc[-1]:.4f}")
    print(f"bull2 (RSI > RSI EMA): {rsi14.iloc[-1] > rsi_ma.iloc[-1]}")

    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)
    print(f"\nMACD Line:   {macd_line.iloc[-1]:.4f}")
    print(f"Signal Line: {signal_line.iloc[-1]:.4f}")
    print(f"bull3 (MACD > Signal): {macd_line.iloc[-1] > signal_line.iloc[-1]}")

    bull1 = bool(ema20.iloc[-1] > ema55.iloc[-1])
    bull2 = bool(rsi14.iloc[-1] > rsi_ma.iloc[-1])
    bull3 = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    score  = int(bull1) + int(bull2) + int(bull3)
    signal = "BUY" if score >= 2 else "SELL"

    print(f"\nScore: {score}/3  →  Signal: {signal}")
    print(f"{'='*60}\n")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    tf_map = {
        "daily":   "1Day",
        "weekly":  "1Week",
        "monthly": "1Month",
    }

    today   = datetime.utcnow().strftime("%b %-d, %Y")
    symbols = [s for s, _ in TICKERS]

    # Determine which timeframes should update their previous signal today
    # Daily:   always update previous (runs every weekday)
    # Weekly:  only update previous on Monday (last week's candle just closed)
    # Monthly: only update previous on the 1st (last month's candle just closed)
    now = datetime.utcnow()
    update_previous = {
        "daily":   True,
        "weekly":  now.weekday() == 0,   # Monday = 0
        "monthly": now.day == 1,
    }
    print(f"Update previous — Daily: {update_previous['daily']} | Weekly: {update_previous['weekly']} | Monthly: {update_previous['monthly']}")

    # Load existing signals.json to carry forward state
    prev_state = {}
    try:
        with open("signals.json", "r") as f:
            prev_data = json.load(f)
        for row in prev_data.get("signals", []):
            prev_state[row["ticker"]] = {
                label: {
                    "signal":   row["timeframes"].get(label, {}).get("signal"),
                    "previous": row["timeframes"].get(label, {}).get("previous"),
                }
                for label in ["daily", "weekly", "monthly"]
            }
        print(f"Loaded previous state for {len(prev_state)} tickers")
    except FileNotFoundError:
        print("No previous signals.json found — first run")

    print("Fetching latest prices…")
    prices = fetch_latest_prices(symbols)

    results = []
    for symbol, sector in TICKERS:
        row = {
            "ticker":  symbol,
            "sector":  sector,
            "updated": today,
            "timeframes": {},
        }

        daily_close = None
        for label, alpaca_tf in tf_map.items():
            try:
                df  = fetch_bars(symbol, alpaca_tf)
                sig = compute_signal(df)
                if label == "daily" and sig.get("last_close"):
                    daily_close = sig["last_close"]
                sig.pop("last_close", None)
                # Add previous signal
                # Previous signal logic per timeframe:
                # Daily   → always update: previous = what yesterday's signal was
                # Weekly  → only update on Monday: previous = last week's completed signal
                # Monthly → only update on 1st: previous = last month's completed signal
                last = prev_state.get(symbol, {}).get(label, {})
                last_signal   = last.get("signal")
                last_previous = last.get("previous")

                if update_previous[label]:
                    # Time to rotate: current becomes previous, new signal becomes current
                    sig["previous"] = last_signal if last_signal else last_previous
                else:
                    # Not time to update this timeframe's previous — keep it frozen
                    sig["previous"] = last_previous

                print(f"  {symbol:10s} {label:8s} bars={len(df):4d}  signal={sig['signal']}  prev={sig['previous']}")
            except Exception as e:
                sig = {"signal": "ERR", "previous": None, "error": str(e)}
                print(f"  WARNING {symbol} {label}: {e}")
            row["timeframes"][label] = sig

        row["price"] = prices.get(symbol) or daily_close
        results.append(row)

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "signals":   results,
    }

    with open("signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone — signals.json written with {len(results)} tickers")
    print("Prices:", {r["ticker"]: r["price"] for r in results})

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "debug":
        # Usage: python signal_engine.py debug XLE 1Month
        ticker    = sys.argv[2] if len(sys.argv) > 2 else "XLE"
        timeframe = sys.argv[3] if len(sys.argv) > 3 else "1Month"
        print("CallingMarkets Signal Engine — Debug Mode")
        debug_ticker(ticker, timeframe)
    else:
        print("CallingMarkets Signal Engine\n")
        run()
