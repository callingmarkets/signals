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
    # ── BROAD MARKET ETFs ──────────────────────────────────────────
    ("SPY",  "ETF - US Market"),
    ("QQQ",  "ETF - US Market"),
    ("IWM",  "ETF - US Market"),
    ("DIA",  "ETF - US Market"),
    ("MDY",  "ETF - US Market"),
    ("VTI",  "ETF - US Market"),
    ("VOO",  "ETF - US Market"),
    ("RSP",  "ETF - US Market"),
    # ── SECTOR ETFs ────────────────────────────────────────────────
    ("XLK",  "ETF - Technology"),
    ("XLF",  "ETF - Financials"),
    ("XLE",  "ETF - Energy"),
    ("XLV",  "ETF - Healthcare"),
    ("XLI",  "ETF - Industrials"),
    ("XLB",  "ETF - Materials"),
    ("XLU",  "ETF - Utilities"),
    ("XLRE", "ETF - Real Estate"),
    ("XLP",  "ETF - Staples"),
    ("XLY",  "ETF - Consumer"),
    ("XLC",  "ETF - Communications"),
    ("XBI",  "ETF - Biotech"),
    ("SMH",  "ETF - Semiconductors"),
    ("SOXX", "ETF - Semiconductors"),
    ("ARKK", "ETF - Innovation"),
    ("ARKG", "ETF - Genomics"),
    ("FINX", "ETF - Fintech"),
    ("CIBR", "ETF - Cybersecurity"),
    ("HACK", "ETF - Cybersecurity"),
    ("ROBO", "ETF - Robotics"),
    ("BOTZ", "ETF - AI & Robotics"),
    # ── INTERNATIONAL ETFs ─────────────────────────────────────────
    ("EFA",  "ETF - International"),
    ("EEM",  "ETF - Emerging Markets"),
    ("VEA",  "ETF - Developed Markets"),
    ("VWO",  "ETF - Emerging Markets"),
    ("FXI",  "ETF - China"),
    ("MCHI", "ETF - China"),
    ("EWJ",  "ETF - Japan"),
    ("EWZ",  "ETF - Brazil"),
    ("INDA", "ETF - India"),
    ("EWG",  "ETF - Germany"),
    ("EWU",  "ETF - UK"),
    ("EWC",  "ETF - Canada"),
    ("EWA",  "ETF - Australia"),
    ("EWY",  "ETF - South Korea"),
    ("EWT",  "ETF - Taiwan"),
    # ── FIXED INCOME ───────────────────────────────────────────────
    ("TLT",  "ETF - Bonds"),
    ("IEF",  "ETF - Bonds"),
    ("SHY",  "ETF - Bonds"),
    ("BND",  "ETF - Bonds"),
    ("AGG",  "ETF - Bonds"),
    ("HYG",  "ETF - High Yield"),
    ("JNK",  "ETF - High Yield"),
    ("LQD",  "ETF - Corp Bonds"),
    ("EMB",  "ETF - EM Bonds"),
    ("TIP",  "ETF - TIPS"),
    ("MUB",  "ETF - Municipal"),
    # ── COMMODITIES ────────────────────────────────────────────────
    ("GLD",  "Commodity"),
    ("IAU",  "Commodity"),
    ("SLV",  "Commodity"),
    ("PPLT", "Commodity"),
    ("USO",  "Commodity"),
    ("BNO",  "Commodity"),
    ("UNG",  "Commodity"),
    ("DBA",  "Commodity"),
    ("CORN", "Commodity"),
    ("WEAT", "Commodity"),
    ("SOYB", "Commodity"),
    ("PDBC", "Commodity"),
    ("DJP",  "Commodity"),
    ("GDX",  "Commodity"),
    ("GDXJ", "Commodity"),
    ("COPX", "Commodity"),
    ("URA",  "Commodity"),
    # ── VOLATILITY & ALTERNATIVES ──────────────────────────────────
    ("VXX",  "ETF - Volatility"),
    ("UVXY", "ETF - Volatility"),
    ("SVXY", "ETF - Volatility"),
    ("IBIT", "ETF - Bitcoin"),
    ("FBTC", "ETF - Bitcoin"),
    ("ETHA", "ETF - Ethereum"),
    # ── CRYPTO ─────────────────────────────────────────────────────
    ("BTC/USD",  "Crypto"),
    ("ETH/USD",  "Crypto"),
    ("SOL/USD",  "Crypto"),
    ("XRP/USD",  "Crypto"),
    ("DOGE/USD", "Crypto"),
    ("AVAX/USD", "Crypto"),
    ("LINK/USD", "Crypto"),
    ("UNI/USD",  "Crypto"),
    ("AAVE/USD", "Crypto"),
    ("LTC/USD",  "Crypto"),
    # ── TECHNOLOGY ─────────────────────────────────────────────────
    ("AAPL",  "Technology"),
    ("MSFT",  "Technology"),
    ("NVDA",  "Technology"),
    ("GOOGL", "Technology"),
    ("META",  "Technology"),
    ("AMZN",  "Technology"),
    ("TSLA",  "Technology"),
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
    ("SNOW",  "Technology"),
    ("PLTR",  "Technology"),
    ("CRWD",  "Technology"),
    ("PANW",  "Technology"),
    ("ZS",    "Technology"),
    ("NET",   "Technology"),
    ("DDOG",  "Technology"),
    ("MDB",   "Technology"),
    ("ADBE",  "Technology"),
    ("NOW",   "Technology"),
    ("INTU",  "Technology"),
    ("WDAY",  "Technology"),
    ("SHOP",  "Technology"),
    ("UBER",  "Technology"),
    ("LYFT",  "Technology"),
    ("ABNB",  "Technology"),
    ("COIN",  "Technology"),
    ("HOOD",  "Technology"),
    ("APP",   "Technology"),
    ("RBLX",  "Technology"),
    ("U",     "Technology"),
    # ── FINANCIALS ─────────────────────────────────────────────────
    ("JPM",  "Financials"),
    ("BAC",  "Financials"),
    ("GS",   "Financials"),
    ("MS",   "Financials"),
    ("WFC",  "Financials"),
    ("C",    "Financials"),
    ("BX",   "Financials"),
    ("KKR",  "Financials"),
    ("APO",  "Financials"),
    ("BLK",  "Financials"),
    ("SCHW", "Financials"),
    ("V",    "Financials"),
    ("MA",   "Financials"),
    ("AXP",  "Financials"),
    ("PYPL", "Financials"),
    ("SQ",   "Financials"),
    ("NU",   "Financials"),
    ("SOFI", "Financials"),
    # ── HEALTHCARE ─────────────────────────────────────────────────
    ("UNH",  "Healthcare"),
    ("JNJ",  "Healthcare"),
    ("LLY",  "Healthcare"),
    ("ABBV", "Healthcare"),
    ("MRK",  "Healthcare"),
    ("PFE",  "Healthcare"),
    ("AMGN", "Healthcare"),
    ("GILD", "Healthcare"),
    ("BIIB", "Healthcare"),
    ("REGN", "Healthcare"),
    ("VRTX", "Healthcare"),
    ("ISRG", "Healthcare"),
    ("BSX",  "Healthcare"),
    ("MDT",  "Healthcare"),
    ("CVS",  "Healthcare"),
    ("HUM",  "Healthcare"),
    # ── ENERGY ─────────────────────────────────────────────────────
    ("XOM",  "Energy"),
    ("CVX",  "Energy"),
    ("COP",  "Energy"),
    ("EOG",  "Energy"),
    ("SLB",  "Energy"),
    ("MPC",  "Energy"),
    ("PSX",  "Energy"),
    ("VLO",  "Energy"),
    ("HAL",  "Energy"),
    ("DVN",  "Energy"),
    ("OXY",  "Energy"),
    ("HES",  "Energy"),
    # ── CONSUMER ───────────────────────────────────────────────────
    ("COST", "Consumer"),
    ("WMT",  "Consumer"),
    ("TGT",  "Consumer"),
    ("HD",   "Consumer"),
    ("LOW",  "Consumer"),
    ("MCD",  "Consumer"),
    ("SBUX", "Consumer"),
    ("NKE",  "Consumer"),
    ("LULU", "Consumer"),
    ("TJX",  "Consumer"),
    ("BKNG", "Consumer"),
    ("MAR",  "Consumer"),
    ("HLT",  "Consumer"),
    ("DIS",  "Consumer"),
    ("NFLX", "Consumer"),
    ("SPOT", "Consumer"),
    # ── INDUSTRIALS ────────────────────────────────────────────────
    ("CAT",  "Industrials"),
    ("DE",   "Industrials"),
    ("RTX",  "Industrials"),
    ("LMT",  "Industrials"),
    ("NOC",  "Industrials"),
    ("GE",   "Industrials"),
    ("HON",  "Industrials"),
    ("BA",   "Industrials"),
    ("UNP",  "Industrials"),
    ("FDX",  "Industrials"),
    ("UPS",  "Industrials"),
    ("WM",   "Industrials"),
    # ── REAL ESTATE ────────────────────────────────────────────────
    ("PLD",  "Real Estate"),
    ("AMT",  "Real Estate"),
    ("EQIX", "Real Estate"),
    ("SPG",  "Real Estate"),
    ("O",    "Real Estate"),
    ("WELL", "Real Estate"),
    # ── MATERIALS ──────────────────────────────────────────────────
    ("NEM",  "Materials"),
    ("FCX",  "Materials"),
    ("AA",   "Materials"),
    ("NUE",  "Materials"),
    ("LIN",  "Materials"),
    ("APD",  "Materials"),
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

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    tf_map = {
        "daily":   "1Day",
        "weekly":  "1Week",
        "monthly": "1Month",
    }

    today   = datetime.utcnow().strftime("%b %-d, %Y")
    symbols = [s for s, _ in TICKERS]

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
                # Capture daily close as price fallback
                if label == "daily" and sig.get("last_close"):
                    daily_close = sig["last_close"]
                # Strip last_close from stored signal (keep json clean)
                sig.pop("last_close", None)
                print(f"  {symbol:10s} {label:8s} bars={len(df):4d}  signal={sig['signal']}")
            except Exception as e:
                sig = {"signal": "ERR", "error": str(e)}
                print(f"  WARNING {symbol} {label}: {e}")
            row["timeframes"][label] = sig

        # Price: prefer live quote, fall back to last daily close
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
    print("CallingMarkets Signal Engine\n")
    run()
