"""
CallingMarkets Signal Engine
Replicates Pine Script indicator logic using Alpaca market data.
Outputs signals.json, which is served directly from this GitHub repo.
"""

import json
import os
from datetime import datetime
import requests
import pandas as pd

# ── API KEYS (loaded from GitHub Secrets — never hardcode these) ───────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
BASE_URL          = "https://data.alpaca.markets/v2"

# ── TICKERS ────────────────────────────────────────────────────────────────────
# Add or remove tickers here. Format: ("SYMBOL", "Sector")
# Crypto symbols must include /USD (e.g. "BTC/USD")
TICKERS = [
    ("SPY",     "ETF"),
    ("QQQ",     "ETF"),
    ("GLD",     "Commodity"),
    ("BTC/USD", "Crypto"),
    ("ETH/USD", "Crypto"),
    ("NVDA",    "Technology"),
    ("AAPL",    "Technology"),
    ("TSLA",    "Technology"),
    ("AMZN",    "Consumer"),
]

# ── INDICATOR PARAMS (must match your Pine Script) ─────────────────────────────
EMA_FAST   = 20
EMA_SLOW   = 55
RSI_LEN    = 14
RSI_MA_LEN = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
ADX_LEN    = 14
ADX_THRESH = 20

# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

def fetch_bars(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    is_crypto  = "/" in symbol
    # Alpaca crypto API uses BTCUSD format (no slash)
    api_symbol = symbol.replace("/", "") if is_crypto else symbol
    endpoint   = "crypto/bars" if is_crypto else "stocks/bars"

    params = {
        "symbols":   api_symbol,
        "timeframe": timeframe,
        "limit":     limit,
        "sort":      "asc",
    }
    # Stocks: use IEX free feed — no paid subscription required
    if not is_crypto:
        params["feed"] = "iex"

    r = requests.get(f"{BASE_URL}/{endpoint}", headers=get_headers(), params=params)
    r.raise_for_status()
    data = r.json()

    bars_raw = data.get("bars", {}).get(api_symbol, [])
    if not bars_raw:
        return pd.DataFrame()

    df = pd.DataFrame(bars_raw)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

# ── INDICATOR MATH ─────────────────────────────────────────────────────────────
def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(series, length):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs    = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast, slow, sig):
    macd_line   = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line

def calc_adx(df, length):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    alpha    = 1 / length
    atr      = tr.ewm(alpha=alpha, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx      = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx

def compute_signal(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < EMA_SLOW + 10:
        return {"signal": "N/A"}

    src                    = df["close"]
    ema20                  = calc_ema(src, EMA_FAST)
    ema55                  = calc_ema(src, EMA_SLOW)
    rsi14                  = calc_rsi(src, RSI_LEN)
    rsi_ma                 = calc_ema(rsi14, RSI_MA_LEN)
    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)
    adx                    = calc_adx(df, ADX_LEN)

    # Replicate Pine Script: hold last signal when ADX drops below threshold
    is_buy = False
    for i in range(len(df)):
        bull1 = ema20.iloc[i]      > ema55.iloc[i]
        bull2 = rsi14.iloc[i]      > rsi_ma.iloc[i]
        bull3 = macd_line.iloc[i]  > signal_line.iloc[i]
        raw   = (bull1 + bull2 + bull3) >= 2
        if adx.iloc[i] >= ADX_THRESH:
            is_buy = raw

    return {"signal": "BUY" if is_buy else "SELL"}

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    tf_map = {
        "daily":   "1Day",
        "weekly":  "1Week",
        "monthly": "1Month",
    }

    today   = datetime.utcnow().strftime("%b %-d, %Y")
    results = []

    for symbol, sector in TICKERS:
        row = {
            "ticker":     symbol,
            "sector":     sector,
            "updated":    today,
            "timeframes": {},
        }
        for label, alpaca_tf in tf_map.items():
            try:
                df  = fetch_bars(symbol, alpaca_tf, limit=300)
                sig = compute_signal(df)
            except Exception as e:
                sig = {"signal": "ERR", "error": str(e)}
                print(f"  WARNING {symbol} {label}: {e}")
            row["timeframes"][label] = sig

        results.append(row)
        print(f"  {symbol:10s}  D={row['timeframes']['daily']['signal']:4s}  "
              f"W={row['timeframes']['weekly']['signal']:4s}  "
              f"M={row['timeframes']['monthly']['signal']}")

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "signals":   results,
    }

    with open("signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone — signals.json written with {len(results)} tickers")

if __name__ == "__main__":
    print("CallingMarkets Signal Engine\n")
    run()
