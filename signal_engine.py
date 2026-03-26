"""
CallingMarkets Signal Engine
Replicates Pine Script indicator logic using Alpaca market data.
Outputs signals.json, which is served directly from this GitHub repo.
"""

import json
import os
from datetime import datetime, timedelta
import requests
import pandas as pd

# ── API KEYS ───────────────────────────────────────────────────────────────────
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

STOCKS_URL = "https://data.alpaca.markets/v2/stocks/bars"
CRYPTO_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

# ── TICKERS ────────────────────────────────────────────────────────────────────
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

# ── INDICATOR PARAMS ───────────────────────────────────────────────────────────
EMA_FAST   = 20
EMA_SLOW   = 55
RSI_LEN    = 14
RSI_MA_LEN = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
ADX_LEN    = 14
ADX_THRESH = 20

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

def fetch_bars(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    is_crypto = "/" in symbol
    url    = CRYPTO_URL if is_crypto else STOCKS_URL
    params = {
        "symbols":   symbol,
        "timeframe": timeframe,
        "start":     START_DATES[timeframe],
        "limit":     limit,
        "sort":      "asc",
    }
    r = requests.get(url, headers=get_headers(), params=params)
    r.raise_for_status()
    bars_raw = r.json().get("bars", {}).get(symbol, [])
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

def compute_signal(df: pd.DataFrame, label: str = "", symbol: str = "") -> dict:
    if df.empty or len(df) < EMA_SLOW + 10:
        return {"signal": "N/A"}

    src                    = df["close"]
    ema20                  = calc_ema(src, EMA_FAST)
    ema55                  = calc_ema(src, EMA_SLOW)
    rsi14                  = calc_rsi(src, RSI_LEN)
    rsi_ma                 = calc_ema(rsi14, RSI_MA_LEN)
    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)
    adx                    = calc_adx(df, ADX_LEN)

    # Debug: print last 3 bars of key values for SPY
    if symbol == "SPY":
        print(f"    [{label}] Last 3 bars:")
        for i in [-3, -2, -1]:
            bull1 = ema20.iloc[i] > ema55.iloc[i]
            bull2 = rsi14.iloc[i] > rsi_ma.iloc[i]
            bull3 = macd_line.iloc[i] > signal_line.iloc[i]
            score = bull1 + bull2 + bull3
            print(f"      {df.index[i].date()}  EMA={bull1} RSI={bull2} MACD={bull3} "
                  f"score={score} ADX={adx.iloc[i]:.1f} trending={adx.iloc[i]>=ADX_THRESH}")

    # Pine Script logic: hold last signal when ADX below threshold
    is_buy = False
    for i in range(len(df)):
        bull1 = ema20.iloc[i]     > ema55.iloc[i]
        bull2 = rsi14.iloc[i]     > rsi_ma.iloc[i]
        bull3 = macd_line.iloc[i] > signal_line.iloc[i]
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
                sig = compute_signal(df, label, symbol)
                print(f"  {symbol:10s} {label:8s} bars={len(df):3d}  signal={sig['signal']}")
            except Exception as e:
                sig = {"signal": "ERR", "error": str(e)}
                print(f"  WARNING {symbol} {label}: {e}")
            row["timeframes"][label] = sig

        results.append(row)

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
