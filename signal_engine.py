"""
CallingMarkets Signal Engine
Matches CallingMarkets Indicator 2 (Pine Script v5) exactly.
Signal = BUY if 2 or more of: EMA20 > EMA55, RSI14 > RSI_EMA14, MACD > Signal
No ADX filter.
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

# ── INDICATOR PARAMS (matches Pine Script exactly) ─────────────────────────────
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
        r = requests.get(url, headers=get_headers(), params=params)
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
    """ta.ema() — matches Pine Script exactly. alpha = 2/(length+1)."""
    return series.ewm(span=length, adjust=False).mean()

def calc_rma(series: pd.Series, length: int) -> pd.Series:
    """ta.rma() — Wilder's MA with SMA seed. alpha = 1/length."""
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
    """ta.rsi() — uses RMA internally, matches Pine exactly."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = calc_rma(gain, length)
    avg_l = calc_rma(loss, length)
    rs    = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int, slow: int, sig: int):
    """ta.macd() — all standard EMA."""
    macd_line   = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line

# ── SIGNAL LOGIC ───────────────────────────────────────────────────────────────
def compute_signal(df: pd.DataFrame, label: str = "", symbol: str = "") -> dict:
    """
    Matches CallingMarkets Indicator 2 exactly:
        bull1 = ema20 > ema55
        bull2 = rsi14 > rsi_ma
        bull3 = macd_line > signal_line
        is_buy = (bull1 + bull2 + bull3) >= 2
    No ADX. No state. Pure 2-of-3 on the last bar.
    """
    if df.empty or len(df) < EMA_SLOW + 10:
        return {"signal": "N/A"}

    src                    = df["close"]
    ema20                  = calc_ema(src, EMA_FAST)
    ema55                  = calc_ema(src, EMA_SLOW)
    rsi14                  = calc_rsi(src, RSI_LEN)
    rsi_ma                 = calc_ema(rsi14, RSI_MA_LEN)
    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)

    # Evaluate on the last completed bar only
    bull1 = bool(ema20.iloc[-1]     > ema55.iloc[-1])
    bull2 = bool(rsi14.iloc[-1]     > rsi_ma.iloc[-1])
    bull3 = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    score = int(bull1) + int(bull2) + int(bull3)
    is_buy = score >= 2

    # Debug for SPY
    if symbol == "SPY":
        print(f"    [{label}] Last bar: {df.index[-1].date()}  "
              f"EMA={bull1} RSI={bull2} MACD={bull3} score={score} -> {'BUY' if is_buy else 'SELL'}")

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
                df  = fetch_bars(symbol, alpaca_tf)
                sig = compute_signal(df, label, symbol)
                print(f"  {symbol:10s} {label:8s} bars={len(df):4d}  signal={sig['signal']}")
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
