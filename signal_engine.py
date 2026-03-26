"""
CallingMarkets Signal Engine
Matches CallingMarkets Indicator 2 (Pine Script v5) exactly.
Signal = BUY if 2 or more of: EMA20 > EMA55, RSI14 > RSI_EMA14, MACD > Signal
No ADX filter. Includes latest price fetch.
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
CRYPTO_QUOTE_URL = "https://data.alpaca.markets/v1beta3/crypto/us/quotes/latest"

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
    Fetch latest ask price for all tickers in one batch call each for stocks and crypto.
    Returns dict of {symbol: price}.
    """
    prices = {}
    stocks  = [s for s in symbols if "/" not in s]
    cryptos = [s for s in symbols if "/" in s]

    if stocks:
        try:
            r = requests.get(STOCKS_QUOTE_URL, headers=get_headers(),
                             params={"symbols": ",".join(stocks)})
            r.raise_for_status()
            for sym, data in r.json().get("quotes", {}).items():
                # Use midpoint of bid/ask, fall back to ask price
                bid = data.get("bp", 0)
                ask = data.get("ap", 0)
                if bid and ask:
                    prices[sym] = round((bid + ask) / 2, 2)
                elif ask:
                    prices[sym] = round(ask, 2)
        except Exception as e:
            print(f"  WARNING stock quotes: {e}")

    if cryptos:
        try:
            r = requests.get(CRYPTO_QUOTE_URL, headers=get_headers(),
                             params={"symbols": ",".join(cryptos)})
            r.raise_for_status()
            for sym, data in r.json().get("quotes", {}).items():
                bid = data.get("bp", 0)
                ask = data.get("ap", 0)
                if bid and ask:
                    prices[sym] = round((bid + ask) / 2, 2)
                elif ask:
                    prices[sym] = round(ask, 2)
        except Exception as e:
            print(f"  WARNING crypto quotes: {e}")

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
        return {"signal": "N/A"}

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

    return {"signal": "BUY" if is_buy else "SELL"}

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    tf_map = {
        "daily":   "1Day",
        "weekly":  "1Week",
        "monthly": "1Month",
    }

    today   = datetime.utcnow().strftime("%b %-d, %Y")
    symbols = [s for s, _ in TICKERS]

    # Fetch all latest prices in one batch
    print("Fetching latest prices…")
    prices = fetch_latest_prices(symbols)

    results = []
    for symbol, sector in TICKERS:
        row = {
            "ticker":     symbol,
            "sector":     sector,
            "price":      prices.get(symbol, None),
            "updated":    today,
            "timeframes": {},
        }
        for label, alpaca_tf in tf_map.items():
            try:
                df  = fetch_bars(symbol, alpaca_tf)
                sig = compute_signal(df)
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
