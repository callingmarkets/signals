"""
CallingMarkets Signal Engine
Replicates Pine Script indicator logic using Alpaca market data.
Outputs signals.json, which is served directly from this GitHub repo.
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

# ── INDICATOR PARAMS ───────────────────────────────────────────────────────────
EMA_FAST   = 20
EMA_SLOW   = 55
RSI_LEN    = 14
RSI_MA_LEN = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9
ADX_LEN    = 14  # adxlen — smoothing
DI_LEN     = 14  # dilen  — DI length
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
    """Pine ta.ema() — alpha = 2/(length+1)."""
    return series.ewm(span=length, adjust=False).mean()

def calc_rma(series: pd.Series, length: int) -> pd.Series:
    """
    Pine ta.rma() — Wilder's MA with SMA seed.
    First value = SMA of first `length` non-NaN bars, then:
    rma[i] = alpha * src[i] + (1 - alpha) * rma[i-1]  where alpha = 1/length
    """
    alpha  = 1.0 / length
    values = series.values.astype(float)
    result = np.full(len(values), np.nan)

    # Find first valid window for SMA seed
    for i in range(length - 1, len(values)):
        window = values[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            result[i] = np.mean(window)
            for j in range(i + 1, len(values)):
                if np.isnan(values[j]):
                    result[j] = result[j - 1]
                else:
                    result[j] = alpha * values[j] + (1 - alpha) * result[j - 1]
            break

    return pd.Series(result, index=series.index)

def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.rsi() — uses RMA for gain/loss."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = calc_rma(gain, length)
    avg_l = calc_rma(loss, length)
    rs    = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int, slow: int, sig: int):
    """Pine ta.macd() — all EMA."""
    macd_line   = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line

def calc_adx(df: pd.DataFrame, dilen: int, adxlen: int) -> pd.Series:
    """
    Matches Pine Script official ADX indicator exactly:

    dirmov(len) =>
        up   = ta.change(high)
        down = -ta.change(low)
        plusDM  = na(up)   ? na : (up > down   and up > 0   ? up   : 0)
        minusDM = na(down) ? na : (down > up   and down > 0 ? down : 0)
        truerange = ta.rma(ta.tr, len)
        plus  = fixnan(100 * ta.rma(plusDM,  len) / truerange)
        minus = fixnan(100 * ta.rma(minusDM, len) / truerange)

    adx(dilen, adxlen) =>
        [plus, minus] = dirmov(dilen)
        sum = plus + minus
        100 * ta.rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # ta.change(high) and -ta.change(low)
    up   = high.diff()
    down = -low.diff()

    # plusDM: na if up is na, else up if up > down and up > 0, else 0
    plus_dm  = np.where(up.isna(),   np.nan,
               np.where((up > down) & (up > 0),   up,   0.0))
    minus_dm = np.where(down.isna(), np.nan,
               np.where((down > up) & (down > 0), down, 0.0))

    plus_dm  = pd.Series(plus_dm,  index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # ta.tr — True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # ta.rma on TR, plusDM, minusDM
    truerange = calc_rma(tr,       dilen)
    rma_plus  = calc_rma(plus_dm,  dilen)
    rma_minus = calc_rma(minus_dm, dilen)

    # DI+ and DI- with fixnan (forward-fill NaNs)
    di_plus  = (100 * rma_plus  / truerange).ffill()
    di_minus = (100 * rma_minus / truerange).ffill()

    # ADX
    di_sum = di_plus + di_minus
    dx     = (di_plus - di_minus).abs() / di_sum.where(di_sum != 0, 1)
    adx    = 100 * calc_rma(dx, adxlen)

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
    adx                    = calc_adx(df, DI_LEN, ADX_LEN)

    # Debug last 3 bars for SPY
    if symbol == "SPY":
        print(f"    [{label}] Last bar: {df.index[-1].date()}")
        for i in [-3, -2, -1]:
            b1    = ema20.iloc[i] > ema55.iloc[i]
            b2    = rsi14.iloc[i] > rsi_ma.iloc[i]
            b3    = macd_line.iloc[i] > signal_line.iloc[i]
            score = int(b1) + int(b2) + int(b3)
            print(f"      {df.index[i].date()}  EMA={b1} RSI={b2} MACD={b3} "
                  f"score={score} ADX={adx.iloc[i]:.2f} trending={adx.iloc[i]>=ADX_THRESH}")

    # Pine Script: hold last valid signal when ADX < threshold
    is_buy = False
    for i in range(len(df)):
        b1  = ema20.iloc[i]     > ema55.iloc[i]
        b2  = rsi14.iloc[i]     > rsi_ma.iloc[i]
        b3  = macd_line.iloc[i] > signal_line.iloc[i]
        raw = (int(b1) + int(b2) + int(b3)) >= 2
        adx_val = adx.iloc[i]
        if not np.isnan(adx_val) and adx_val >= ADX_THRESH:
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
