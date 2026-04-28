#!/usr/bin/env python3
"""
DMG Capital — Top 5 Crypto + Gold Rotation Backtest
- Universe: Top 5 crypto by market cap (yearly snapshots, survivorship-bias-free)
- Signal: Weekly 2-of-3 momentum (EMA20>EMA55, RSI14>RSI_EMA14, MACD>Signal)
- BTC Gate: When BTC weekly SELL:
    → If PAXG weekly BUY: allocate to PAXG (gold)
    → If PAXG weekly SELL: hold USDT (cash)
- When BTC gate OPEN: equal weight all top-5 assets with BUY signal
- Data: Kraken public OHLC API (weekly bars, no key required)
"""

import json, time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

STARTING_CAPITAL = 100_000.0
KRAKEN_BASE      = "https://api.kraken.com/0/public"

# ── Top 5 universe per year (survivorship-bias-free) ─────────────────────────
# Excludes stablecoins, exchange tokens, wrapped tokens
UNIVERSE_BY_YEAR = {
    2018: ["BTC", "ETH", "XRP", "BCH", "LTC"],
    2019: ["BTC", "ETH", "XRP", "LTC", "BCH"],
    2020: ["BTC", "ETH", "XRP", "BCH", "LTC"],
    2021: ["BTC", "ETH", "XRP", "ADA", "LTC"],
    2022: ["BTC", "ETH", "SOL", "ADA", "XRP"],
    2023: ["BTC", "ETH", "XRP", "DOGE", "ADA"],
    2024: ["BTC", "ETH", "SOL", "XRP", "ADA"],
    2025: ["BTC", "ETH", "XRP", "SOL", "DOGE"],
}

KRAKEN_PAIRS = {
    "BTC":  "XBTUSD",
    "ETH":  "ETHUSD",
    "XRP":  "XRPUSD",
    "LTC":  "LTCUSD",
    "BCH":  "BCHUSD",
    "ADA":  "ADAUSD",
    "SOL":  "SOLUSD",
    "DOGE": "XDGUSD",
    "PAXG": "PAXGUSD",  # Gold token
}

ALL_TICKERS = sorted(set(
    t for yr in UNIVERSE_BY_YEAR.values() for t in yr
) | {"PAXG"})

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def calc_rsi(s, n=14):
    d  = s.diff()
    ag = calc_rma(d.clip(lower=0), n)
    al = calc_rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def compute_signal(weekly_close):
    """2-of-3 weekly momentum signal."""
    if len(weekly_close) < 60: return None
    ema20 = calc_ema(weekly_close, 20)
    ema55 = calc_ema(weekly_close, 55)
    rsi   = calc_rsi(weekly_close, 14)
    rma   = calc_ema(rsi, 14)
    macd  = calc_ema(weekly_close, 12) - calc_ema(weekly_close, 26)
    sig   = calc_ema(macd, 9)
    score = ((ema20>ema55).astype(int) +
             (rsi>rma).astype(int) +
             (macd>sig).astype(int))
    return score.apply(lambda s: "BUY" if s >= 2 else "SELL")

# ── Fetch Kraken weekly OHLC ──────────────────────────────────────────────────
def fetch_weekly(ticker):
    pair = KRAKEN_PAIRS.get(ticker)
    if not pair: return None
    try:
        r = requests.get(
            f"{KRAKEN_BASE}/OHLC",
            params={"pair": pair, "interval": 10080},
            timeout=30
        )
        if r.status_code != 200: return None
        data = r.json()
        if data.get("error"): return None
        result = data.get("result", {})
        key = [k for k in result if k != "last"]
        if not key: return None
        bars = result[key[0]]
        if not bars: return None
        df = pd.DataFrame(bars, columns=["time","open","high","low","close","vwap","volume","count"])
        df["date"]  = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df["close"] = df["close"].astype(float)
        df = df.drop_duplicates("date").set_index("date").sort_index()
        weekly = df["close"].resample("W-FRI").last().dropna()
        return weekly if len(weekly) >= 52 else None
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return None

def fetch_all():
    price_data = {}
    print(f"\nFetching {len(ALL_TICKERS)} tickers from Kraken...")
    for i, ticker in enumerate(ALL_TICKERS):
        series = fetch_weekly(ticker)
        if series is not None:
            price_data[ticker] = series
            print(f"  {ticker:6s}: {len(series):3d} weeks "
                  f"({series.index[0].strftime('%Y-%m')} → {series.index[-1].strftime('%Y-%m')})")
        else:
            print(f"  {ticker:6s}: ✗ no data")
        if (i+1) % 4 == 0: time.sleep(0.5)
    return price_data

# ── Universe helper ───────────────────────────────────────────────────────────
def get_universe(date):
    year = date.year
    available = [y for y in sorted(UNIVERSE_BY_YEAR.keys()) if y <= year]
    return UNIVERSE_BY_YEAR[available[-1]] if available else []

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data):
    # Compute signals for all tickers including PAXG
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    # Weekly date range
    start = pd.Timestamp("2018-01-05", tz="UTC")
    end   = pd.Timestamp.now(tz="UTC").normalize()
    all_weeks = pd.date_range(start, end, freq="W-FRI")

    capital      = STARTING_CAPITAL
    holdings     = {}
    cash         = STARTING_CAPITAL
    equity_curve = []
    trades       = []
    weekly_rets  = []
    prev_val     = STARTING_CAPITAL

    for date in all_weeks:
        universe = get_universe(date)

        # Current prices
        prices_now = {}
        for t in list(universe) + ["PAXG"]:
            if t in price_data:
                mask = price_data[t].index <= date
                if mask.any():
                    prices_now[t] = float(price_data[t][mask].iloc[-1])

        # Portfolio value
        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val
        weekly_rets.append((port_val/prev_val)-1 if prev_val > 0 else 0)
        prev_val  = port_val

        # ── BTC gate ─────────────────────────────────────────────────────────
        btc_signal = "SELL"
        if "BTC" in signals:
            mask = signals["BTC"].index <= date
            if mask.any():
                btc_signal = signals["BTC"][mask].iloc[-1]

        # ── PAXG signal (used when BTC gate is SELL) ─────────────────────────
        paxg_signal = "SELL"
        if "PAXG" in signals:
            mask = signals["PAXG"].index <= date
            if mask.any():
                paxg_signal = signals["PAXG"][mask].iloc[-1]

        # ── Determine target allocation ───────────────────────────────────────
        buy_tickers = []
        defensive   = None   # "PAXG" or None (USDT)
        sig_snap    = {}

        if btc_signal == "BUY":
            # Gate open — equal weight BUY signals in universe
            for ticker in universe:
                if ticker not in signals or ticker not in prices_now:
                    continue
                mask = signals[ticker].index <= date
                if mask.any():
                    s = signals[ticker][mask].iloc[-1]
                    sig_snap[ticker] = s
                    if s == "BUY":
                        buy_tickers.append(ticker)
        else:
            # Gate closed — go defensive
            sig_snap["BTC"] = "SELL"
            if paxg_signal == "BUY" and "PAXG" in prices_now:
                defensive = "PAXG"
            # else: 100% USDT (cash)

        # ── Track changes for trade logging ──────────────────────────────────
        prev_buy_set = set(holdings.keys())
        new_buy_set  = set(buy_tickers) | ({defensive} if defensive else set())
        entered      = new_buy_set - prev_buy_set
        exited       = prev_buy_set - new_buy_set

        # ── Rebalance ─────────────────────────────────────────────────────────
        proceeds = cash
        for ticker, shares in holdings.items():
            p = prices_now.get(ticker, 0)
            proceeds += shares * p
            if ticker in exited:
                trades.append({"date": date.strftime("%Y-%m-%d"),
                               "action": "SELL", "ticker": ticker,
                               "price": round(p,4),
                               "value": round(shares*p,2),
                               "reason": "Signal SELL or universe exit"})

        holdings = {}
        cash     = proceeds  # all to USDT

        # Allocate to holdings
        if defensive:
            # 100% PAXG
            p = prices_now.get("PAXG", 0)
            if p > 0:
                shares = proceeds / p
                holdings["PAXG"] = shares
                cash = 0.0
                if "PAXG" in entered:
                    trades.append({"date": date.strftime("%Y-%m-%d"),
                                   "action": "BUY", "ticker": "PAXG",
                                   "price": round(p,4), "shares": round(shares,6),
                                   "value": round(proceeds,2), "weight": 100.0,
                                   "reason": "BTC gate SELL, PAXG BUY"})
        elif buy_tickers:
            w = 1.0 / len(buy_tickers)
            for ticker in buy_tickers:
                alloc = proceeds * w
                p = prices_now.get(ticker, 0)
                if p > 0:
                    shares = alloc / p
                    holdings[ticker] = shares
                    cash -= alloc
                    if ticker in entered:
                        trades.append({"date": date.strftime("%Y-%m-%d"),
                                       "action": "BUY", "ticker": ticker,
                                       "price": round(p,4), "shares": round(shares,6),
                                       "value": round(alloc,2),
                                       "weight": round(w*100,2),
                                       "signal": sig_snap.get(ticker,"—")})

        # Log HOLD
        if not entered and not exited:
            state = "PAXG defensive" if defensive else (f"{len(buy_tickers)} assets" if buy_tickers else "100% USDT")
            trades.append({"date": date.strftime("%Y-%m-%d"), "action": "HOLD",
                           "note": f"No changes — {state}"})

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val

        equity_curve.append({
            "date":        date.strftime("%Y-%m-%d"),
            "value":       round(port_val,2),
            "btc_signal":  btc_signal,
            "paxg_signal": paxg_signal,
            "defensive":   defensive,
            "buy_tickers": buy_tickers,
            "cash_pct":    round(cash/port_val*100,1) if port_val>0 else 100,
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq_vals  = [e["value"] for e in equity_curve]
    final    = eq_vals[-1] if eq_vals else STARTING_CAPITAL
    total_r  = (final/STARTING_CAPITAL-1)*100
    n_weeks  = len(eq_vals)
    years    = n_weeks/52
    cagr     = ((final/STARTING_CAPITAL)**(1/max(years,0.1))-1)*100

    arr    = np.array(weekly_rets[1:])
    sharpe = float(np.mean(arr)/np.std(arr)*np.sqrt(52)) if np.std(arr)>0 else 0

    peak, max_dd = STARTING_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak-v)/peak*100
        if dd > max_dd: max_dd = dd

    # BTC B&H benchmark
    btc_bah = None
    if "BTC" in price_data:
        btc = price_data["BTC"]
        mask = btc.index >= pd.Timestamp("2018-01-05", tz="UTC")
        if mask.any():
            btc_s = btc[mask]
            btc_bah = round((float(btc_s.iloc[-1])/float(btc_s.iloc[0])-1)*100, 2)

    # Time breakdown
    n_paxg   = sum(1 for e in equity_curve if e.get("defensive")=="PAXG")
    n_usdt   = sum(1 for e in equity_curve if not e.get("buy_tickers") and not e.get("defensive"))
    n_crypto = sum(1 for e in equity_curve if e.get("buy_tickers"))

    print(f"\n{'─'*52}")
    print(f"  Strategy: Top 5 Crypto + PAXG Defensive")
    print(f"  Backtest: {equity_curve[0]['date']} → {equity_curve[-1]['date']}")
    print(f"  Portfolio:     ${final:>12,.2f}")
    print(f"  Total Return:  {total_r:>+.2f}%")
    print(f"  BTC B&H:       {btc_bah:>+.2f}%" if btc_bah else "")
    print(f"  CAGR:          {cagr:>+.2f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Max Drawdown:  -{max_dd:.2f}%")
    print(f"  Weeks:         {n_weeks}")
    print(f"\n  Time allocation:")
    print(f"    In crypto:   {n_crypto/n_weeks*100:.1f}% ({n_crypto} weeks)")
    print(f"    In PAXG:     {n_paxg/n_weeks*100:.1f}% ({n_paxg} weeks)")
    print(f"    In USDT:     {n_usdt/n_weeks*100:.1f}% ({n_usdt} weeks)")

    # Current signals
    now = pd.Timestamp.now(tz="UTC")
    universe_now = get_universe(now)
    cur_sigs = {}
    for t in universe_now + ["PAXG"]:
        if t in signals and len(signals[t]) > 0:
            cur_sigs[t] = signals[t].iloc[-1]

    print(f"\n  Current Signals:")
    print(f"    BTC gate:  {cur_sigs.get('BTC','?')}")
    print(f"    PAXG:      {cur_sigs.get('PAXG','?')}")
    buy  = [t for t,s in cur_sigs.items() if s=="BUY" and t not in ("BTC","PAXG")]
    sell = [t for t,s in cur_sigs.items() if s=="SELL" and t not in ("BTC","PAXG")]
    print(f"    BUY  ({len(buy)}): {', '.join(sorted(buy))}")
    print(f"    SELL ({len(sell)}): {', '.join(sorted(sell))}")

    return {
        "total_return_pct":   round(total_r, 2),
        "cagr_pct":           round(cagr, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "final_value":        round(final, 2),
        "current_value":      round(final, 2),
        "btc_bah_pct":        btc_bah,
        "n_weeks":            n_weeks,
        "n_weeks_crypto":     n_crypto,
        "n_weeks_paxg":       n_paxg,
        "n_weeks_usdt":       n_usdt,
        "equity_curve":       equity_curve,
        "trades":             trades,
        "current_signals":    cur_sigs,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("DMG Capital — Top 5 Crypto + PAXG Defensive Backtest")
    print(f"Universe: Top 5 by market cap (yearly snapshots)")
    print(f"Gate: BTC weekly signal — SELL → PAXG if BUY else USDT")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")

    price_data = fetch_all()
    print(f"\nGot data for {len(price_data)}/{len(ALL_TICKERS)} tickers")

    if "PAXG" not in price_data:
        print("WARNING: PAXG not available — defensive allocation will be 100% USDT")

    result = run_backtest(price_data)

    with open("crypto_top5_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\n✓ Results saved to crypto_top5_result.json")

if __name__ == "__main__":
    main()
