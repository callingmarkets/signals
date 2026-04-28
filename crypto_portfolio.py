#!/usr/bin/env python3
"""
DMG Capital — Top 10 Crypto + PAXG Defensive Rotation
- Universe: Top 10 crypto by market cap (yearly snapshots, survivorship-bias-free)
- Signal: Weekly 2-of-3 momentum (EMA20>EMA55, RSI14>EMA14, MACD>Signal)
- BTC Gate: When BTC weekly SELL:
    → PAXG weekly BUY → 100% PAXG (gold)
    → PAXG weekly SELL → 100% USDT (cash)
- Gate OPEN: equal weight all top-10 assets with BUY signal
- Data: Kraken public OHLC API (weekly bars, no key required)
"""

import json, time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

STARTING_CAPITAL = 100_000.0
KRAKEN_BASE      = "https://api.kraken.com/0/public"

# ── Top 10 universe per year (survivorship-bias-free) ─────────────────────────
# Excludes stablecoins (USDT/USDC/BUSD/DAI), exchange tokens (BNB/LEO/HT),
# wrapped tokens (WBTC), and dead coins (LUNA/FTT)
UNIVERSE_BY_YEAR = {
    2018: ["BTC","ETH","XRP","BCH","ADA","LTC","XMR","ETC","ZEC","DASH"],
    2019: ["BTC","ETH","XRP","LTC","BCH","XLM","ADA","TRX","XMR","DOGE"],
    2020: ["BTC","ETH","XRP","BCH","LTC","XLM","ADA","LINK","TRX","XMR"],
    2021: ["BTC","ETH","XRP","ADA","LTC","DOT","LINK","DOGE","BCH","UNI"],
    2022: ["BTC","ETH","SOL","ADA","XRP","DOT","DOGE","AVAX","LTC","TRX"],
    2023: ["BTC","ETH","XRP","DOGE","ADA","SOL","LTC","TRX","AVAX","LINK"],
    2024: ["BTC","ETH","SOL","XRP","ADA","AVAX","DOGE","TRX","LINK","LTC"],
    2025: ["BTC","ETH","XRP","SOL","DOGE","ADA","TRX","AVAX","LINK","BCH"],
}

KRAKEN_PAIRS = {
    "BTC":"XBTUSD","ETH":"ETHUSD","XRP":"XRPUSD","LTC":"LTCUSD",
    "BCH":"BCHUSD","ADA":"ADAUSD","DOT":"DOTUSD","LINK":"LINKUSD",
    "XLM":"XLMUSD","DOGE":"XDGUSD","SOL":"SOLUSD","AVAX":"AVAXUSD",
    "TRX":"TRXUSD","XMR":"XMRUSD","ZEC":"ZECUSD","ETC":"ETCUSD",
    "ATOM":"ATOMUSD","UNI":"UNIUSD","DASH":"DASHUSD","PAXG":"PAXGUSD",
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
    """2-of-3 weekly momentum — matches Pine Script CallingMarkets Indicator 2."""
    if len(weekly_close) < 60: return None
    ema20 = calc_ema(weekly_close, 20)
    ema55 = calc_ema(weekly_close, 55)
    rsi   = calc_rsi(weekly_close, 14)
    rma   = calc_ema(rsi, 14)          # ta.ema(rsi14, 14) in Pine
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
        if (i+1) % 5 == 0: time.sleep(0.5)
    return price_data

# ── Universe helper ───────────────────────────────────────────────────────────
def get_universe(date):
    year = date.year
    available = [y for y in sorted(UNIVERSE_BY_YEAR.keys()) if y <= year]
    return UNIVERSE_BY_YEAR[available[-1]] if available else []

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data):
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    start     = pd.Timestamp("2018-01-05", tz="UTC")
    end       = pd.Timestamp.now(tz="UTC").normalize()
    all_weeks = pd.date_range(start, end, freq="W-FRI")

    holdings     = {}
    cash         = STARTING_CAPITAL
    equity_curve = []
    trades       = []
    weekly_rets  = []
    prev_val     = STARTING_CAPITAL

    for date in all_weeks:
        universe = get_universe(date)

        prices_now = {}
        for t in list(universe) + ["PAXG"]:
            if t in price_data:
                mask = price_data[t].index <= date
                if mask.any():
                    prices_now[t] = float(price_data[t][mask].iloc[-1])

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val
        weekly_rets.append((port_val/prev_val)-1 if prev_val > 0 else 0)
        prev_val  = port_val

        # ── BTC gate ──────────────────────────────────────────────────────────
        btc_signal = "SELL"
        if "BTC" in signals:
            mask = signals["BTC"].index <= date
            if mask.any():
                btc_signal = signals["BTC"][mask].iloc[-1]

        # ── PAXG signal ───────────────────────────────────────────────────────
        paxg_signal = "SELL"
        if "PAXG" in signals:
            mask = signals["PAXG"].index <= date
            if mask.any():
                paxg_signal = signals["PAXG"][mask].iloc[-1]

        # ── Target allocation ─────────────────────────────────────────────────
        buy_tickers = []
        defensive   = None
        sig_snap    = {}

        if btc_signal == "BUY":
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
            sig_snap["BTC"] = "SELL"
            if paxg_signal == "BUY" and "PAXG" in prices_now:
                defensive = "PAXG"

        # ── Universe exits ────────────────────────────────────────────────────
        for ticker in list(holdings.keys()):
            if ticker not in universe and ticker != "PAXG":
                p = prices_now.get(ticker, 0)
                if p > 0:
                    val = holdings[ticker] * p
                    cash += val
                    trades.append({"date": date.strftime("%Y-%m-%d"),
                                   "action": "SELL", "ticker": ticker,
                                   "price": round(p,4), "value": round(val,2),
                                   "reason": "Exited top 10 universe"})
                del holdings[ticker]

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
                               "reason": "Signal → SELL"})

        holdings = {}
        cash     = proceeds

        if defensive:
            p = prices_now.get("PAXG", 0)
            if p > 0:
                holdings["PAXG"] = proceeds / p
                cash = 0.0
                if "PAXG" in entered:
                    trades.append({"date": date.strftime("%Y-%m-%d"),
                                   "action": "BUY", "ticker": "PAXG",
                                   "price": round(p,4),
                                   "shares": round(proceeds/p,6),
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
                                       "price": round(p,4),
                                       "shares": round(shares,6),
                                       "value": round(alloc,2),
                                       "weight": round(w*100,2),
                                       "signal": sig_snap.get(ticker,"—")})

        if not entered and not exited:
            state = "PAXG defensive" if defensive else (
                    f"{len(buy_tickers)} assets" if buy_tickers else "100% USDT")
            trades.append({"date": date.strftime("%Y-%m-%d"),
                           "action": "HOLD", "note": f"No changes — {state}"})

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val

        equity_curve.append({
            "date":        date.strftime("%Y-%m-%d"),
            "value":       round(port_val,2),
            "btc_signal":  btc_signal,
            "paxg_signal": paxg_signal,
            "defensive":   defensive,
            "buy_tickers": buy_tickers,
            "n_universe":  len(universe),
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

    # Drawdown from post-rebalance equity curve values
    peak = eq_vals[0] if eq_vals else STARTING_CAPITAL
    max_dd = 0.0
    for v in eq_vals:
        if v > peak: peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100
            if dd > max_dd: max_dd = dd

    # BTC B&H benchmark
    btc_bah = None
    if "BTC" in price_data:
        btc = price_data["BTC"]
        mask = btc.index >= pd.Timestamp("2018-01-05", tz="UTC")
        if mask.any():
            btc_s = btc[mask]
            btc_bah = round((float(btc_s.iloc[-1])/float(btc_s.iloc[0])-1)*100, 2)

    n_paxg   = sum(1 for e in equity_curve if e.get("defensive")=="PAXG")
    n_usdt   = sum(1 for e in equity_curve if not e.get("buy_tickers") and not e.get("defensive"))
    n_crypto = sum(1 for e in equity_curve if e.get("buy_tickers"))

    print(f"\n{'─'*52}")
    print(f"  Strategy: Top 10 Crypto + PAXG Defensive")
    print(f"  Backtest: {equity_curve[0]['date']} → {equity_curve[-1]['date']}")
    print(f"  Portfolio:     ${final:>12,.2f}")
    print(f"  Total Return:  {total_r:>+.2f}%")
    if btc_bah: print(f"  BTC B&H:       {btc_bah:>+.2f}%")
    print(f"  CAGR:          {cagr:>+.2f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Max Drawdown:  -{max_dd:.2f}%")
    print(f"  Weeks:         {n_weeks}")
    print(f"\n  Time in crypto: {n_crypto/n_weeks*100:.1f}% ({n_crypto} weeks)")
    print(f"  Time in PAXG:   {n_paxg/n_weeks*100:.1f}% ({n_paxg} weeks)")
    print(f"  Time in USDT:   {n_usdt/n_weeks*100:.1f}% ({n_usdt} weeks)")

    # Current signals
    now = pd.Timestamp.now(tz="UTC")
    universe_now = get_universe(now)
    cur_sigs = {}
    for t in universe_now + ["PAXG"]:
        if t in signals and len(signals[t]) > 0:
            cur_sigs[t] = signals[t].iloc[-1]

    buy  = [t for t,s in cur_sigs.items() if s=="BUY" and t not in ("BTC","PAXG")]
    sell = [t for t,s in cur_sigs.items() if s=="SELL" and t not in ("BTC","PAXG")]
    print(f"\n  Current Signals (last completed weekly bar):")
    print(f"    BTC gate: {cur_sigs.get('BTC','?')}")
    print(f"    PAXG:     {cur_sigs.get('PAXG','?')}")
    print(f"    BUY  ({len(buy)}): {', '.join(sorted(buy))}")
    print(f"    SELL ({len(sell)}): {', '.join(sorted(sell))}")

    # Period returns
    def port_return_days(days):
        cutoff = now - pd.Timedelta(days=days)
        past = [e for e in equity_curve if pd.Timestamp(e["date"]).tz_localize("UTC") <= cutoff]
        return round((final/past[-1]["value"]-1)*100, 2) if past else None

    def btc_return_days(days):
        if "BTC" not in price_data: return None
        btc = price_data["BTC"]
        cutoff = now - pd.Timedelta(days=days)
        past = btc[btc.index <= cutoff]
        return round((float(btc.iloc[-1])/float(past.iloc[-1])-1)*100, 2) if not past.empty else None

    return holdings, {
        "total_return_pct":   round(total_r, 2),
        "cagr_pct":           round(cagr, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "final_value":        round(final, 2),
        "current_value":      round(final, 2),
        "total_return_dollar": round(final - STARTING_CAPITAL, 2),
        "btc_bah_pct":        btc_bah,
        "n_weeks":            n_weeks,
        "n_buy_stocks":       len(buy),
        "n_universe":         len(universe_now),
        "n_weeks_crypto":     n_crypto,
        "n_weeks_paxg":       n_paxg,
        "n_weeks_usdt":       n_usdt,
        "return_1y":          port_return_days(365),
        "return_3y":          port_return_days(365*3),
        "return_5y":          port_return_days(365*5),
        "return_10y":         port_return_days(365*10),
        "bah_return_1y":      btc_return_days(365),
        "bah_return_3y":      btc_return_days(365*3),
        "bah_return_5y":      btc_return_days(365*5),
        "bah_return_10y":     btc_return_days(365*10),
        "equity_curve":       equity_curve,
        "trades":             trades,
        "current_signals":    cur_sigs,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("DMG Capital — Top 10 Crypto + PAXG Defensive Backtest")
    print(f"Universe: Top 10 by market cap (yearly snapshots, 2018-2025)")
    print(f"Gate: BTC weekly signal — SELL → PAXG if BUY, else USDT")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")

    price_data = fetch_all()
    print(f"\nGot data for {len(price_data)}/{len(ALL_TICKERS)} tickers")

    if "PAXG" not in price_data:
        print("WARNING: PAXG unavailable — defensive will be 100% USDT")

    print("\nRunning backtest...")
    holdings, result = run_backtest(price_data)

    # Current holdings with weights
    final = result["final_value"]
    result["current_holdings"] = [
        {"ticker": t, "name": t,
         "shares": round(s, 6),
         "price":  round(float(price_data[t].iloc[-1]), 4) if t in price_data else 0,
         "value":  round(s * float(price_data[t].iloc[-1]), 2) if t in price_data else 0,
         "weight": round(s * float(price_data[t].iloc[-1]) / final * 100, 2) if t in price_data and final > 0 else 0,
         "signal": "BUY"}
        for t, s in holdings.items() if s > 0
    ]

    # Write CSV export
    import csv, io
    all_tickers_csv = sorted(set(
        t for e in result["equity_curve"] for t in e.get("buy_tickers", [])
    ))
    rows = [["TRADE HISTORY"],
            ["Date","Action","Ticker","Name","Price","Shares","Value","Weight %"]]
    for t in result["trades"]:
        if t.get("action") in ("BUY","SELL"):
            rows.append([t.get("date",""), t.get("action",""), t.get("ticker",""),
                         t.get("ticker",""), t.get("price",""), t.get("shares",""),
                         t.get("value",""), t.get("weight","")])
    rows.append([])
    rows.append(["WEEKLY PORTFOLIO VALUE & ALLOCATION"])
    rows.append(["Date","Portfolio Value","Assets in BUY","Cash %"] + all_tickers_csv)
    for e in result["equity_curve"]:
        buy_set = set(e.get("buy_tickers", []))
        n = len(buy_set); cp = e.get("cash_pct", 0)
        eq_pct = round((100-cp)/n, 2) if n > 0 else 0
        rows.append([e["date"], e["value"], n, cp] +
                    [eq_pct if t in buy_set else 0 for t in all_tickers_csv])
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    with open("crypto-rotation-full-history.csv", "w") as f:
        f.write(buf.getvalue())
    print("✓ crypto-rotation-full-history.csv written")

    # Pull latest portfolios.json before merging
    import subprocess
    subprocess.run(["git", "pull", "--rebase", "--quiet"], capture_output=True)

    for attempt in range(3):
        try:
            with open("portfolios.json", "r") as f:
                output = json.load(f)
            if "portfolios" not in output:
                raise ValueError("Invalid")
            output["portfolios"] = [p for p in output["portfolios"] if p["id"] != "crypto-rotation"]
            break
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            output = {"portfolios": []} if attempt == 2 else None
            if output is None: time.sleep(2)

    output["generated"] = datetime.now(timezone.utc).isoformat()
    output["portfolios"].append({
        "id":          "crypto-rotation",
        "name":        "Crypto Rotation",
        "description": "Top-10 crypto by market cap (yearly rebalanced universe). BTC gate: PAXG when BTC SELL + PAXG BUY, else USDT. Weekly 2-of-3 momentum signal.",
        "timeframe":   "weekly",
        "starting_capital": STARTING_CAPITAL,
        **result,
    })

    with open("portfolios.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("✓ portfolios.json updated")

if __name__ == "__main__":
    main()
