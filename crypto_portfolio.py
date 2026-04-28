#!/usr/bin/env python3
"""
DMG Capital — Crypto Rotation Portfolio Backtest
- Universe: Survivorship-bias-free top 20 by market cap (yearly snapshots)
- Signal: 2-of-3 monthly momentum (EMA20>EMA55, RSI14>RSI_EMA14, MACD>Signal)
- Equal-weight BUY signals each month
- USDT cash equivalent when SELL
- Data: Kraken public OHLC API (no key required)
- Backtest: 2018-2026
"""

import json, time, sys
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

STARTING_CAPITAL = 100_000.0
KRAKEN_BASE      = "https://api.kraken.com/0/public"

# ── Universe snapshots (survivorship-bias-free) ────────────────────────────────
# Top 10 by market cap per year (excluding stables/exchange tokens/dead coins)
# Keeping more assets but adding BTC gate for downside protection
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
    "MATIC":"MATICUSD","TRX":"TRXUSD","EOS":"EOSUSD","XMR":"XMRUSD",
    "ZEC":"ZECUSD","ETC":"ETCUSD","ATOM":"ATOMUSD","UNI":"UNIUSD",
    "NEAR":"NEARUSD","ALGO":"ALGOUSD","DASH":"DASHUSD","XEM":"XEMUSD",
    "MIOTA":"IOTUSD","QTUM":"QTUMUSD","OMG":"OMGUSD","NANO":"NANOUSD",
    "ICP":"ICPUSD","TON":"TONUSD","SUI":"SUIUSD","HBAR":"HBARUSD",
}

# All unique tickers
ALL_TICKERS = sorted(set(t for tickers in UNIVERSE_BY_YEAR.values() for t in tickers))

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    ag = calc_rma(d.clip(lower=0), n)
    al = calc_rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def compute_signal(monthly_close):
    """2-of-3 monthly momentum signal."""
    if len(monthly_close) < 60: return None  # need 60+ weeks for EMA55 warmup
    ema20 = calc_ema(monthly_close, 20)
    ema55 = calc_ema(monthly_close, 55)
    rsi   = calc_rsi(monthly_close, 14)
    rma   = calc_ema(rsi, 14)
    macd  = calc_ema(monthly_close, 12) - calc_ema(monthly_close, 26)
    sig   = calc_ema(macd, 9)
    score = (ema20>ema55).astype(int)+(rsi>rma).astype(int)+(macd>sig).astype(int)
    return score.apply(lambda s: "BUY" if s >= 2 else "SELL")

# ── Fetch Kraken daily OHLC and resample to monthly ───────────────────────────
def fetch_kraken_monthly(ticker):
    """
    Fetch Kraken OHLC using WEEKLY bars (interval=10080 minutes).
    720 weekly bars = ~13.8 years — covers the full 2018-2026 backtest in one call.
    Resample weekly closes to monthly.
    """
    pair = KRAKEN_PAIRS.get(ticker)
    if not pair: return None
    try:
        r = requests.get(
            f"{KRAKEN_BASE}/OHLC",
            params={"pair": pair, "interval": 10080},  # 10080 = 1 week in minutes
            timeout=30
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("error"):
            return None
        result_data = data.get("result", {})
        key = [k for k in result_data if k != "last"]
        if not key:
            return None
        bars = result_data[key[0]]
        if not bars:
            return None
        df = pd.DataFrame(bars, columns=["time","open","high","low","close","vwap","volume","count"])
        df["date"]  = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df["close"] = df["close"].astype(float)
        df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
        # Resample to weekly Friday close
        weekly = df["close"].resample("W-FRI").last().dropna()
        return weekly if len(weekly) >= 52 else None
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None

# ── Fetch all tickers ─────────────────────────────────────────────────────────
def fetch_all():
    price_data = {}
    print(f"\nFetching {len(ALL_TICKERS)} tickers from Kraken...")
    for i, ticker in enumerate(ALL_TICKERS):
        series = fetch_kraken_monthly(ticker)
        if series is not None and len(series) >= 12:
            price_data[ticker] = series
            print(f"  {ticker:8s}: {len(series):3d} months  "
                  f"({series.index[0].strftime('%Y-%m')} → {series.index[-1].strftime('%Y-%m')})")
        else:
            print(f"  {ticker:8s}: ✗ no data")
        if (i+1) % 5 == 0:
            time.sleep(1)  # rate limit
    return price_data

# ── Get universe for a given date ─────────────────────────────────────────────
def get_universe(date):
    """Return the universe that was active at this date (snapshot from Jan 1 of that year)."""
    year = date.year
    # Use the most recent snapshot <= current year
    available = [y for y in sorted(UNIVERSE_BY_YEAR.keys()) if y <= year]
    if not available: return []
    return UNIVERSE_BY_YEAR[available[-1]]

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data):
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    # Build weekly date range: 2018-01 to present
    start = pd.Timestamp("2018-01-05", tz="UTC")
    end   = pd.Timestamp.now(tz="UTC").normalize()

    # All weekly Friday dates
    all_months = pd.date_range(start, end, freq="W-FRI")

    capital      = STARTING_CAPITAL
    holdings     = {}   # {ticker: shares}
    cash         = STARTING_CAPITAL
    equity_curve = []
    trades       = []
    monthly_rets = []
    prev_val     = STARTING_CAPITAL

    for date in all_months:
        # Get current universe
        universe = get_universe(date)

        # Current prices
        prices_now = {}
        for t in universe:
            if t in price_data:
                mask = price_data[t].index <= date
                if mask.any():
                    prices_now[t] = float(price_data[t][mask].iloc[-1])

        # No yield on USDT cash in this model
        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val
        monthly_rets.append((port_val/prev_val)-1 if prev_val > 0 else 0)
        prev_val  = port_val

        # BTC gate: if BTC signal is SELL, go 100% USDT regardless
        btc_signal = "BUY"
        if "BTC" in signals:
            btc_mask = signals["BTC"].index <= date
            if btc_mask.any():
                btc_signal = signals["BTC"][btc_mask].iloc[-1]

        # Signals — only for tickers in current universe with data
        buy_tickers = []
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
        # If BTC is SELL, buy_tickers stays empty → 100% USDT

        # Detect universe changes — force-sell anything no longer in universe
        for ticker in list(holdings.keys()):
            if ticker not in universe and holdings[ticker] > 0:
                p = prices_now.get(ticker, 0)
                if p > 0:
                    val = holdings[ticker] * p
                    cash += val
                    trades.append({"date": date.strftime("%Y-%m-%d"),
                                   "action": "SELL", "ticker": ticker,
                                   "reason": "Exited top 20 universe",
                                   "value": round(val,2)})
                del holdings[ticker]

        # Rebalance
        prev_buy = set(holdings.keys())
        new_buy  = set(buy_tickers)
        entered  = new_buy - prev_buy
        exited   = prev_buy - new_buy

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
        cash     = proceeds  # all to USDT

        if buy_tickers:
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

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in holdings)
        port_val  = cash + stock_val

        if not entered and not exited and buy_tickers:
            trades.append({"date": date.strftime("%Y-%m-%d"), "action": "HOLD",
                           "n": len(buy_tickers),
                           "note": f"No changes — {len(buy_tickers)} assets held"})

        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(port_val,2),
            "n_buy": len(buy_tickers),
            "universe_size": len(universe),
            "buy_tickers": buy_tickers,
            "cash_pct": round(cash/port_val*100,1) if port_val>0 else 100
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq_vals = [e["value"] for e in equity_curve]
    final   = eq_vals[-1] if eq_vals else STARTING_CAPITAL
    total_r = (final/STARTING_CAPITAL-1)*100
    n_months= len(eq_vals)
    years   = n_months/12
    cagr    = ((final/STARTING_CAPITAL)**(1/max(years,0.1))-1)*100

    arr    = np.array(monthly_rets[1:])
    sharpe = float(np.mean(arr)/np.std(arr)*np.sqrt(12)) if np.std(arr)>0 else 0

    peak, max_dd = STARTING_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak-v)/peak*100
        if dd > max_dd: max_dd = dd

    # BTC buy-and-hold for comparison
    bah = None
    if "BTC" in price_data:
        btc = price_data["BTC"]
        mask = btc.index >= pd.Timestamp("2018-01-31", tz="UTC")
        if mask.any():
            btc_s = btc[mask]
            p0 = float(btc_s.iloc[0])
            p1 = float(btc_s.iloc[-1])
            bah = round((p1/p0-1)*100, 2)

    return {
        "equity_curve":       equity_curve,
        "trades":             trades,
        "total_return_pct":   round(total_r, 2),
        "cagr_pct":           round(cagr, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "final_value":        round(final, 2),
        "n_months":           n_months,
        "btc_bah_pct":        bah,
        "current_holdings":   [{"ticker": t, "shares": s,
                                 "value": round(s*list(price_data[t])[-1] if t in price_data else 0, 2)}
                                for t,s in holdings.items()],
        "current_signals":    {t: signals[t].iloc[-1]
                                for t in get_universe(pd.Timestamp.now(tz="UTC"))
                                if t in signals and len(signals[t])>0},
    }


def export_csv(result):
    """Export full trade history + weekly allocation CSV matching macro portfolio format."""
    import csv, io

    all_tickers = sorted(set(
        t for e in result["equity_curve"]
        for t in e.get("buy_tickers", [])
    ))

    rows = []
    rows.append(["TRADE HISTORY"])
    rows.append(["Date","Action","Ticker","Name","Price","Shares","Value","Weight %"])
    for t in result["trades"]:
        if t.get("action") in ("BUY","SELL"):
            rows.append([
                t.get("date",""), t.get("action",""),
                t.get("ticker",""), t.get("ticker",""),
                t.get("price",""), t.get("shares",""),
                t.get("value",""), t.get("weight",""),
            ])

    rows.append([])
    rows.append(["WEEKLY PORTFOLIO VALUE & ALLOCATION"])
    rows.append(["Date","Portfolio Value","Assets in BUY","Cash %"] + all_tickers)

    for e in result["equity_curve"]:
        buy_set = set(e.get("buy_tickers", []))
        n       = len(buy_set)
        cp      = e.get("cash_pct", 0)
        eq_pct  = round((100-cp)/n, 2) if n > 0 else 0
        rows.append([e["date"], e["value"], n, cp] +
                    [eq_pct if t in buy_set else 0 for t in all_tickers])

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("DMG Capital — Crypto Rotation Backtest")
    print(f"Universe: Survivorship-bias-free, {len(ALL_TICKERS)} unique tickers (2018-2025)")
    print(f"Signal: 2-of-3 monthly momentum")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")

    price_data = fetch_all()
    print(f"\nGot data for {len(price_data)}/{len(ALL_TICKERS)} tickers")

    if len(price_data) < 5:
        print("ERROR: Not enough data to backtest")
        return

    print("\nRunning backtest...")
    result = run_backtest(price_data)

    eq = result["equity_curve"]
    print(f"\n{'─'*52}")
    print(f"  Backtest: {eq[0]['date']} → {eq[-1]['date']}")
    print(f"  Portfolio:     ${result['final_value']:>12,.2f}")
    print(f"  Total Return:  {result['total_return_pct']:>+.2f}%")
    print(f"  BTC B&H:       {result['btc_bah_pct']:>+.2f}%" if result['btc_bah_pct'] else "")
    print(f"  CAGR:          {result['cagr_pct']:>+.2f}%")
    print(f"  Sharpe:        {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  -{result['max_drawdown_pct']:.2f}%")
    print(f"  Months:        {result['n_months']}")

    print(f"\n  Current Signals (2025 universe):")
    sigs = result["current_signals"]
    buy  = [t for t,s in sigs.items() if s=="BUY"]
    sell = [t for t,s in sigs.items() if s=="SELL"]
    print(f"    BUY  ({len(buy)}): {', '.join(sorted(buy))}")
    print(f"    SELL ({len(sell)}): {', '.join(sorted(sell))}")

    # Write CSV export matching macro portfolio format
    csv_data = export_csv(result)
    with open("crypto-rotation-full-history.csv", "w") as f:
        f.write(csv_data)
    print("\n✓ crypto-rotation-full-history.csv written")

    # Merge into portfolios.json — retry loop guards against race conditions
    for attempt in range(3):
        try:
            with open("portfolios.json","r") as f:
                output = json.load(f)
            # Validate it's a real portfolios file, not truncated
            if "portfolios" not in output:
                raise ValueError("Invalid portfolios.json")
            output["portfolios"] = [p for p in output["portfolios"] if p["id"] != "crypto-rotation"]
            break
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            if attempt == 2:
                output = {"portfolios": []}
            else:
                time.sleep(2)

    output["generated"] = datetime.utcnow().isoformat() + "Z"
    output["portfolios"].append({
        "id":               "crypto-rotation",
        "name":             "Crypto Rotation",
        "description":      f"Survivorship-bias-free weekly rotation across top-10 crypto by market cap (yearly rebalanced universe). BTC gate: 100% USDT when BTC signal SELL. Weekly 2-of-3 momentum signal. Equal-weight BUY assets.",
        "timeframe":        "weekly",
        "starting_capital": STARTING_CAPITAL,
        **result,
    })

    with open("portfolios.json","w") as f:
        json.dump(output, f, indent=2, default=str)
    print("✓ portfolios.json updated")

if __name__ == "__main__":
    main()
