#!/usr/bin/env python3
"""
Global Macro Rotation Portfolio
- Universe: 15 most-traded iShares ETFs + Bitcoin (BTC)
- Signal: 2-of-3 MONTHLY momentum (EMA20>EMA55, RSI14>RSI_EMA14, MACD>Signal)
- Entry: Equal weight all assets with monthly BUY signal
- Cash: 100% SGOV when zero assets signal BUY
- No benchmark gate — each asset is its own signal
- Rebalance: First Monday of each month
- Lookback: 20 years (back to ~2005)
"""

import json, os, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests

TIINGO_KEY = os.environ["TIINGO_API_KEY"]
TIINGO_HDR = {"Authorization": f"Token {TIINGO_KEY}", "Content-Type": "application/json"}
TIINGO_URL = "https://api.tiingo.com/tiingo/daily"

STARTING_CAPITAL = 100_000.0
SGOV_FALLBACK    = 0.05 / 12   # Monthly SGOV yield

EMA_FAST  = 20; EMA_SLOW = 55; RSI_LEN = 14
RSI_MA    = 14; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIG = 9

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Equities
    {"ticker": "IVV",  "name": "iShares Core S&P 500 ETF",              "aum": "777B",  "tracks": "S&P 500"},
    {"ticker": "IWF",  "name": "iShares Russell 1000 Growth ETF",        "aum": "123B",  "tracks": "US Large Cap Growth"},
    {"ticker": "IJH",  "name": "iShares Core S&P Mid-Cap ETF",           "aum": "116B",  "tracks": "US Mid Caps"},
    {"ticker": "IJR",  "name": "iShares Core S&P Small-Cap ETF",         "aum": "101B",  "tracks": "US Small Caps (S&P 600)"},
    {"ticker": "IWM",  "name": "iShares Russell 2000 ETF",               "aum": "76B",   "tracks": "US Small Caps"},
    {"ticker": "IWD",  "name": "iShares Russell 1000 Value ETF",         "aum": "73B",   "tracks": "US Large Cap Value"},
    {"ticker": "IVW",  "name": "iShares S&P 500 Growth ETF",             "aum": "69B",   "tracks": "S&P 500 Growth"},
    {"ticker": "IVE",  "name": "iShares S&P 500 Value ETF",              "aum": "49B",   "tracks": "S&P 500 Value"},
    # International
    {"ticker": "IEFA", "name": "iShares Core MSCI EAFE ETF",             "aum": "179B",  "tracks": "Developed Intl"},
    {"ticker": "IEMG", "name": "iShares Core MSCI Emerging Markets ETF", "aum": "149B",  "tracks": "Emerging Markets"},
    {"ticker": "EFA",  "name": "iShares MSCI EAFE ETF",                  "aum": "75B",   "tracks": "Developed Intl ex-US"},
    # Bonds
    {"ticker": "AGG",  "name": "iShares Core U.S. Aggregate Bond ETF",   "aum": "136B",  "tracks": "US Investment Grade Bonds"},
    {"ticker": "IEF",  "name": "iShares 7-10 Year Treasury Bond ETF",    "aum": "49B",   "tracks": "7-10yr Treasuries"},
    {"ticker": "TLT",  "name": "iShares 20+ Year Treasury Bond ETF",     "aum": "45B",   "tracks": "Long-term Treasuries"},
    # Real Assets
    {"ticker": "IAU",  "name": "iShares Gold Trust",                     "aum": "74B",   "tracks": "Physical Gold"},
    # Crypto
    {"ticker": "BTC",  "name": "Bitcoin",                                "aum": "N/A",   "tracks": "Bitcoin"},
]
TICKERS     = [s["ticker"] for s in UNIVERSE]
TICKER_META = {s["ticker"]: s for s in UNIVERSE}

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def calc_rsi(s, n):
    d  = s.diff()
    ag = calc_rma(d.clip(lower=0), n)
    al = calc_rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

def compute_signal(src):
    if len(src) < EMA_SLOW + 10: return None
    ema20 = calc_ema(src, EMA_FAST); ema55 = calc_ema(src, EMA_SLOW)
    rsi   = calc_rsi(src, RSI_LEN);  rma   = calc_ema(rsi, RSI_MA)
    macd  = calc_ema(src, MACD_FAST) - calc_ema(src, MACD_SLOW)
    sig   = calc_ema(macd, MACD_SIG)
    score = (ema20>ema55).astype(int)+(rsi>rma).astype(int)+(macd>sig).astype(int)
    return score.apply(lambda s: "BUY" if s >= 2 else "SELL")

# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_tiingo_monthly(ticker, lookback_days=7300):
    """Fetch daily data and resample to monthly for signal computation."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {"startDate": start.strftime("%Y-%m-%d"),
              "endDate":   end.strftime("%Y-%m-%d"),
              "token": TIINGO_KEY}
    try:
        r = requests.get(f"{TIINGO_URL}/{ticker}/prices",
                         headers=TIINGO_HDR, params=params, timeout=30)
        if r.status_code == 404: return None
        r.raise_for_status()
        data = r.json()
        if not data: return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        col = "adjClose" if "adjClose" in df.columns else "close"
        daily  = df[col].dropna()
        # Resample to month-end
        monthly = daily.resample("ME").last().dropna()
        return monthly
    except Exception:
        return None

def fetch_btc_monthly(lookback_days=7300):
    """Fetch BTC daily via Tiingo crypto endpoint, resample to monthly."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    # Use daily endpoint — no resampleFreq param for crypto
    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate":   end.strftime("%Y-%m-%d"),
        "token":     TIINGO_KEY
    }
    try:
        r = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            headers=TIINGO_HDR,
            params={"tickers": "btcusd", **params},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        if not data or not data[0].get("priceData"): return None
        rows = data[0]["priceData"]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        col = "close"
        daily   = df[col].dropna()
        monthly = daily.resample("ME").last().dropna()
        print(f"  BTC monthly: {len(monthly)} bars ({monthly.index[0].date()} → {monthly.index[-1].date()})")
        return monthly
    except Exception as e:
        print(f"  BTC fetch failed: {e}")
        return None

def fetch_all_monthly(lookback_days=7300):
    """Fetch monthly bars for all assets."""
    price_data = {}
    for i, ticker in enumerate(TICKERS):
        if ticker == "BTC":
            series = fetch_btc_monthly(lookback_days)
        else:
            series = fetch_tiingo_monthly(ticker, lookback_days)
        if series is not None and len(series) > 20:
            price_data[ticker] = series
            print(f"  {ticker}: {len(series)} months ({series.index[0].date()} → {series.index[-1].date()})")
        else:
            print(f"  WARNING: No data for {ticker}")
        if (i + 1) % 5 == 0:
            time.sleep(1)
    return price_data

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data, backtest_start="2006-01-01"):
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    if not signals:
        print("ERROR: No valid signals — likely rate limit.")
        raise SystemExit(0)

    # Use monthly dates from any available asset
    all_dates = sorted(set().union(*[set(s.index) for s in signals.values()]))
    start_ts  = pd.Timestamp(backtest_start).tz_localize("UTC")
    all_dates = [d for d in all_dates if d >= start_ts]

    capital    = STARTING_CAPITAL
    holdings   = {}   # {ticker: shares}
    cash       = STARTING_CAPITAL
    sgov_val   = 0.0
    equity_curve = []
    trades     = []
    monthly_rets = []
    prev_val   = STARTING_CAPITAL

    for date in all_dates:
        prices_now = {}
        for ticker in TICKERS:
            if ticker in price_data:
                mask = price_data[ticker].index <= date
                if mask.any():
                    prices_now[ticker] = float(price_data[ticker][mask].iloc[-1])

        # Monthly SGOV yield
        sgov_val *= (1 + SGOV_FALLBACK)
        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val
        monthly_rets.append((port_val/prev_val)-1 if prev_val > 0 else 0)
        prev_val  = port_val

        # Monthly signals — equal weight all BUY assets
        buy_tickers = []
        sig_snap    = {}
        for ticker in TICKERS:
            if ticker not in signals: continue
            mask = signals[ticker].index <= date
            if mask.any():
                s = signals[ticker][mask].iloc[-1]
                sig_snap[ticker] = s
                if s == "BUY": buy_tickers.append(ticker)

        if buy_tickers:
            base_w = 1.0 / len(buy_tickers)
            weights = {t: base_w for t in buy_tickers}
            cash_w  = 0.0
        else:
            weights = {}
            cash_w  = 1.0

        # Rebalance
        proceeds = cash + sgov_val
        for ticker, shares in holdings.items():
            p = prices_now.get(ticker, 0)
            proceeds += shares * p
            if shares > 0:
                trades.append({"date": date.strftime("%Y-%m-%d"),
                               "action": "SELL", "ticker": ticker,
                               "price": round(p,2), "value": round(shares*p,2)})

        holdings = {}
        cash     = 0.0
        sgov_val = proceeds * cash_w

        for ticker, w in weights.items():
            alloc = proceeds * w
            p = prices_now.get(ticker, 0)
            if p > 0:
                shares = alloc / p
                holdings[ticker] = shares
                trades.append({"date": date.strftime("%Y-%m-%d"), "action": "BUY",
                               "ticker": ticker, "name": TICKER_META[ticker]["name"],
                               "price": round(p,2), "shares": round(shares,6),
                               "value": round(alloc,2), "weight": round(w*100,2),
                               "signal": sig_snap.get(ticker,"—")})

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "value": round(port_val,2),
                             "n_assets": len(holdings), "cash_pct": round(cash_w*100,1),
                             "buy_tickers": buy_tickers})

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq_vals = [e["value"] for e in equity_curve]
    current_value = eq_vals[-1] if eq_vals else STARTING_CAPITAL
    total_return  = (current_value/STARTING_CAPITAL-1)*100
    n_months = len(eq_vals)
    years    = n_months/12
    cagr     = ((current_value/STARTING_CAPITAL)**(1/max(years,0.1))-1)*100

    arr = np.array(monthly_rets[1:])
    vol = float(np.std(arr)*np.sqrt(12)*100) if len(arr)>1 else 0
    rf_m = SGOV_FALLBACK
    sharpe = float(np.mean(arr-rf_m)/np.std(arr-rf_m)*np.sqrt(12)) if np.std(arr)>0 else 0

    peak, max_dd = STARTING_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak-v)/peak*100
        if dd > max_dd: max_dd = dd

    calmar = cagr/max_dd if max_dd > 0 else 0

    def roll(m):
        if len(eq_vals) < m+1: return None
        return round((eq_vals[-1]/eq_vals[-m]-1)*100, 2)

    yr = datetime.utcnow().year
    ytd_base = next((e["value"] for e in equity_curve if e["date"].startswith(str(yr))), eq_vals[0])
    ytd = round((current_value/ytd_base-1)*100, 2)

    sgov_months = sum(1 for e in equity_curve if e["cash_pct"] > 50)
    pct_in_mkt  = round((n_months-sgov_months)/max(n_months,1)*100, 1)

    # Current holdings
    last_prices = {}
    for ticker in TICKERS:
        if ticker in price_data and len(price_data[ticker]) > 0:
            last_prices[ticker] = float(price_data[ticker].iloc[-1])

    cur_stock_val = sum(holdings.get(t,0)*last_prices.get(t,0) for t in TICKERS)
    current_value = cash + sgov_val + cur_stock_val

    current_holdings = []
    for ticker, shares in holdings.items():
        p = last_prices.get(ticker, 0)
        val = shares * p
        current_holdings.append({
            "ticker": ticker, "name": TICKER_META[ticker]["name"],
            "price": round(p,4), "value": round(val,2),
            "weight": round(val/current_value*100,2) if current_value > 0 else 0,
            "signal": "BUY"
        })
    current_holdings.sort(key=lambda x: -x["value"])

    current_signals = {t: signals[t].iloc[-1] for t in TICKERS if t in signals and len(signals[t])>0}

    return {
        "total_trades":       len([t for t in trades if t["action"]=="BUY"]),
        "current_value":      round(current_value,2),
        "total_return_pct":   round(total_return,2),
        "total_return_dollar":round(current_value-STARTING_CAPITAL,2),
        "cagr_pct":           round(cagr,2),
        "ytd_return_pct":     ytd,
        "return_1m":          roll(1),
        "return_3m":          roll(3),
        "return_6m":          roll(6),
        "return_1y":          roll(12),
        "max_drawdown_pct":   round(max_dd,2),
        "ann_volatility_pct": round(vol,2),
        "sharpe_ratio":       round(sharpe,2),
        "calmar_ratio":       round(calmar,2),
        "n_buy_stocks":       len(holdings),
        "n_universe":         len(TICKERS),
        "cash_pct":           round(sgov_val/current_value*100,1) if current_value>0 else 100,
        "pct_time_in_market": pct_in_mkt,
        "sgov_weeks":         sgov_months,
        "sgov_yield":         5.0,
        "current_holdings":   current_holdings,
        "current_signals":    {t:v for t,v in current_signals.items()},
        "bah_return_pct":     None,
        "bah_equity_curve":   [],
        "equity_curve":       equity_curve,
        "trades":             trades,
        "universe":           UNIVERSE,
        "timeframe":          "monthly",
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Global Macro Rotation Portfolio — Monthly Signals")
    print(f"Universe: {len(TICKERS)} assets | Starting capital: ${STARTING_CAPITAL:,.0f}")

    print("\nFetching monthly bars...")
    price_data = fetch_all_monthly(lookback_days=7300)
    print(f"\n  Got data for {len(price_data)}/{len(TICKERS)} assets")

    # Median backtest start
    min_dates = []
    ticker_starts = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            valid = sig.dropna()
            if len(valid) > 0:
                d = valid.index[0]
                min_dates.append(d)
                ticker_starts[ticker] = d

    if min_dates:
        min_dates.sort()
        threshold_idx = max(0, len(min_dates) // 2 - 1)
        backtest_start = min_dates[threshold_idx].strftime("%Y-%m-%d")
        n_available = sum(1 for d in min_dates if d <= min_dates[threshold_idx])
        print(f"  Backtest start: {backtest_start} ({n_available}/{len(TICKERS)} assets available)")
        late = {t: d.strftime("%Y-%m-%d") for t, d in ticker_starts.items()
                if d > min_dates[threshold_idx]}
        if late:
            print(f"  Late-starting: {', '.join(late.keys())}")
    else:
        backtest_start = "2006-01-01"

    print("\nRunning backtest...")
    result = run_backtest(price_data, backtest_start=backtest_start)

    print(f"\n── Results ─────────────────────────────────────────")
    print(f"  Portfolio:     ${result['current_value']:,.2f}")
    print(f"  Total Return:  {result['total_return_pct']:+.2f}%")
    print(f"  CAGR:          {result['cagr_pct']:+.2f}%")
    print(f"  Sharpe:        {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  -{result['max_drawdown_pct']:.2f}%")
    print(f"  Assets in BUY: {result['n_buy_stocks']}/{result['n_universe']}")
    print(f"  Cash (SGOV):   {result['cash_pct']:.1f}%")
    print(f"\n  Current Holdings:")
    for h in result["current_holdings"][:8]:
        print(f"    {h['ticker']:6s} {h['weight']:5.1f}%  ${h['value']:>10,.2f}")

    print(f"\n  Current Signals:")
    buy  = [t for t,s in result["current_signals"].items() if s=="BUY"]
    sell = [t for t,s in result["current_signals"].items() if s=="SELL"]
    print(f"    BUY  ({len(buy)}):  {', '.join(buy)}")
    print(f"    SELL ({len(sell)}): {', '.join(sell)}")

    # Merge into portfolios.json
    try:
        with open("portfolios.json","r") as f:
            output = json.load(f)
        output["portfolios"] = [p for p in output.get("portfolios",[]) if p["id"] != "macro-rotation"]
    except FileNotFoundError:
        output = {"portfolios": []}

    output["generated"] = datetime.utcnow().isoformat() + "Z"
    output["portfolios"].append({
        "id":               "macro-rotation",
        "name":             "Global Macro Rotation",
        "description":      f"Equal-weight across {len(TICKERS)} global asset classes — equities, bonds, gold, and Bitcoin — using monthly BUY signals. Only holds assets in active monthly uptrends. Rotates to SGOV when assets flip SELL. Rebalances monthly.",
        "ticker":           "IVV",
        "benchmark":        "IVV",
        "cash_instrument":  "SGOV",
        "timeframe":        "monthly",
        "starting_capital": STARTING_CAPITAL,
        "disclaimer":       "Simulated performance. Past performance does not guarantee future results.",
        **result,
    })

    with open("portfolios.json","w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n✓ portfolios.json updated")

if __name__ == "__main__":
    main()
