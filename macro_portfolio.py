2026-04-27T01:09:09.7543924Z Current runner version: '2.333.1'
2026-04-27T01:09:09.7578531Z ##[group]Runner Image Provisioner#!/usr/bin/env python3
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
    """
    Fetch BTC via Tiingo standard daily endpoint (BTCUSD is listed as a regular ticker).
    Resample daily closes to monthly.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate":   end.strftime("%Y-%m-%d"),
        "token":     TIINGO_KEY
    }
    # Try standard daily endpoint first (BTCUSD listed as equity-style ticker)
    for ticker_sym in ["BTCUSD", "BTC"]:
        try:
            r = requests.get(
                f"{TIINGO_URL}/{ticker_sym}/prices",
                headers=TIINGO_HDR,
                params=params,
                timeout=30
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                continue
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date").sort_index()
            col = "adjClose" if "adjClose" in df.columns else "close"
            daily   = df[col].dropna()
            monthly = daily.resample("ME").last().dropna()
            if len(monthly) > 10:
                print(f"  BTC monthly: {len(monthly)} bars ({monthly.index[0].date()} → {monthly.index[-1].date()})")
                return monthly
        except Exception:
            continue

    # Fallback: crypto endpoint with 1day resample
    try:
        r = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            headers=TIINGO_HDR,
            params={"tickers": "btcusd", "resampleFreq": "1Day",
                    "startDate": start.strftime("%Y-%m-%d"),
                    "endDate":   end.strftime("%Y-%m-%d"),
                    "token": TIINGO_KEY},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        if data and data[0].get("priceData"):
            rows = data[0]["priceData"]
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date").sort_index()
            daily   = df["close"].dropna()
            monthly = daily.resample("ME").last().dropna()
            if len(monthly) > 10:
                print(f"  BTC monthly (crypto API): {len(monthly)} bars ({monthly.index[0].date()} → {monthly.index[-1].date()})")
                return monthly
    except Exception as e:
        print(f"  BTC crypto fallback failed: {e}")

    print("  BTC: no data available from any endpoint")
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

        # Track actual signal changes for trade logging (tax-aware display)
        prev_buy_set = set(holdings.keys())
        new_buy_set  = set(weights.keys())
        entered = new_buy_set - prev_buy_set   # flipped to BUY
        exited  = prev_buy_set - new_buy_set   # flipped to SELL

        # Full rebalance internally (correct backtest math)
        proceeds = cash + sgov_val
        for ticker, shares in holdings.items():
            p = prices_now.get(ticker, 0)
            proceeds += shares * p
            if ticker in exited:
                trades.append({"date": date.strftime("%Y-%m-%d"),
                               "action": "SELL", "ticker": ticker,
                               "name": TICKER_META[ticker]["name"],
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
                if ticker in entered:
                    trades.append({"date": date.strftime("%Y-%m-%d"), "action": "BUY",
                                   "ticker": ticker, "name": TICKER_META[ticker]["name"],
                                   "price": round(p,4), "shares": round(shares,6),
                                   "value": round(alloc,2), "weight": round(w*100,2),
                                   "signal": sig_snap.get(ticker,"—")})

        # Log HOLD when nothing changed
        if not entered and not exited and new_buy_set:
            trades.append({"date": date.strftime("%Y-%m-%d"), "action": "HOLD",
                           "tickers": sorted(list(new_buy_set)),
                           "n": len(new_buy_set),
                           "note": f"No changes — {len(new_buy_set)} assets held"})

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

2026-04-27T01:09:09.7579797Z Hosted Compute Agent
2026-04-27T01:09:09.7580734Z Version: 20260213.493
2026-04-27T01:09:09.7581781Z Commit: 5c115507f6dd24b8de37d8bbe0bb4509d0cc0fa3
2026-04-27T01:09:09.7583324Z Build Date: 2026-02-13T00:28:41Z
2026-04-27T01:09:09.7584511Z Worker ID: {c5865721-9d60-433a-8b56-4535f525a16f}
2026-04-27T01:09:09.7585888Z Azure Region: centralus
2026-04-27T01:09:09.7586901Z ##[endgroup]
2026-04-27T01:09:09.7589195Z ##[group]Operating System
2026-04-27T01:09:09.7590319Z Ubuntu
2026-04-27T01:09:09.7591151Z 24.04.4
2026-04-27T01:09:09.7592069Z LTS
2026-04-27T01:09:09.7593196Z ##[endgroup]
2026-04-27T01:09:09.7594208Z ##[group]Runner Image
2026-04-27T01:09:09.7595223Z Image: ubuntu-24.04
2026-04-27T01:09:09.7596223Z Version: 20260413.86.1
2026-04-27T01:09:09.7598408Z Included Software: https://github.com/actions/runner-images/blob/ubuntu24/20260413.86/images/ubuntu/Ubuntu2404-Readme.md
2026-04-27T01:09:09.7601240Z Image Release: https://github.com/actions/runner-images/releases/tag/ubuntu24%2F20260413.86
2026-04-27T01:09:09.7603313Z ##[endgroup]
2026-04-27T01:09:09.7608464Z ##[group]GITHUB_TOKEN Permissions
2026-04-27T01:09:09.7611623Z Actions: write
2026-04-27T01:09:09.7612963Z ArtifactMetadata: write
2026-04-27T01:09:09.7614039Z Attestations: write
2026-04-27T01:09:09.7615123Z Checks: write
2026-04-27T01:09:09.7615978Z Contents: write
2026-04-27T01:09:09.7616952Z Deployments: write
2026-04-27T01:09:09.7617910Z Discussions: write
2026-04-27T01:09:09.7618906Z Issues: write
2026-04-27T01:09:09.7619994Z Metadata: read
2026-04-27T01:09:09.7620835Z Models: read
2026-04-27T01:09:09.7621682Z Packages: write
2026-04-27T01:09:09.7622904Z Pages: write
2026-04-27T01:09:09.7623888Z PullRequests: write
2026-04-27T01:09:09.7624957Z RepositoryProjects: write
2026-04-27T01:09:09.7626104Z SecurityEvents: write
2026-04-27T01:09:09.7627021Z Statuses: write
2026-04-27T01:09:09.7628001Z VulnerabilityAlerts: read
2026-04-27T01:09:09.7629064Z ##[endgroup]
2026-04-27T01:09:09.7631956Z Secret source: Actions
2026-04-27T01:09:09.7633779Z Prepare workflow directory
2026-04-27T01:09:09.8043105Z Prepare all required actions
2026-04-27T01:09:09.8099300Z Getting action download info
2026-04-27T01:09:10.1705142Z Download action repository 'actions/checkout@v4' (SHA:34e114876b0b11c390a56381ad16ebd13914f8d5)
2026-04-27T01:09:10.2862074Z Download action repository 'actions/setup-python@v5' (SHA:a26af69be951a213d495a4c3e4e4022e16d87065)
2026-04-27T01:09:10.4734446Z Complete job name: update-portfolio
2026-04-27T01:09:10.5496003Z ##[group]Run actions/checkout@v4
2026-04-27T01:09:10.5496913Z with:
2026-04-27T01:09:10.5497358Z   repository: callingmarkets/signals
2026-04-27T01:09:10.5498082Z   token: ***
2026-04-27T01:09:10.5498518Z   ssh-strict: true
2026-04-27T01:09:10.5498941Z   ssh-user: git
2026-04-27T01:09:10.5499379Z   persist-credentials: true
2026-04-27T01:09:10.5499863Z   clean: true
2026-04-27T01:09:10.5500302Z   sparse-checkout-cone-mode: true
2026-04-27T01:09:10.5500839Z   fetch-depth: 1
2026-04-27T01:09:10.5501266Z   fetch-tags: false
2026-04-27T01:09:10.5501699Z   show-progress: true
2026-04-27T01:09:10.5502352Z   lfs: false
2026-04-27T01:09:10.5502765Z   submodules: false
2026-04-27T01:09:10.5503194Z   set-safe-directory: true
2026-04-27T01:09:10.5503955Z ##[endgroup]
2026-04-27T01:09:10.6592942Z Syncing repository: callingmarkets/signals
2026-04-27T01:09:10.6594952Z ##[group]Getting Git version info
2026-04-27T01:09:10.6595677Z Working directory is '/home/runner/work/signals/signals'
2026-04-27T01:09:10.6596773Z [command]/usr/bin/git version
2026-04-27T01:09:10.6659450Z git version 2.53.0
2026-04-27T01:09:10.6686732Z ##[endgroup]
2026-04-27T01:09:10.6702484Z Temporarily overriding HOME='/home/runner/work/_temp/967dce9f-2efe-4788-9cc5-ea46872c88a8' before making global git config changes
2026-04-27T01:09:10.6704074Z Adding repository directory to the temporary git global config as a safe directory
2026-04-27T01:09:10.6716335Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/signals/signals
2026-04-27T01:09:10.6749327Z Deleting the contents of '/home/runner/work/signals/signals'
2026-04-27T01:09:10.6753250Z ##[group]Initializing the repository
2026-04-27T01:09:10.6757433Z [command]/usr/bin/git init /home/runner/work/signals/signals
2026-04-27T01:09:10.6870009Z hint: Using 'master' as the name for the initial branch. This default branch name
2026-04-27T01:09:10.6872113Z hint: will change to "main" in Git 3.0. To configure the initial branch name
2026-04-27T01:09:10.6873648Z hint: to use in all of your new repositories, which will suppress this warning,
2026-04-27T01:09:10.6874616Z hint: call:
2026-04-27T01:09:10.6875032Z hint:
2026-04-27T01:09:10.6875557Z hint: 	git config --global init.defaultBranch <name>
2026-04-27T01:09:10.6876194Z hint:
2026-04-27T01:09:10.6877202Z hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
2026-04-27T01:09:10.6878476Z hint: 'development'. The just-created branch can be renamed via this command:
2026-04-27T01:09:10.6879277Z hint:
2026-04-27T01:09:10.6879673Z hint: 	git branch -m <name>
2026-04-27T01:09:10.6880144Z hint:
2026-04-27T01:09:10.6880763Z hint: Disable this message with "git config set advice.defaultBranchName false"
2026-04-27T01:09:10.6881778Z Initialized empty Git repository in /home/runner/work/signals/signals/.git/
2026-04-27T01:09:10.6884248Z [command]/usr/bin/git remote add origin https://github.com/callingmarkets/signals
2026-04-27T01:09:10.6913734Z ##[endgroup]
2026-04-27T01:09:10.6914570Z ##[group]Disabling automatic garbage collection
2026-04-27T01:09:10.6918042Z [command]/usr/bin/git config --local gc.auto 0
2026-04-27T01:09:10.6946071Z ##[endgroup]
2026-04-27T01:09:10.6946798Z ##[group]Setting up auth
2026-04-27T01:09:10.6953521Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
2026-04-27T01:09:10.6982618Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
2026-04-27T01:09:10.7305975Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
2026-04-27T01:09:10.7336904Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
2026-04-27T01:09:10.7550833Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
2026-04-27T01:09:10.7582508Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
2026-04-27T01:09:10.7837257Z [command]/usr/bin/git config --local http.https://github.com/.extraheader AUTHORIZATION: basic ***
2026-04-27T01:09:10.7875848Z ##[endgroup]
2026-04-27T01:09:10.7877342Z ##[group]Fetching the repository
2026-04-27T01:09:10.7887199Z [command]/usr/bin/git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin +04f5a24277b9d66bab1c3c8c070ac7b6dae0b162:refs/remotes/origin/main
2026-04-27T01:09:11.1680772Z From https://github.com/callingmarkets/signals
2026-04-27T01:09:11.1681783Z  * [new ref]         04f5a24277b9d66bab1c3c8c070ac7b6dae0b162 -> origin/main
2026-04-27T01:09:11.1708586Z ##[endgroup]
2026-04-27T01:09:11.1709505Z ##[group]Determining the checkout info
2026-04-27T01:09:11.1710853Z ##[endgroup]
2026-04-27T01:09:11.1715932Z [command]/usr/bin/git sparse-checkout disable
2026-04-27T01:09:11.1751186Z [command]/usr/bin/git config --local --unset-all extensions.worktreeConfig
2026-04-27T01:09:11.1776664Z ##[group]Checking out the ref
2026-04-27T01:09:11.1780062Z [command]/usr/bin/git checkout --progress --force -B main refs/remotes/origin/main
2026-04-27T01:09:11.1880964Z Switched to a new branch 'main'
2026-04-27T01:09:11.1884059Z branch 'main' set up to track 'origin/main'.
2026-04-27T01:09:11.1891046Z ##[endgroup]
2026-04-27T01:09:11.1925085Z [command]/usr/bin/git log -1 --format=%H
2026-04-27T01:09:11.1946838Z 04f5a24277b9d66bab1c3c8c070ac7b6dae0b162
2026-04-27T01:09:11.2203336Z ##[group]Run actions/setup-python@v5
2026-04-27T01:09:11.2203992Z with:
2026-04-27T01:09:11.2204397Z   python-version: 3.11
2026-04-27T01:09:11.2204865Z   check-latest: false
2026-04-27T01:09:11.2205473Z   token: ***
2026-04-27T01:09:11.2205896Z   update-environment: true
2026-04-27T01:09:11.2206408Z   allow-prereleases: false
2026-04-27T01:09:11.2206892Z   freethreaded: false
2026-04-27T01:09:11.2207343Z ##[endgroup]
2026-04-27T01:09:11.3967977Z ##[group]Installed versions
2026-04-27T01:09:11.4075325Z Successfully set up CPython (3.11.15)
2026-04-27T01:09:11.4077251Z ##[endgroup]
2026-04-27T01:09:11.4228368Z ##[group]Run pip install requests pandas numpy
2026-04-27T01:09:11.4230025Z [36;1mpip install requests pandas numpy[0m
2026-04-27T01:09:11.4272900Z shell: /usr/bin/bash -e {0}
2026-04-27T01:09:11.4274132Z env:
2026-04-27T01:09:11.4275321Z   pythonLocation: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:11.4277368Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib/pkgconfig
2026-04-27T01:09:11.4279319Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:11.4281042Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:11.4283175Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:11.4284977Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib
2026-04-27T01:09:11.4286470Z ##[endgroup]
2026-04-27T01:09:12.2580728Z Collecting requests
2026-04-27T01:09:12.3348910Z   Downloading requests-2.33.1-py3-none-any.whl.metadata (4.8 kB)
2026-04-27T01:09:12.4950817Z Collecting pandas
2026-04-27T01:09:12.5082711Z   Downloading pandas-3.0.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
2026-04-27T01:09:12.7627386Z Collecting numpy
2026-04-27T01:09:12.7754886Z   Downloading numpy-2.4.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
2026-04-27T01:09:12.8841932Z Collecting charset_normalizer<4,>=2 (from requests)
2026-04-27T01:09:12.8960709Z   Downloading charset_normalizer-3.4.7-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
2026-04-27T01:09:12.9271837Z Collecting idna<4,>=2.5 (from requests)
2026-04-27T01:09:12.9385891Z   Downloading idna-3.13-py3-none-any.whl.metadata (8.0 kB)
2026-04-27T01:09:12.9722873Z Collecting urllib3<3,>=1.26 (from requests)
2026-04-27T01:09:12.9837646Z   Downloading urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
2026-04-27T01:09:13.0145328Z Collecting certifi>=2023.5.7 (from requests)
2026-04-27T01:09:13.0261013Z   Downloading certifi-2026.4.22-py3-none-any.whl.metadata (2.5 kB)
2026-04-27T01:09:13.0566329Z Collecting python-dateutil>=2.8.2 (from pandas)
2026-04-27T01:09:13.0676698Z   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
2026-04-27T01:09:13.0954641Z Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas)
2026-04-27T01:09:13.1094298Z   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
2026-04-27T01:09:13.1248027Z Downloading requests-2.33.1-py3-none-any.whl (64 kB)
2026-04-27T01:09:13.1436760Z Downloading charset_normalizer-3.4.7-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (214 kB)
2026-04-27T01:09:13.1834079Z Downloading idna-3.13-py3-none-any.whl (68 kB)
2026-04-27T01:09:13.2048945Z Downloading urllib3-2.6.3-py3-none-any.whl (131 kB)
2026-04-27T01:09:13.2358644Z Downloading pandas-3.0.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (11.3 MB)
2026-04-27T01:09:13.6349286Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.3/11.3 MB 29.7 MB/s  0:00:00
2026-04-27T01:09:13.6498808Z Downloading numpy-2.4.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.9 MB)
2026-04-27T01:09:13.8249515Z    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.9/16.9 MB 97.5 MB/s  0:00:00
2026-04-27T01:09:13.8368904Z Downloading certifi-2026.4.22-py3-none-any.whl (135 kB)
2026-04-27T01:09:13.8512837Z Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
2026-04-27T01:09:13.8652297Z Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
2026-04-27T01:09:13.9743928Z Installing collected packages: urllib3, six, numpy, idna, charset_normalizer, certifi, requests, python-dateutil, pandas
2026-04-27T01:09:20.2525462Z 
2026-04-27T01:09:20.2538292Z Successfully installed certifi-2026.4.22 charset_normalizer-3.4.7 idna-3.13 numpy-2.4.4 pandas-3.0.2 python-dateutil-2.9.0.post0 requests-2.33.1 six-1.17.0 urllib3-2.6.3
2026-04-27T01:09:20.4091464Z 
2026-04-27T01:09:20.4092977Z [notice] A new release of pip is available: 26.0.1 -> 26.1
2026-04-27T01:09:20.4094571Z [notice] To update, run: pip install --upgrade pip
2026-04-27T01:09:20.5033302Z ##[group]Run python macro_portfolio.py
2026-04-27T01:09:20.5033667Z [36;1mpython macro_portfolio.py[0m
2026-04-27T01:09:20.5056452Z shell: /usr/bin/bash -e {0}
2026-04-27T01:09:20.5056695Z env:
2026-04-27T01:09:20.5056968Z   pythonLocation: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:20.5057400Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib/pkgconfig
2026-04-27T01:09:20.5057836Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:20.5058214Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:20.5058590Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:20.5058967Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib
2026-04-27T01:09:20.5059569Z   TIINGO_API_KEY: ***
2026-04-27T01:09:20.5059773Z ##[endgroup]
2026-04-27T01:09:35.1283833Z Global Macro Rotation Portfolio — Monthly Signals
2026-04-27T01:09:35.1284747Z Universe: 16 assets | Starting capital: $100,000
2026-04-27T01:09:35.1285275Z 
2026-04-27T01:09:35.1285456Z Fetching monthly bars...
2026-04-27T01:09:35.1286092Z   IVV: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1286806Z   IWF: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1287543Z   IJH: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1288193Z   IJR: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1288903Z   IWM: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1289614Z   IWD: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1290302Z   IVW: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1291009Z   IVE: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1291706Z   IEFA: 163 months (2012-10-31 → 2026-04-30)
2026-04-27T01:09:35.1292679Z   IEMG: 163 months (2012-10-31 → 2026-04-30)
2026-04-27T01:09:35.1293441Z   EFA: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1294131Z   AGG: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1294845Z   IEF: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1295530Z   TLT: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1296201Z   IAU: 240 months (2006-05-31 → 2026-04-30)
2026-04-27T01:09:35.1296931Z   BTC monthly: 177 bars (2011-08-31 → 2026-04-30)
2026-04-27T01:09:35.1297682Z   BTC: 177 months (2011-08-31 → 2026-04-30)
2026-04-27T01:09:35.1298082Z 
2026-04-27T01:09:35.1298279Z   Got data for 16/16 assets
2026-04-27T01:09:35.1298876Z   Backtest start: 2006-05-31 (13/16 assets available)
2026-04-27T01:09:35.1299612Z   Late-starting: IEFA, IEMG, BTC
2026-04-27T01:09:35.1299981Z 
2026-04-27T01:09:35.1300159Z Running backtest...
2026-04-27T01:09:35.1300434Z 
2026-04-27T01:09:35.1300843Z ── Results ─────────────────────────────────────────
2026-04-27T01:09:35.1301492Z   Portfolio:     $348,367.17
2026-04-27T01:09:35.1301887Z   Total Return:  +248.37%
2026-04-27T01:09:35.1302419Z   CAGR:          +6.44%
2026-04-27T01:09:35.1302764Z   Sharpe:        0.15
2026-04-27T01:09:35.1303165Z   Max Drawdown:  -39.74%
2026-04-27T01:09:35.1303406Z   Assets in BUY: 12/16
2026-04-27T01:09:35.1303614Z   Cash (SGOV):   0.3%
2026-04-27T01:09:35.1303736Z 
2026-04-27T01:09:35.1303818Z   Current Holdings:
2026-04-27T01:09:35.1304006Z     IEMG     9.9%  $ 34,531.06
2026-04-27T01:09:35.1304499Z     IWM      9.7%  $ 33,679.94
2026-04-27T01:09:35.1304709Z     IWD      8.9%  $ 30,949.42
2026-04-27T01:09:35.1304917Z     IEFA     8.7%  $ 30,207.58
2026-04-27T01:09:35.1305123Z     IAU      8.5%  $ 29,747.55
2026-04-27T01:09:35.1305321Z     IJH      8.2%  $ 28,520.12
2026-04-27T01:09:35.1305524Z     IVV      8.0%  $ 27,801.81
2026-04-27T01:09:35.1305725Z     IJR      8.0%  $ 27,760.48
2026-04-27T01:09:35.1305857Z 
2026-04-27T01:09:35.1305933Z   Current Signals:
2026-04-27T01:09:35.1306229Z     BUY  (14):  IVV, IJH, IJR, IWM, IWD, IVW, IVE, IEFA, IEMG, EFA, AGG, IEF, TLT, IAU
2026-04-27T01:09:35.1306592Z     SELL (2): IWF, BTC
2026-04-27T01:09:35.1306721Z 
2026-04-27T01:09:35.1306872Z ✓ portfolios.json updated
2026-04-27T01:09:35.1943429Z ##[group]Run git config user.name  "github-actions[bot]"
2026-04-27T01:09:35.1943898Z [36;1mgit config user.name  "github-actions[bot]"[0m
2026-04-27T01:09:35.1944338Z [36;1mgit config user.email "github-actions[bot]@users.noreply.github.com"[0m
2026-04-27T01:09:35.1944763Z [36;1mgit add portfolios.json[0m
2026-04-27T01:09:35.1945189Z [36;1mgit diff --cached --quiet || git commit -m "chore: update macro portfolio [skip ci]"[0m
2026-04-27T01:09:35.1945612Z [36;1mgit push[0m
2026-04-27T01:09:35.1967538Z shell: /usr/bin/bash -e {0}
2026-04-27T01:09:35.1967780Z env:
2026-04-27T01:09:35.1968036Z   pythonLocation: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:35.1968471Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib/pkgconfig
2026-04-27T01:09:35.1968880Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:35.1969249Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:35.1969637Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.15/x64
2026-04-27T01:09:35.1969997Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.11.15/x64/lib
2026-04-27T01:09:35.1970311Z ##[endgroup]
2026-04-27T01:09:35.2748779Z [main ae43104] chore: update macro portfolio [skip ci]
2026-04-27T01:09:35.2749452Z  1 file changed, 2407 insertions(+), 2172 deletions(-)
2026-04-27T01:09:36.3828674Z To https://github.com/callingmarkets/signals
2026-04-27T01:09:36.3829174Z    04f5a24..ae43104  main -> main
2026-04-27T01:09:36.3938511Z Post job cleanup.
2026-04-27T01:09:36.5676886Z Post job cleanup.
2026-04-27T01:09:36.6624237Z [command]/usr/bin/git version
2026-04-27T01:09:36.6660349Z git version 2.53.0
2026-04-27T01:09:36.6705411Z Temporarily overriding HOME='/home/runner/work/_temp/e7c64187-e19d-4992-a677-b6266d035398' before making global git config changes
2026-04-27T01:09:36.6706969Z Adding repository directory to the temporary git global config as a safe directory
2026-04-27T01:09:36.6711035Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/signals/signals
2026-04-27T01:09:36.6756373Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
2026-04-27T01:09:36.6788977Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
2026-04-27T01:09:36.7019130Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
2026-04-27T01:09:36.7040661Z http.https://github.com/.extraheader
2026-04-27T01:09:36.7053162Z [command]/usr/bin/git config --local --unset-all http.https://github.com/.extraheader
2026-04-27T01:09:36.7083562Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
2026-04-27T01:09:36.7306405Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
2026-04-27T01:09:36.7338837Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
2026-04-27T01:09:36.7678050Z Cleaning up orphan processes
2026-04-27T01:09:36.7951047Z ##[warning]Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, actions/setup-python@v5. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner on September 16th, 2026. Please check if updated versions of these actions are available that support Node.js 24. To opt into Node.js 24 now, set the FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true environment variable on the runner or in your workflow file. Once Node.js 24 becomes the default, you can temporarily opt out by setting ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true. For more information see: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/
