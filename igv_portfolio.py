#!/usr/bin/env python3
"""
Tech Software Alpha Portfolio (vs IGV)
- Universe: Top 40 IGV holdings (established software companies)
- Signal: Same 2-of-3 weekly momentum (EMA20>EMA55, RSI14>RSI_MA, MACD>Signal)
- Entry: Weekly BUY signal
- Weighting: Equal weight among BUY stocks, max 15% per position
- Cash: SGOV (~5% yield) for unfilled slots
- Benchmark: IGV buy-and-hold comparison
- Rebalance: Every Monday
"""

import json, os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests

TIINGO_KEY  = os.environ["TIINGO_API_KEY"]
TIINGO_HDR  = {"Authorization": f"Token {TIINGO_KEY}", "Content-Type": "application/json"}
TIINGO_URL  = "https://api.tiingo.com/tiingo/daily"

STARTING_CAPITAL = 100_000.0
MAX_POSITION     = 0.15   # 15% cap per stock
SGOV_FALLBACK    = 0.05 / 52

EMA_FAST  = 20; EMA_SLOW  = 55; RSI_LEN = 14
RSI_MA    = 14; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIG = 9

# ── IGV Top 40 universe ───────────────────────────────────────────────────────
# Top holdings by weight — established, revenue-generating software companies
IGV_UNIVERSE = [
    {"ticker": "ORCL",  "name": "Oracle Corporation"},
    {"ticker": "MSFT",  "name": "Microsoft Corporation"},
    {"ticker": "PLTR",  "name": "Palantir Technologies"},
    {"ticker": "CRM",   "name": "Salesforce Inc."},
    {"ticker": "PANW",  "name": "Palo Alto Networks"},
    {"ticker": "ADBE",  "name": "Adobe Inc."},
    {"ticker": "NOW",   "name": "ServiceNow Inc."},
    {"ticker": "INTU",  "name": "Intuit Inc."},
    {"ticker": "CDNS",  "name": "Cadence Design Systems"},
    {"ticker": "SNPS",  "name": "Synopsys Inc."},
    {"ticker": "FTNT",  "name": "Fortinet Inc."},
    {"ticker": "ANSS",  "name": "ANSYS Inc."},
    {"ticker": "TEAM",  "name": "Atlassian Corporation"},
    {"ticker": "WDAY",  "name": "Workday Inc."},
    {"ticker": "CRWD",  "name": "CrowdStrike Holdings"},
    {"ticker": "VEEV",  "name": "Veeva Systems"},
    {"ticker": "DDOG",  "name": "Datadog Inc."},
    {"ticker": "HUBS",  "name": "HubSpot Inc."},
    {"ticker": "DOCN",  "name": "DigitalOcean Holdings"},
    {"ticker": "PAYC",  "name": "Paycom Software"},
    {"ticker": "CSGP",  "name": "CoStar Group"},
    {"ticker": "ZS",    "name": "Zscaler Inc."},
    {"ticker": "GWRE",  "name": "Guidewire Software"},
    {"ticker": "MANH",  "name": "Manhattan Associates"},
    {"ticker": "PCTY",  "name": "Paylocity Holding"},
    {"ticker": "MTTR",  "name": "Matterport Inc."},
    {"ticker": "BSY",   "name": "Bentley Systems"},
    {"ticker": "DOCU",  "name": "DocuSign Inc."},
    {"ticker": "FICO",  "name": "Fair Isaac Corporation"},
    {"ticker": "GEN",   "name": "Gen Digital Inc."},
    {"ticker": "VRSK",  "name": "Verisk Analytics"},
    {"ticker": "IBM",   "name": "IBM Corporation"},
    {"ticker": "GDDY",  "name": "GoDaddy Inc."},
    {"ticker": "ACN",   "name": "Accenture plc"},
    {"ticker": "PAYX",  "name": "Paychex Inc."},
    {"ticker": "BR",    "name": "Broadridge Financial"},
    {"ticker": "SSNC",  "name": "SS&C Technologies"},
    {"ticker": "VRSN",  "name": "VeriSign Inc."},
    {"ticker": "GRMN",  "name": "Garmin Ltd."},
    {"ticker": "APP",   "name": "AppLovin Corporation"},
]
TICKERS = [s["ticker"] for s in IGV_UNIVERSE]
TICKER_META = {s["ticker"]: s for s in IGV_UNIVERSE}

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_rma(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def calc_rsi(s, n):
    d = s.diff()
    ag = calc_rma(d.clip(lower=0), n)
    al = calc_rma((-d).clip(lower=0), n)
    return 100 - 100/(1 + ag/al.replace(0, np.nan))

def compute_signal(src):
    if len(src) < EMA_SLOW + 10: return None
    ema20 = calc_ema(src, EMA_FAST); ema55 = calc_ema(src, EMA_SLOW)
    rsi   = calc_rsi(src, RSI_LEN);  rma   = calc_ema(rsi, RSI_MA)
    macd  = calc_ema(src, MACD_FAST) - calc_ema(src, MACD_SLOW)
    sig   = calc_ema(macd, MACD_SIG)
    score = (ema20>ema55).astype(int) + (rsi>rma).astype(int) + (macd>sig).astype(int)
    return score.apply(lambda s: "BUY" if s >= 2 else "SELL")

# ── Fetch (Tiingo) ────────────────────────────────────────────────────────────
def fetch_tiingo_weekly(ticker, lookback_days=7300):
    """Fetch daily prices from Tiingo and resample to weekly."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate":   end.strftime("%Y-%m-%d"),
        "resampleFreq": "weekly",
        "token": TIINGO_KEY,
    }
    try:
        r = requests.get(f"{TIINGO_URL}/{ticker}/prices",
                         headers=TIINGO_HDR, params=params, timeout=30)
        if r.status_code == 404:
            return None  # ticker not found
        r.raise_for_status()
        data = r.json()
        if not data: return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        # Use adjClose for split/dividend adjustment
        col = "adjClose" if "adjClose" in df.columns else "close"
        return df[col].dropna()
    except Exception as e:
        return None

def fetch_weekly_stocks(tickers, lookback_days=7300):
    """Fetch weekly bars for all tickers via Tiingo."""
    all_data = {}
    for ticker in tickers:
        series = fetch_tiingo_weekly(ticker, lookback_days)
        if series is not None and len(series) > 20:
            all_data[ticker] = series
        else:
            print(f"  WARNING: No data for {ticker}")
    return all_data

def fetch_igv_weekly(lookback_days=7300):
    """Fetch IGV ETF weekly prices for benchmark."""
    series = fetch_tiingo_weekly("IGV", lookback_days)
    if series is not None:
        print(f"  IGV: {len(series)} weeks ({series.index[0].date()} → {series.index[-1].date()})")
    return series

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data, igv_prices, backtest_start="2021-01-01"):
    # Compute signals
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    if not signals:
        raise ValueError("No valid signals computed")

    # Common weekly dates from backtest start
    all_dates = sorted(set().union(*[set(s.index) for s in signals.values()]))
    start_ts  = pd.Timestamp(backtest_start).tz_localize("UTC")
    all_dates = [d for d in all_dates if d >= start_ts]

    capital      = STARTING_CAPITAL
    holdings     = {}   # {ticker: shares}
    cash         = STARTING_CAPITAL
    sgov_val     = 0.0
    equity_curve = []
    trades       = []
    weekly_rets  = []
    prev_val     = STARTING_CAPITAL

    for date in all_dates:
        # Current prices
        prices_now = {}
        for ticker in TICKERS:
            if ticker in price_data:
                mask = price_data[ticker].index <= date
                if mask.any():
                    prices_now[ticker] = float(price_data[ticker][mask].iloc[-1])

        sgov_val *= (1 + SGOV_FALLBACK)

        stock_val = sum(holdings.get(t, 0) * prices_now.get(t, 0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val

        weekly_rets.append((port_val / prev_val) - 1 if prev_val > 0 else 0)
        prev_val = port_val

        # Determine BUY stocks this week
        buy_tickers = []
        sig_snap    = {}
        for ticker in TICKERS:
            if ticker not in signals: continue
            mask = signals[ticker].index <= date
            if mask.any():
                s = signals[ticker][mask].iloc[-1]
                sig_snap[ticker] = s
                if s == "BUY": buy_tickers.append(ticker)

        # Equal weight with 15% cap
        if buy_tickers:
            base_w = 1.0 / len(buy_tickers)
            weights = {t: min(base_w, MAX_POSITION) for t in buy_tickers}
            total_w = sum(weights.values())
            # Renormalize if any were capped
            if total_w < 0.9999:
                weights = {t: w/total_w for t, w in weights.items()}
            cash_w = max(0.0, 1.0 - sum(weights.values()))
        else:
            weights = {}
            cash_w  = 1.0

        # Sell everything
        proceeds = cash + sgov_val
        for ticker, shares in holdings.items():
            p = prices_now.get(ticker, 0)
            proceeds += shares * p
            if shares > 0:
                trades.append({"date": date.strftime("%Y-%m-%d"),
                               "action": "SELL", "ticker": ticker,
                               "price": round(p, 2), "value": round(shares*p, 2)})

        holdings = {}
        cash     = 0.0
        sgov_val = proceeds * cash_w

        for ticker, w in weights.items():
            alloc  = proceeds * w
            p      = prices_now.get(ticker, 0)
            if p > 0:
                shares = alloc / p
                holdings[ticker] = shares
                trades.append({"date": date.strftime("%Y-%m-%d"), "action": "BUY",
                               "ticker": ticker, "name": TICKER_META[ticker]["name"],
                               "price": round(p, 2), "shares": round(shares, 4),
                               "value": round(alloc, 2), "weight": round(w*100, 2),
                               "signal": sig_snap.get(ticker, "—")})

        stock_val = sum(holdings.get(t, 0) * prices_now.get(t, 0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "value": round(port_val, 2),
                              "n_stocks": len(holdings), "cash_pct": round(cash_w*100, 1),
                              "buy_tickers": buy_tickers})

    # ── Performance metrics ──────────────────────────────────────────────────
    eq_vals = [e["value"] for e in equity_curve]
    current_value = eq_vals[-1] if eq_vals else STARTING_CAPITAL
    total_return  = (current_value / STARTING_CAPITAL - 1) * 100
    n_weeks       = len(eq_vals)
    years         = n_weeks / 52
    cagr          = ((current_value/STARTING_CAPITAL)**(1/max(years,0.1)) - 1) * 100

    arr  = np.array(weekly_rets[1:])  # skip first
    vol  = float(np.std(arr)*np.sqrt(52)*100) if len(arr)>1 else 0
    rf_w = SGOV_FALLBACK
    sharpe = float(np.mean(arr-rf_w)/np.std(arr-rf_w)*np.sqrt(52)) if np.std(arr)>0 else 0

    peak, max_dd = STARTING_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak - v)/peak * 100
        if dd > max_dd: max_dd = dd

    calmar = cagr/max_dd if max_dd > 0 else 0

    def roll(w):
        if len(eq_vals) < w+1: return None
        return round((eq_vals[-1]/eq_vals[-w]-1)*100, 2)

    yr = datetime.utcnow().year
    ytd_base = next((e["value"] for e in equity_curve if e["date"].startswith(str(yr))), eq_vals[0])
    ytd = round((current_value/ytd_base-1)*100, 2)

    sgov_weeks = sum(1 for e in equity_curve if e["cash_pct"] > 50)
    stock_weeks = n_weeks - sgov_weeks
    pct_in_mkt = round(stock_weeks/max(n_weeks,1)*100, 1)

    # Current holdings snapshot
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
            "price": round(p, 2), "value": round(val, 2),
            "weight": round(val/current_value*100, 2) if current_value > 0 else 0,
            "signal": "BUY"
        })
    current_holdings.sort(key=lambda x: -x["value"])

    current_signals = {}
    for ticker in TICKERS:
        if ticker in signals and len(signals[ticker]) > 0:
            current_signals[ticker] = signals[ticker].iloc[-1]

    # ── IGV Buy & Hold benchmark ──────────────────────────────────────────────
    bah_curve, bah_return = [], None
    if igv_prices is not None and len(igv_prices) > 0:
        # Find first trade date
        first_buy = next((t for t in trades if t["action"]=="BUY"), None)
        if first_buy:
            fb_ts = pd.Timestamp(first_buy["date"]).tz_localize("UTC")
            mask  = igv_prices.index >= fb_ts
            if mask.any():
                igv_slice  = igv_prices[mask]
                igv_start  = float(igv_slice.iloc[0])
                bah_curve  = [{"date": d.strftime("%Y-%m-%d"),
                               "value": round(STARTING_CAPITAL*(float(v)/igv_start), 2)}
                              for d, v in igv_slice.items()]
                bah_current = round(STARTING_CAPITAL*(float(igv_prices.iloc[-1])/igv_start), 2)
                bah_return  = round((bah_current/STARTING_CAPITAL-1)*100, 2)

    return {
        "current_value":      round(current_value, 2),
        "total_return_pct":   round(total_return, 2),
        "total_return_dollar":round(current_value - STARTING_CAPITAL, 2),
        "cagr_pct":           round(cagr, 2),
        "ytd_return_pct":     ytd,
        "return_1m":          roll(4),
        "return_3m":          roll(13),
        "return_6m":          roll(26),
        "return_1y":          roll(52),
        "max_drawdown_pct":   round(max_dd, 2),
        "ann_volatility_pct": round(vol, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "calmar_ratio":       round(calmar, 2),
        "total_trades":       len([t for t in trades if t["action"] == "BUY"]),
        "n_buy_stocks":       len(holdings),
        "n_universe":         len(TICKERS),
        "cash_pct":           round(sgov_val/current_value*100, 1) if current_value > 0 else 100,
        "pct_time_in_market": pct_in_mkt,
        "sgov_weeks":         sgov_weeks,
        "sgov_yield":         5.0,
        "current_holdings":   current_holdings,
        "current_signals":    {t: v for t, v in current_signals.items()},
        "bah_return_pct":     bah_return,
        "bah_equity_curve":   bah_curve,
        "equity_curve":       equity_curve,
        "trades":             trades[-60:],
        "universe":           IGV_UNIVERSE,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Tech Software Alpha Portfolio Engine (vs IGV) — Tiingo data")
    print(f"Universe: {len(TICKERS)} stocks | Starting capital: ${STARTING_CAPITAL:,.0f}")

    print("\nFetching weekly bars...")
    price_data = fetch_weekly_stocks(TICKERS, lookback_days=7300)
    print(f"  Got data for {len(price_data)}/{len(TICKERS)} tickers")

    print("Fetching IGV benchmark...")
    igv_prices = fetch_igv_weekly(lookback_days=7300)
    print(f"  IGV: {len(igv_prices) if igv_prices is not None else 0} weeks")

    # Dynamic start — first date all indicators are warmed up
    min_dates = []
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            valid = sig.dropna()
            if len(valid) > 0:
                min_dates.append(valid.index[0])
    backtest_start = max(min_dates).strftime("%Y-%m-%d") if min_dates else "2021-01-01"
    print(f"  Backtest start: {backtest_start}")

    print("\nRunning backtest...")
    result = run_backtest(price_data, igv_prices, backtest_start=backtest_start)

    print(f"\n── Results ────────────────────────────────────")
    print(f"  Portfolio:     ${result['current_value']:,.2f}")
    print(f"  Total Return:  {result['total_return_pct']:+.2f}%")
    print(f"  IGV B&H:       {result['bah_return_pct']:+.2f}%" if result['bah_return_pct'] else "  IGV B&H:       —")
    print(f"  CAGR:          {result['cagr_pct']:+.2f}%")
    print(f"  Sharpe:        {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  -{result['max_drawdown_pct']:.2f}%")
    print(f"  Stocks in BUY: {result['n_buy_stocks']}/{result['n_universe']}")
    print(f"  Cash (SGOV):   {result['cash_pct']:.1f}%")
    print(f"\n  Current Holdings:")
    for h in result["current_holdings"][:10]:
        print(f"    {h['ticker']:6s} {h['weight']:5.1f}%  ${h['value']:>10,.2f}")

    # Merge into portfolios.json
    try:
        with open("portfolios.json", "r") as f:
            output = json.load(f)
        output["portfolios"] = [p for p in output.get("portfolios", []) if p["id"] != "igv-alpha"]
    except FileNotFoundError:
        output = {"portfolios": []}

    output["generated"] = datetime.utcnow().isoformat() + "Z"
    output["portfolios"].append({
        "id":               "igv-alpha",
        "name":             "Tech Software Alpha",
        "description":      f"Equal-weight across top IGV holdings with weekly BUY signal. Max 15% per stock. Cash to SGOV (~5% yield). Rebalances every Monday. Universe: {len(TICKERS)} stocks. Benchmark: IGV.",
        "ticker":           "IGV",
        "benchmark":        "IGV",
        "cash_instrument":  "SGOV",
        "timeframe":        "weekly",
        "starting_capital": STARTING_CAPITAL,
        "disclaimer":       "Simulated performance. Past performance does not guarantee future results.",
        **result,
    })

    with open("portfolios.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n✓ portfolios.json updated")

if __name__ == "__main__":
    main()
