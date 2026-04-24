#!/usr/bin/env python3
"""
Semiconductor Alpha Portfolio (vs SOXX)
- Universe: Top 30 SOXX holdings — integrated, fabless, equipment, EDA
- Signal: 2-of-3 weekly momentum (EMA20>EMA55, RSI14>RSI_EMA14, MACD>Signal)
- Benchmark gate: Weekly SOXX signal must be BUY — if SELL, 100% SGOV
- Entry: Equal weight all stocks with weekly BUY signals (gate open only)
- Cash: 100% SGOV (~5% yield) when gate is SELL
- Benchmark: SOXX buy-and-hold
- Rebalance: Every Monday
- Lookback: ~12 years (2014 start)
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
SGOV_FALLBACK    = 0.05 / 52

EMA_FAST  = 20; EMA_SLOW = 55; RSI_LEN = 14
RSI_MA    = 14; MACD_FAST = 12; MACD_SLOW = 26; MACD_SIG = 9

# ── SOXX Top 30 universe ──────────────────────────────────────────────────────
# Integrated IDMs, fabless designers, foundries, equipment, EDA
SOXX_UNIVERSE = [
    # Large-cap fabless / AI
    {"ticker": "NVDA",  "name": "NVIDIA Corporation"},
    {"ticker": "AMD",   "name": "Advanced Micro Devices"},
    {"ticker": "QCOM",  "name": "Qualcomm Incorporated"},
    {"ticker": "AVGO",  "name": "Broadcom Inc."},
    {"ticker": "MRVL",  "name": "Marvell Technology Inc."},
    {"ticker": "MPWR",  "name": "Monolithic Power Systems"},
    # Integrated Device Manufacturers
    {"ticker": "INTC",  "name": "Intel Corporation"},
    {"ticker": "TXN",   "name": "Texas Instruments Inc."},
    {"ticker": "ADI",   "name": "Analog Devices Inc."},
    {"ticker": "MCHP",  "name": "Microchip Technology Inc."},
    {"ticker": "ON",    "name": "ON Semiconductor Corp."},
    {"ticker": "STM",   "name": "STMicroelectronics N.V."},
    # Foundry / Memory
    {"ticker": "TSM",   "name": "Taiwan Semiconductor (ADR)"},
    {"ticker": "MU",    "name": "Micron Technology Inc."},
    {"ticker": "WDC",   "name": "Western Digital Corporation"},
    # Equipment
    {"ticker": "AMAT",  "name": "Applied Materials Inc."},
    {"ticker": "LRCX",  "name": "Lam Research Corporation"},
    {"ticker": "KLAC",  "name": "KLA Corporation"},
    {"ticker": "ASML",  "name": "ASML Holding N.V. (ADR)"},
    {"ticker": "ONTO",  "name": "Onto Innovation Inc."},
    {"ticker": "ENTG",  "name": "Entegris Inc."},
    {"ticker": "UCTT",  "name": "Ultra Clean Holdings Inc."},
    # EDA / IP
    {"ticker": "CDNS",  "name": "Cadence Design Systems"},
    {"ticker": "SNPS",  "name": "Synopsys Inc."},
    {"ticker": "ARMH",  "name": "ARM Holdings plc (ADR)"},
    # RF / Wireless
    {"ticker": "SWKS",  "name": "Skyworks Solutions Inc."},
    {"ticker": "QRVO",  "name": "Qorvo Inc."},
    {"ticker": "MACOM", "name": "MACOM Technology Solutions"},
    # Power / Auto
    {"ticker": "WOLF",  "name": "Wolfspeed Inc."},
    {"ticker": "ACLS",  "name": "Axcelis Technologies Inc."},
]
TICKERS     = [s["ticker"] for s in SOXX_UNIVERSE]
TICKER_META = {s["ticker"]: s for s in SOXX_UNIVERSE}

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
def fetch_tiingo_weekly(ticker, lookback_days=4380):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {"startDate": start.strftime("%Y-%m-%d"),
              "endDate":   end.strftime("%Y-%m-%d"),
              "resampleFreq": "weekly", "token": TIINGO_KEY}
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
        return df[col].dropna()
    except Exception:
        return None

def fetch_soxx_weekly(lookback_days=4380):
    for attempt in range(3):
        series = fetch_tiingo_weekly("SOXX", lookback_days)
        if series is not None and len(series) > 20:
            print(f"  SOXX benchmark: {len(series)} weeks ({series.index[0].date()} → {series.index[-1].date()})")
            return series
        print(f"  SOXX fetch attempt {attempt+1} failed, retrying...")
        time.sleep(5)
    print("  SOXX fetch failed after 3 attempts")
    return None

def fetch_weekly_stocks(tickers, lookback_days=4380):
    all_data = {}
    for i, ticker in enumerate(tickers):
        series = fetch_tiingo_weekly(ticker, lookback_days)
        if series is not None and len(series) > 20:
            all_data[ticker] = series
        else:
            print(f"  WARNING: No data for {ticker}")
        if (i + 1) % 3 == 0:
            time.sleep(2)
    return all_data

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(price_data, soxx_prices, soxx_weekly_signals=None, backtest_start="2014-01-01"):
    signals = {}
    for ticker, prices in price_data.items():
        sig = compute_signal(prices)
        if sig is not None:
            signals[ticker] = sig

    if not signals:
        print("ERROR: No valid signals — likely Tiingo rate limit. Try again in 1 hour.")
        raise SystemExit(0)

    all_dates = sorted(set().union(*[set(s.index) for s in signals.values()]))
    start_ts  = pd.Timestamp(backtest_start).tz_localize("UTC")
    all_dates = [d for d in all_dates if d >= start_ts]

    capital    = STARTING_CAPITAL
    holdings   = {}
    cash       = STARTING_CAPITAL
    sgov_val   = 0.0
    equity_curve = []
    trades     = []
    weekly_rets = []
    prev_val   = STARTING_CAPITAL
    macro_signal = "BUY"
    macro_open   = True

    for date in all_dates:
        prices_now = {}
        for ticker in TICKERS:
            if ticker in price_data:
                mask = price_data[ticker].index <= date
                if mask.any():
                    prices_now[ticker] = float(price_data[ticker][mask].iloc[-1])

        sgov_val *= (1 + SGOV_FALLBACK)
        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val
        weekly_rets.append((port_val/prev_val)-1 if prev_val > 0 else 0)
        prev_val  = port_val

        # Benchmark gate: weekly SOXX signal must be BUY
        macro_signal = "BUY"
        macro_open   = True
        if soxx_weekly_signals is not None:
            gate_mask = soxx_weekly_signals.index <= date
            if gate_mask.any():
                macro_signal = soxx_weekly_signals[gate_mask].iloc[-1]
                macro_open   = (macro_signal == "BUY")

        buy_tickers = []
        sig_snap    = {}
        if macro_open:
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
                               "price": round(p,2), "shares": round(shares,4),
                               "value": round(alloc,2), "weight": round(w*100,2),
                               "signal": sig_snap.get(ticker,"—")})

        stock_val = sum(holdings.get(t,0)*prices_now.get(t,0) for t in TICKERS)
        port_val  = cash + sgov_val + stock_val
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "value": round(port_val,2),
                             "n_stocks": len(holdings), "cash_pct": round(cash_w*100,1),
                             "buy_tickers": buy_tickers, "macro_signal": macro_signal})

    # ── Metrics ───────────────────────────────────────────────────────────────
    eq_vals = [e["value"] for e in equity_curve]
    current_value = eq_vals[-1] if eq_vals else STARTING_CAPITAL
    total_return  = (current_value/STARTING_CAPITAL-1)*100
    n_weeks = len(eq_vals)
    years   = n_weeks/52
    cagr    = ((current_value/STARTING_CAPITAL)**(1/max(years,0.1))-1)*100

    arr = np.array(weekly_rets[1:])
    vol = float(np.std(arr)*np.sqrt(52)*100) if len(arr)>1 else 0
    rf_w = SGOV_FALLBACK
    sharpe = float(np.mean(arr-rf_w)/np.std(arr-rf_w)*np.sqrt(52)) if np.std(arr)>0 else 0

    peak, max_dd = STARTING_CAPITAL, 0.0
    for v in eq_vals:
        if v > peak: peak = v
        dd = (peak-v)/peak*100
        if dd > max_dd: max_dd = dd

    calmar = cagr/max_dd if max_dd > 0 else 0

    def roll(w):
        if len(eq_vals) < w+1: return None
        return round((eq_vals[-1]/eq_vals[-w]-1)*100, 2)

    yr = datetime.utcnow().year
    ytd_base = next((e["value"] for e in equity_curve if e["date"].startswith(str(yr))), eq_vals[0])
    ytd = round((current_value/ytd_base-1)*100, 2)

    sgov_weeks = sum(1 for e in equity_curve if e["cash_pct"] > 50)
    pct_in_mkt = round((n_weeks-sgov_weeks)/max(n_weeks,1)*100, 1)

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
            "price": round(p,2), "value": round(val,2),
            "weight": round(val/current_value*100,2) if current_value > 0 else 0,
            "signal": "BUY"
        })
    current_holdings.sort(key=lambda x: -x["value"])

    current_signals = {t: signals[t].iloc[-1] for t in TICKERS if t in signals and len(signals[t])>0}

    # BAH benchmark
    bah_curve, bah_return = [], None
    if soxx_prices is not None and len(soxx_prices) > 0:
        first_buy = next((t for t in trades if t["action"]=="BUY"), None)
        if first_buy:
            fb_ts = pd.Timestamp(first_buy["date"]).tz_localize("UTC")
            mask  = soxx_prices.index >= fb_ts
            if mask.any():
                soxx_slice  = soxx_prices[mask]
                soxx_start  = float(soxx_slice.iloc[0])
                bah_curve   = [{"date": d.strftime("%Y-%m-%d"),
                                "value": round(STARTING_CAPITAL*(float(v)/soxx_start),2)}
                               for d,v in soxx_slice.items()]
                bah_current = round(STARTING_CAPITAL*(float(soxx_prices.iloc[-1])/soxx_start),2)
                bah_return  = round((bah_current/STARTING_CAPITAL-1)*100, 2)

    return {
        "total_trades":       len([t for t in trades if t["action"]=="BUY"]),
        "current_value":      round(current_value,2),
        "total_return_pct":   round(total_return,2),
        "total_return_dollar":round(current_value-STARTING_CAPITAL,2),
        "cagr_pct":           round(cagr,2),
        "ytd_return_pct":     ytd,
        "return_1m":          roll(4),
        "return_3m":          roll(13),
        "return_6m":          roll(26),
        "return_1y":          roll(52),
        "max_drawdown_pct":   round(max_dd,2),
        "ann_volatility_pct": round(vol,2),
        "sharpe_ratio":       round(sharpe,2),
        "calmar_ratio":       round(calmar,2),
        "n_buy_stocks":       len(holdings),
        "n_universe":         len(TICKERS),
        "cash_pct":           round(sgov_val/current_value*100,1) if current_value>0 else 100,
        "pct_time_in_market": pct_in_mkt,
        "sgov_weeks":         sgov_weeks,
        "sgov_yield":         5.0,
        "macro_signal":       macro_signal,
        "macro_gate_open":    macro_open,
        "current_holdings":   current_holdings,
        "current_signals":    {t:v for t,v in current_signals.items()},
        "bah_return_pct":     bah_return,
        "bah_equity_curve":   bah_curve,
        "equity_curve":       equity_curve,
        "trades":             trades[-60:],
        "universe":           SOXX_UNIVERSE,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Semiconductor Alpha Portfolio Engine (vs SOXX) — Tiingo data")
    print(f"Universe: {len(TICKERS)} stocks | Starting capital: ${STARTING_CAPITAL:,.0f}")

    print("Fetching SOXX benchmark...")
    soxx_prices = fetch_soxx_weekly(lookback_days=4380)

    soxx_weekly_signals = None
    if soxx_prices is not None and len(soxx_prices) > EMA_SLOW + 10:
        soxx_weekly_signals = compute_signal(soxx_prices)
        if soxx_weekly_signals is not None:
            print(f"  SOXX weekly signal (gate): {soxx_weekly_signals.iloc[-1]}")

    print("\nFetching weekly bars...")
    price_data = fetch_weekly_stocks(TICKERS, lookback_days=4380)
    print(f"  Got data for {len(price_data)}/{len(TICKERS)} tickers")

    # Median backtest start — when ≥50% of stocks have valid signals
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
        print(f"  Backtest start: {backtest_start} ({n_available}/{len(TICKERS)} stocks available)")
        late = {t: d.strftime("%Y-%m-%d") for t, d in ticker_starts.items()
                if d > min_dates[threshold_idx]}
        if late:
            print(f"  Late-starting tickers: {', '.join(late.keys())}")
    else:
        backtest_start = "2014-01-01"

    print("\nRunning backtest...")
    result = run_backtest(price_data, soxx_prices,
                          soxx_weekly_signals=soxx_weekly_signals,
                          backtest_start=backtest_start)

    print(f"\n── Results ─────────────────────────────────────────")
    print(f"  Portfolio:     ${result['current_value']:,.2f}")
    print(f"  Total Return:  {result['total_return_pct']:+.2f}%")
    print(f"  SOXX B&H:      {result['bah_return_pct']:+.2f}%" if result['bah_return_pct'] else "  SOXX B&H:      —")
    print(f"  CAGR:          {result['cagr_pct']:+.2f}%")
    print(f"  Sharpe:        {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:  -{result['max_drawdown_pct']:.2f}%")
    print(f"  Stocks in BUY: {result['n_buy_stocks']}/{result['n_universe']}")
    print(f"  Cash (SGOV):   {result['cash_pct']:.1f}%")
    gate_status = "OPEN" if result.get("macro_gate_open") else "CLOSED - 100% SGOV"
    print(f"  Weekly Gate:   {result.get('macro_signal','--')} ({gate_status})")
    print(f"\n  Current Holdings:")
    for h in result["current_holdings"][:8]:
        print(f"    {h['ticker']:6s} {h['weight']:5.1f}%  ${h['value']:>10,.2f}")

    # Merge into portfolios.json
    try:
        with open("portfolios.json","r") as f:
            output = json.load(f)
        output["portfolios"] = [p for p in output.get("portfolios",[]) if p["id"] != "soxx-alpha"]
    except FileNotFoundError:
        output = {"portfolios": []}

    output["generated"] = datetime.utcnow().isoformat() + "Z"
    output["portfolios"].append({
        "id":               "soxx-alpha",
        "name":             "Semiconductor Alpha",
        "description":      f"Equal-weight across top SOXX holdings with weekly BUY signal. Benchmark gate: 100% SGOV when SOXX weekly is SELL. Universe: {len(TICKERS)} stocks. Benchmark: SOXX.",
        "ticker":           "SOXX",
        "benchmark":        "SOXX",
        "cash_instrument":  "SGOV",
        "timeframe":        "weekly",
        "starting_capital": STARTING_CAPITAL,
        "disclaimer":       "Simulated performance. Past performance does not guarantee future results.",
        **result,
    })

    with open("portfolios.json","w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n✓ portfolios.json updated")

if __name__ == "__main__":
    main()
