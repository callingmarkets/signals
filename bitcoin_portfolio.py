#!/usr/bin/env python3
"""
Bitcoin Long-Only Portfolio Engine
- Pulls 2 years of weekly BTC/USD bars from Alpaca
- Applies the exact same 2-of-3 signal logic: EMA20>EMA55, RSI14>RSI_EMA14, MACD>Signal
- BUY signal  → 100% BTC
- SELL signal → 100% SGOV (iShares 0-3M T-Bill ETF, ~5% yield)
- Publishes portfolios.json to the repo
"""

import json, os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests

ALPACA_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET_KEY"]
ALPACA_HDR    = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

CRYPTO_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
STOCKS_URL = "https://data.alpaca.markets/v2/stocks/bars"

STARTING_CAPITAL = 100_000.0

EMA_FAST   = 20
EMA_SLOW   = 55
RSI_LEN    = 14
RSI_MA_LEN = 14
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9

# ── Indicator helpers (identical to signal_engine.py) ─────────────────────────

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def calc_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, adjust=False).mean()

def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = calc_rma(gain, length)
    avg_loss = calc_rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast: int, slow: int, sig: int):
    macd_line   = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, sig)
    return macd_line, signal_line

def compute_signals(src: pd.Series):
    if len(src) < EMA_SLOW + 10:
        return None
    ema20                  = calc_ema(src, EMA_FAST)
    ema55                  = calc_ema(src, EMA_SLOW)
    rsi14                  = calc_rsi(src, RSI_LEN)
    rsi_ma                 = calc_ema(rsi14, RSI_MA_LEN)
    macd_line, signal_line = calc_macd(src, MACD_FAST, MACD_SLOW, MACD_SIG)
    scores  = (ema20 > ema55).astype(int) + (rsi14 > rsi_ma).astype(int) + (macd_line > signal_line).astype(int)
    return scores.apply(lambda s: "BUY" if s >= 2 else "SELL")

# ── Fetch bars ────────────────────────────────────────────────────────────────

def fetch_weekly_crypto(symbol: str, lookback_days: int = 800) -> pd.Series:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {"symbols": symbol, "timeframe": "1Day",
               "start": start.isoformat(), "end": end.isoformat(),
               "limit": 10000, "sort": "asc"}
    bars = []
    while True:
        r = requests.get(CRYPTO_URL, headers=ALPACA_HDR, params=params, timeout=30)
        r.raise_for_status()
        data  = r.json()
        chunk = data.get("bars", {}).get(symbol, [])
        bars.extend(chunk)
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not bars:
        raise ValueError(f"No bars for {symbol}")
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    weekly = df["c"].resample("W-FRI").last().dropna()
    weekly.index = weekly.index - pd.Timedelta(days=4)
    return weekly

def fetch_weekly_stock(symbol: str, lookback_days: int = 800) -> pd.Series:
    """Fetch weekly OHLCV bars for a stock (SGOV)."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {"symbols": symbol, "timeframe": "1Week",
               "start": start.isoformat(), "end": end.isoformat(),
               "limit": 1000, "sort": "asc", "adjustment": "all"}
    r = requests.get(STOCKS_URL, headers=ALPACA_HDR, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    bars = data.get("bars", {}).get(symbol, [])
    if not bars:
        print(f"  WARNING: No SGOV bars — using 5% annualized fallback")
        return None
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    return df["c"]

# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(btc_weekly: pd.Series, sgov_weekly: pd.Series | None, backtest_start: str = "2019-01-01") -> dict:
    signals = compute_signals(btc_weekly)
    if signals is None:
        raise ValueError("Not enough data")

    # Build SGOV weekly return series (week-over-week % change)
    # If no SGOV data, fall back to 5% annualized = ~0.096% per week
    SGOV_FALLBACK_WEEKLY = 0.05 / 52

    if sgov_weekly is not None:
        # Align to BTC index, forward-fill
        sgov_aligned = sgov_weekly.reindex(btc_weekly.index, method="ffill")
        sgov_returns = sgov_aligned.pct_change().fillna(SGOV_FALLBACK_WEEKLY)
        sgov_returns = sgov_returns.clip(lower=0)  # T-bill never negative
        current_sgov_yield = float(sgov_aligned.pct_change().tail(12).mean() * 52 * 100)
    else:
        sgov_returns = pd.Series(SGOV_FALLBACK_WEEKLY, index=btc_weekly.index)
        current_sgov_yield = 5.0

    capital      = STARTING_CAPITAL
    btc_units    = 0.0
    sgov_units   = 0.0   # notional SGOV shares (price ~$100.50)
    sgov_price   = 100.0 # approximate starting price
    entry_price  = None
    entry_date   = None
    prev_signal  = None
    trades       = []
    equity_curve = []

    # Pre-warm prev_signal by iterating up to backtest_start without recording anything
    # This ensures the first real week sees a valid prev_signal context
    start_ts = pd.Timestamp(backtest_start)
    for date, _ in btc_weekly.items():
        if date >= start_ts:
            break
        sig = signals.loc[date] if date in signals.index else None
        if sig is not None:
            prev_signal = sig

    for date, btc_close in btc_weekly.items():
        if date < start_ts:
            continue
        sig = signals.loc[date] if date in signals.index else None
        if sig is None:
            continue

        # Current SGOV price (approximate from returns)
        sgov_ret = float(sgov_returns.loc[date]) if date in sgov_returns.index else SGOV_FALLBACK_WEEKLY
        sgov_price = sgov_price * (1 + sgov_ret)

        # Portfolio value
        portfolio_val = capital + btc_units * btc_close + sgov_units * sgov_price

        # ── SIGNAL TRANSITIONS ─────────────────────────────────────────
        if sig == "BUY" and prev_signal != "BUY":
            # Exit SGOV → Enter BTC
            sell_val = capital + sgov_units * sgov_price
            if sgov_units > 0:
                sgov_pnl = sgov_units * sgov_price - sgov_units * 100.0  # approx cost basis
                trades.append({
                    "date":   date.strftime("%Y-%m-%d"),
                    "action": "SELL SGOV",
                    "asset":  "SGOV",
                    "price":  round(sgov_price, 4),
                    "value":  round(sgov_units * sgov_price, 2),
                    "note":   "Rotating into BTC",
                })
            capital   = sell_val
            sgov_units = 0.0

            # Buy BTC
            btc_units   = capital / btc_close
            entry_price = btc_close
            entry_date  = date
            capital     = 0.0
            trades.append({
                "date":   date.strftime("%Y-%m-%d"),
                "action": "BUY",
                "asset":  "BTC/USD",
                "price":  round(btc_close, 2),
                "units":  round(btc_units, 6),
                "value":  round(btc_units * btc_close, 2),
                "note":   "Weekly BUY signal",
            })

        elif sig == "SELL" and prev_signal == "BUY":
            # Exit BTC → Enter SGOV
            sell_val = btc_units * btc_close
            pnl      = sell_val - (btc_units * entry_price)
            pnl_pct  = (btc_close / entry_price - 1) * 100
            trades.append({
                "date":       date.strftime("%Y-%m-%d"),
                "action":     "SELL",
                "asset":      "BTC/USD",
                "price":      round(btc_close, 2),
                "units":      round(btc_units, 6),
                "value":      round(sell_val, 2),
                "pnl":        round(pnl, 2),
                "pnl_pct":    round(pnl_pct, 2),
                "held_weeks": max(1, round((date - entry_date).days / 7)),
                "note":       "Weekly SELL signal → rotating to SGOV",
            })
            capital   = sell_val
            btc_units = 0.0
            entry_price = None
            entry_date  = None

            # Buy SGOV
            sgov_units = capital / sgov_price
            capital    = 0.0
            trades.append({
                "date":   date.strftime("%Y-%m-%d"),
                "action": "BUY SGOV",
                "asset":  "SGOV",
                "price":  round(sgov_price, 4),
                "units":  round(sgov_units, 4),
                "value":  round(sgov_units * sgov_price, 2),
                "note":   "T-bill yield while awaiting BUY signal",
            })

        prev_signal = sig
        total_val   = capital + btc_units * btc_close + sgov_units * sgov_price
        equity_curve.append({
            "date":   date.strftime("%Y-%m-%d"),
            "value":  round(total_val, 2),
            "btc":    round(btc_close, 2),
            "signal": sig,
            "in_btc": btc_units > 0,
        })

    # Current state
    current_btc    = float(btc_weekly.iloc[-1])
    current_sgov   = sgov_price
    current_value  = capital + btc_units * current_btc + sgov_units * current_sgov
    current_signal = signals.iloc[-1]
    in_position    = btc_units > 0
    in_sgov        = sgov_units > 0

    # Performance metrics
    total_return = (current_value / STARTING_CAPITAL - 1) * 100
    sell_trades  = [t for t in trades if t["action"] == "SELL"]
    win_trades   = [t for t in sell_trades if t.get("pnl", 0) > 0]
    loss_trades  = [t for t in sell_trades if t.get("pnl", 0) <= 0]
    win_rate     = len(win_trades) / max(1, len(sell_trades)) * 100

    # Estimate SGOV contribution
    sgov_weeks_in_cash = sum(1 for e in equity_curve if not e["in_btc"])
    sgov_contribution  = round(STARTING_CAPITAL * (SGOV_FALLBACK_WEEKLY * sgov_weeks_in_cash) if sgov_weekly is None
                               else sgov_weeks_in_cash * STARTING_CAPITAL * 0.001, 2)

    # Max drawdown
    peak, max_dd = STARTING_CAPITAL, 0.0
    for e in equity_curve:
        if e["value"] > peak: peak = e["value"]
        dd = (peak - e["value"]) / peak * 100
        if dd > max_dd: max_dd = dd

    # ── Extended performance metrics ──────────────────────────────────────────
    equity_vals  = [e["value"] for e in equity_curve]
    equity_dates = [e["date"]  for e in equity_curve]
    n_weeks      = len(equity_vals)

    # CAGR
    years = n_weeks / 52
    cagr  = ((current_value / STARTING_CAPITAL) ** (1 / max(years, 0.1)) - 1) * 100

    # Weekly returns
    weekly_rets = []
    for i in range(1, len(equity_vals)):
        r = (equity_vals[i] / equity_vals[i-1]) - 1
        weekly_rets.append(r)
    weekly_arr = np.array(weekly_rets)

    # Annualized volatility
    ann_vol = float(np.std(weekly_arr) * np.sqrt(52) * 100) if len(weekly_arr) > 1 else 0.0

    # Sharpe (using current SGOV yield as risk-free)
    rf_weekly   = (current_sgov_yield / 100) / 52
    excess_rets = weekly_arr - rf_weekly
    sharpe      = float(np.mean(excess_rets) / np.std(excess_rets) * np.sqrt(52)) if np.std(excess_rets) > 0 else 0.0

    # Calmar
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    # Rolling returns
    def rolling_return(weeks_back):
        if len(equity_vals) < weeks_back + 1:
            return None
        v_now  = equity_vals[-1]
        v_then = equity_vals[-weeks_back]
        return round((v_now / v_then - 1) * 100, 2)

    # YTD: find first equity entry of current year
    current_year = datetime.utcnow().year
    ytd_start    = next((e["value"] for e in equity_curve if e["date"].startswith(str(current_year))), equity_vals[0])
    ytd_return   = round((current_value / ytd_start - 1) * 100, 2)

    # Trade stats
    win_pcts  = [t["pnl_pct"] for t in sell_trades if t.get("pnl", 0) > 0]
    loss_pcts = [t["pnl_pct"] for t in sell_trades if t.get("pnl", 0) <= 0]
    avg_win   = round(float(np.mean(win_pcts)),  2) if win_pcts  else 0.0
    avg_loss  = round(float(np.mean(loss_pcts)), 2) if loss_pcts else 0.0
    avg_hold  = round(float(np.mean([t.get("held_weeks", 1) for t in sell_trades])), 1) if sell_trades else 0.0
    best_trade  = round(max((t.get("pnl_pct", 0) for t in sell_trades), default=0), 2)
    worst_trade = round(min((t.get("pnl_pct", 0) for t in sell_trades), default=0), 2)

    # Winning/losing streak
    results = [1 if t.get("pnl", 0) > 0 else 0 for t in sell_trades]
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for r in results:
        if r == 1: cur_win += 1; cur_loss = 0
        else:      cur_loss += 1; cur_win = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Time in market
    btc_weeks  = sum(1 for e in equity_curve if e.get("in_btc"))
    pct_in_mkt = round(btc_weeks / max(n_weeks, 1) * 100, 1)

    # ── Buy & Hold comparison ─────────────────────────────────────────────────
    bah_start   = float(btc_weekly.iloc[0])
    bah_curve   = [{"date": d.strftime("%Y-%m-%d"), "value": round(STARTING_CAPITAL * (float(v) / bah_start), 2)}
                   for d, v in btc_weekly.items()]
    bah_current = round(STARTING_CAPITAL * (float(btc_weekly.iloc[-1]) / bah_start), 2)
    bah_return  = round((bah_current / STARTING_CAPITAL - 1) * 100, 2)

    return {
        "current_signal":      current_signal,
        "bah_current_value":   bah_current,
        "bah_return_pct":      bah_return,
        "bah_equity_curve":    bah_curve,
        "current_value":       round(current_value, 2),
        "current_btc_price":   round(current_btc, 2),
        "current_sgov_price":  round(current_sgov, 4),
        "current_sgov_yield":  round(current_sgov_yield, 2),
        "in_position":         in_position,
        "in_sgov":             in_sgov,
        "btc_units":           round(btc_units, 6),
        "sgov_units":          round(sgov_units, 4),
        "cash":                round(capital, 2),
        # Return metrics
        "total_return_pct":    round(total_return, 2),
        "total_return_dollar": round(current_value - STARTING_CAPITAL, 2),
        "cagr_pct":            round(cagr, 2),
        "ytd_return_pct":      ytd_return,
        "return_1m":           rolling_return(4),
        "return_3m":           rolling_return(13),
        "return_6m":           rolling_return(26),
        "return_1y":           rolling_return(52),
        # Risk metrics
        "max_drawdown_pct":    round(max_dd, 2),
        "ann_volatility_pct":  round(ann_vol, 2),
        "sharpe_ratio":        round(sharpe, 2),
        "calmar_ratio":        round(calmar, 2),
        # Trade metrics
        "total_trades":        len(sell_trades),
        "winning_trades":      len(win_trades),
        "losing_trades":       len(loss_trades),
        "win_rate_pct":        round(win_rate, 2),
        "avg_win_pct":         avg_win,
        "avg_loss_pct":        avg_loss,
        "avg_hold_weeks":      avg_hold,
        "best_trade_pct":      best_trade,
        "worst_trade_pct":     worst_trade,
        "max_win_streak":      max_win_streak,
        "max_loss_streak":     max_loss_streak,
        # Cash leg
        "sgov_weeks":          sgov_weeks_in_cash,
        "btc_weeks":           btc_weeks,
        "pct_time_in_market":  pct_in_mkt,
        "sgov_contribution_approx": sgov_contribution,
        # Current trade
        "entry_price":         round(entry_price, 2) if entry_price else None,
        "entry_date":          entry_date.strftime("%Y-%m-%d") if entry_date else None,
        # Data
        "trades":              trades[-30:],
        "equity_curve":        equity_curve,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Bitcoin Long-Only Portfolio Engine (with SGOV cash leg)")
    print(f"Starting capital: ${STARTING_CAPITAL:,.0f}")

    print("Fetching BTC/USD weekly bars...")
    # Fetch from ~2016 so indicators are fully warmed up by Jan 2019
    # EMA55 + MACD26 need ~80+ weeks before producing valid signals
    btc = fetch_weekly_crypto("BTC/USD", lookback_days=3300)
    print(f"  {len(btc)} weeks full history ({btc.index[0].date()} → {btc.index[-1].date()})")

    print("Fetching SGOV weekly bars...")
    try:
        sgov = fetch_weekly_stock("SGOV", lookback_days=2400)
        if sgov is not None:
            print(f"  {len(sgov)} weeks of SGOV data")
    except Exception as e:
        print(f"  SGOV fetch failed ({e}), using 5% annualized fallback")
        sgov = None

    print("Running backtest...")
    result = run_backtest(btc, sgov, backtest_start="2019-01-01")

    print(f"\n── Results ──────────────────────────────────────")
    print(f"  Signal:        {result['current_signal']}")
    print(f"  In BTC:        {'Yes' if result['in_position'] else 'No'}")
    print(f"  In SGOV:       {'Yes' if result['in_sgov'] else 'No'}")
    print(f"  Portfolio:     ${result['current_value']:,.2f}")
    print(f"  Total Return:  {result['total_return_pct']:+.2f}%")
    print(f"  Max Drawdown:  -{result['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:      {result['win_rate_pct']:.1f}%")
    print(f"  SGOV Weeks:    {result['sgov_weeks']} weeks in T-bills")
    print(f"  SGOV Yield:    ~{result['current_sgov_yield']:.2f}% annualized")

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "portfolios": [{
            "id":              "btc-long-only",
            "name":            "Bitcoin Long-Only",
            "description":     "100% BTC on weekly BUY signal. 100% SGOV (T-bill ETF, ~5% yield) on SELL. Signal: 2-of-3 momentum (EMA20/55, RSI14, MACD) on weekly bars. Simulated from January 2019.",
            "ticker":          "BTC/USD",
            "cash_instrument": "SGOV",
            "cash_instrument_desc": "iShares 0-3 Month Treasury Bill ETF (~5% annualized yield)",
            "timeframe":       "weekly",
            "starting_capital": STARTING_CAPITAL,
            "disclaimer":      "Simulated performance based on historical signals. Past performance does not guarantee future results.",
            **result,
        }]
    }

    with open("portfolios.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n✓ portfolios.json written")

if __name__ == "__main__":
    main()
