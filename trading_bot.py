#!/usr/bin/env python3
"""
CallingMarkets Trading Bot — Paper Trading
Runs every Monday morning after signals update.
Entry: Individual stock, sector Bullish/Accumulation, weekly signal BUY, monthly BUY
Exit:  Weekly flips SELL OR sector drops to Bearish/Distribution
Sizing: Equal weight across all open positions
"""

import os
import json
import requests
from datetime import datetime, date

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_KEY     = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET  = os.environ["ALPACA_SECRET_KEY"]

# Paper trading base URL
ALPACA_BASE    = "https://paper-api.alpaca.markets/v2"
ALPACA_HDR     = {
    "APCA-API-KEY-ID":     ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
    "Content-Type":        "application/json",
}

SIGNALS_URL    = "https://raw.githubusercontent.com/callingmarkets/signals/main/signals.json"
ANALYSIS_URL   = "https://raw.githubusercontent.com/callingmarkets/signals/main/analysis.json"

# Individual stocks only — exclude ETFs and crypto
ETF_SUFFIXES   = ["/USD"]  # crypto
ETF_PREFIXES   = [
    "SPY","QQQ","IWM","DIA","MDY","VTI","VOO","RSP",
    "XLK","XLF","XLE","XLV","XLI","XLB","XLU","XLRE","XLP","XLY","XLC",
    "XBI","SMH","SOXX","ARKK","ARKG","FINX","CIBR","HACK","ROBO","BOTZ",
    "EFA","EEM","VEA","VWO","FXI","MCHI","EWJ","EWZ","INDA","EWG","EWU",
    "EWC","EWA","EWY","EWT","TLT","IEF","SHY","BND","AGG","HYG","JNK",
    "LQD","EMB","TIP","MUB","GLD","IAU","SLV","PPLT","USO","BNO","UNG",
    "DBA","CORN","WEAT","SOYB","PDBC","DJP","GDX","GDXJ","COPX","URA",
    "VXX","UVXY","SVXY","IBIT","FBTC","ETHA",
]

MAX_POSITIONS  = 10   # max concurrent positions
MIN_PRICE      = 5.0  # skip penny stocks
MAX_PRICE      = 5000.0

# Bullish entry conditions
ENTRY_BIASES   = {"Bullish", "Accumulation"}

# ── Alpaca helpers ─────────────────────────────────────────────────────────────
def alpaca_get(path):
    res = requests.get(f"{ALPACA_BASE}{path}", headers=ALPACA_HDR)
    res.raise_for_status()
    return res.json()

def alpaca_post(path, body):
    res = requests.post(f"{ALPACA_BASE}{path}", headers=ALPACA_HDR, json=body)
    return res.json()

def alpaca_delete(path):
    res = requests.delete(f"{ALPACA_BASE}{path}", headers=ALPACA_HDR)
    return res.status_code

def get_account():
    return alpaca_get("/account")

def get_positions():
    return alpaca_get("/positions")

def get_orders(status="open"):
    return alpaca_get(f"/orders?status={status}&limit=100")

def close_position(symbol):
    res = requests.delete(f"{ALPACA_BASE}/positions/{symbol}", headers=ALPACA_HDR)
    if res.status_code in (200, 204):
        print(f"  ✓ Closed position: {symbol}")
    else:
        print(f"  ✗ Failed to close {symbol}: {res.status_code} {res.text[:100]}")
    return res.status_code

def place_order(symbol, notional):
    body = {
        "symbol":        symbol,
        "notional":      str(round(notional, 2)),
        "side":          "buy",
        "type":          "market",
        "time_in_force": "day",
    }
    res = alpaca_post("/orders", body)
    if res.get("id"):
        print(f"  ✓ Buy order placed: {symbol} ~${notional:.0f}")
    else:
        print(f"  ✗ Order failed {symbol}: {res}")
    return res

# ── Data loading ───────────────────────────────────────────────────────────────
def load_signals():
    res = requests.get(SIGNALS_URL + f"?t={int(datetime.now().timestamp())}")
    return res.json()

def load_analysis():
    res = requests.get(ANALYSIS_URL + f"?t={int(datetime.now().timestamp())}")
    return res.json()

def is_stock(ticker):
    if "/" in ticker: return False
    if ticker in ETF_PREFIXES: return False
    return True

# ── Signal logic ───────────────────────────────────────────────────────────────
def get_sector_bias(sector, analysis):
    for s in analysis.get("sectors", []):
        if s["sector"] == sector:
            return s.get("bias", "")
    return ""

def qualifies_for_entry(row, analysis):
    if not is_stock(row["ticker"]): return False
    tf_w = row.get("timeframes", {}).get("weekly", {})
    # Weekly must be BUY
    if tf_w.get("signal") != "BUY": return False
    # Sector must be Bullish or Accumulation
    # (sector bias already incorporates monthly signal — no need to check separately)
    bias = get_sector_bias(row.get("sector",""), analysis)
    if bias not in ENTRY_BIASES: return False
    return True

def should_exit(row, analysis):
    tf_w = row.get("timeframes", {}).get("weekly", {})
    # Exit if weekly flipped to SELL
    if tf_w.get("signal") == "SELL": return True, "weekly flipped SELL"
    # Exit if sector dropped to Bearish or Distribution
    bias = get_sector_bias(row.get("sector",""), analysis)
    if bias in {"Bearish", "Distribution"}: return True, f"sector now {bias}"
    return False, ""

# ── Main bot logic ─────────────────────────────────────────────────────────────
def run():
    print(f"\nCallingMarkets Trading Bot — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load data
    print("\nLoading signals and analysis...")
    signals_data = load_signals()
    analysis     = load_analysis()
    signals      = {r["ticker"]: r for r in signals_data.get("signals", [])}
    print(f"  {len(signals)} tickers loaded | Week of {analysis.get('week_of','?')}")

    # Account info
    account = get_account()
    portfolio_value = float(account.get("portfolio_value", 0))
    cash            = float(account.get("cash", 0))
    print(f"\nAccount: Portfolio ${portfolio_value:,.2f} | Cash ${cash:,.2f}")

    # ── Step 1: Check exits ────────────────────────────────────────────────────
    print("\n── CHECKING EXITS ──")
    positions = get_positions()
    open_symbols = {p["symbol"] for p in positions}

    for pos in positions:
        sym = pos["symbol"]
        if sym not in signals:
            print(f"  {sym} — not in signals, holding")
            continue
        row = signals[sym]
        should_sell, reason = should_exit(row, analysis)
        if should_sell:
            print(f"  EXIT {sym} — {reason}")
            close_position(sym)
        else:
            tf_w = row.get("timeframes", {}).get("weekly", {})
            bias = get_sector_bias(row.get("sector",""), analysis)
            print(f"  HOLD {sym} — weekly {tf_w.get('signal')} | sector {bias}")

    # Refresh positions after exits
    positions    = get_positions()
    open_symbols = {p["symbol"] for p in positions}
    open_count   = len(open_symbols)
    print(f"\n  Open positions after exits: {open_count}")

    # ── Step 2: Find entries ───────────────────────────────────────────────────
    print("\n── SCANNING FOR ENTRIES ──")
    candidates = []
    for ticker, row in signals.items():
        if ticker in open_symbols: continue  # already holding
        if not qualifies_for_entry(row, analysis): continue
        tf_w   = row.get("timeframes", {}).get("weekly", {})
        bias   = get_sector_bias(row.get("sector",""), analysis)
        price  = row.get("price", 0) or 0
        # Price filter
        if price < MIN_PRICE or price > MAX_PRICE: continue
        # Prefer fresh weekly flips (prev was SELL)
        is_fresh = tf_w.get("previous") == "SELL"
        candidates.append({
            "ticker":   ticker,
            "sector":   row.get("sector",""),
            "bias":     bias,
            "weekly":   tf_w.get("signal"),
            "price":    price,
            "fresh":    is_fresh,
        })

    # Sort: fresh flips first, then by bias (Bullish before Accumulation)
    candidates.sort(key=lambda x: (
        0 if x["fresh"] else 1,
        0 if x["bias"] == "Bullish" else 1,
    ))

    print(f"  {len(candidates)} candidates found")
    for c in candidates[:15]:
        print(f"  → {c['ticker']:8} | {c['bias']:12} | ${c['price']:.2f} | {'🔄 fresh flip' if c['fresh'] else 'continued BUY'}")

    # ── Step 3: Place entries ──────────────────────────────────────────────────
    slots_available = MAX_POSITIONS - open_count
    to_buy          = candidates[:slots_available]

    if not to_buy:
        print("\n  No new entries — portfolio full or no qualifying signals")
    else:
        print(f"\n── PLACING {len(to_buy)} NEW ORDERS ──")
        # Equal weight: split available cash evenly across all positions
        total_positions  = open_count + len(to_buy)
        per_position     = portfolio_value / total_positions if total_positions else 0
        print(f"  Equal weight: ${per_position:,.0f} per position ({total_positions} total)")

        for c in to_buy:
            if per_position < 1:
                print(f"  ✗ Skipping {c['ticker']} — position size too small")
                continue
            place_order(c["ticker"], per_position)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── SUMMARY ──")
    final_positions = get_positions()
    print(f"  Open positions: {len(final_positions)}")
    for p in final_positions:
        pnl = float(p.get("unrealized_pl", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
        print(f"  {p['symbol']:8} | {float(p['qty']):.2f} shares | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

    account = get_account()
    print(f"\n  Portfolio value: ${float(account['portfolio_value']):,.2f}")
    print(f"  Cash:           ${float(account['cash']):,.2f}")
    print(f"\nBot run complete — {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    run()
