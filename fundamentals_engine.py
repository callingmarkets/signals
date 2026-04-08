"""
CallingMarkets Fundamentals Engine
Runs weekly Sunday night before Monday signal run.
Scores each stock 0-100 across 5 fundamental metrics using yfinance.
Skips ETFs, crypto, and volatility instruments.
Saves fundamentals.json for consumption by analysis engine and widgets.
"""

import json
import os
import time
from datetime import datetime
import yfinance as yf

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = "fundamentals.json"
RATE_DELAY    = 0.5   # seconds between yfinance calls to avoid rate limiting
MAX_RETRIES   = 2

# ── SKIP LISTS ─────────────────────────────────────────────────────────────────
# ETFs — no fundamentals apply
ETF_TICKERS = {
    "SPY","QQQ","IWM","DIA","MDY","VTI","VOO","RSP","SPLG","SCHB","ITOT",
    "XLK","XLC","XLY","XLP","XLF","XLV","XLE","XLI","XLB","XLU","XLRE",
    "XBI","SMH","SOXX","CIBR","HACK","BOTZ","ROBO","FINX","ARKK","ARKG",
    "IBIT","FBTC","ETHA",
    "PDBC","GLD","IAU","SLV","PPLT","USO","BNO","UNG","DBA","CORN","WEAT",
    "SOYB","DJP","GDX","GDXJ","COPX","URA",
    "VXX","UVXY","SVXY","VIXY","VIXM","SPXS","SQQQ","SH","PSQ","TAIL","BTAL",
    "TLT","IEF","SHY","BND","AGG","HYG","JNK","LQD","EMB","TIP","MUB",
    "EFA","VEA","EWJ","EWG","EWU","EWC","EWA","EWL","EWQ","HEZU","DXJ",
    "EEM","VWO","FXI","MCHI","EWZ","INDA","EWY","EWT","KWEB","GXC","INDY",
    "ASML","TSM",  # ADRs — yfinance sometimes misreads, skip
}

# Sector medians for P/E comparison (approximate, updated periodically)
SECTOR_PE_MEDIANS = {
    "Technology":             28.0,
    "Communications":         20.0,
    "Consumer Discretionary": 22.0,
    "Consumer Staples":       20.0,
    "Financials":             13.0,
    "Health Care":            20.0,
    "Energy":                 11.0,
    "Industrials":            22.0,
    "Materials":              17.0,
    "Utilities":              17.0,
    "Real Estate":            35.0,
    "Semiconductors":         25.0,
    "Biotech":                30.0,
    "Cybersecurity":          35.0,
    "AI & Robotics":          35.0,
    "Fintech":                25.0,
    "Broad Market":           22.0,
}

# Debt/Equity thresholds — some sectors carry more debt by nature
SECTOR_DE_THRESHOLD = {
    "Utilities":    4.0,   # utility companies are naturally leveraged
    "Real Estate":  3.0,   # REITs use leverage structurally
    "Financials":   5.0,   # banks have high leverage by design
    "Default":      2.0,   # all other sectors
}

# ── SCORING ────────────────────────────────────────────────────────────────────
def score_ticker(ticker: str, sector: str) -> dict:
    """
    Fetch fundamentals via yfinance and score 0-100.

    Metrics (20 points each):
    1. Revenue growth YoY positive
    2. Earnings (EPS) growth YoY positive
    3. Free cash flow positive (TTM)
    4. P/E below sector median
    5. Debt/Equity below sector threshold
    """
    result = {
        "ticker":          ticker,
        "sector":          sector,
        "score":           None,
        "grade":           None,
        "revenue_growth":  None,
        "earnings_growth": None,
        "fcf_positive":    None,
        "pe_vs_sector":    None,
        "debt_equity_ok":  None,
        "pe":              None,
        "debt_equity":     None,
        "error":           None,
        "updated":         datetime.utcnow().strftime("%Y-%m-%d"),
    }

    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info

            if not info or info.get("quoteType") not in ("EQUITY",):
                result["error"] = f"Not equity: {info.get('quoteType','unknown')}"
                return result

            # ── Metric 1: Revenue growth ──────────────────────────────────────
            rev_growth = info.get("revenueGrowth")  # YoY as decimal e.g. 0.12
            if rev_growth is not None:
                result["revenue_growth"] = rev_growth > 0
            else:
                # fallback: check financials
                try:
                    fin = stock.financials
                    if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                        revs = fin.loc["Total Revenue"].dropna()
                        if len(revs) >= 2:
                            result["revenue_growth"] = float(revs.iloc[0]) > float(revs.iloc[1])
                except: pass

            # ── Metric 2: Earnings growth ─────────────────────────────────────
            earn_growth = info.get("earningsGrowth")  # YoY
            if earn_growth is not None:
                result["earnings_growth"] = earn_growth > 0
            else:
                eps_curr = info.get("trailingEps")
                eps_fwd  = info.get("forwardEps")
                if eps_curr and eps_fwd:
                    result["earnings_growth"] = eps_fwd > eps_curr
                else:
                    try:
                        fin = stock.financials
                        if fin is not None and not fin.empty and "Net Income" in fin.index:
                            ni = fin.loc["Net Income"].dropna()
                            if len(ni) >= 2:
                                result["earnings_growth"] = float(ni.iloc[0]) > float(ni.iloc[1])
                    except: pass

            # ── Metric 3: Free cash flow positive ────────────────────────────
            fcf = info.get("freeCashflow")
            if fcf is not None:
                result["fcf_positive"] = fcf > 0
            else:
                try:
                    cf = stock.cashflow
                    if cf is not None and not cf.empty:
                        op_cf  = cf.loc["Operating Cash Flow"].iloc[0]  if "Operating Cash Flow"  in cf.index else None
                        capex  = cf.loc["Capital Expenditure"].iloc[0]  if "Capital Expenditure"  in cf.index else 0
                        if op_cf is not None:
                            result["fcf_positive"] = (float(op_cf) + float(capex or 0)) > 0
                except: pass

            # ── Metric 4: P/E vs sector median ───────────────────────────────
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe and pe > 0:
                result["pe"] = round(float(pe), 1)
                median = SECTOR_PE_MEDIANS.get(sector, SECTOR_PE_MEDIANS.get("Default", 22.0))
                result["pe_vs_sector"] = "below" if pe < median else "above"
                result["pe_vs_sector_ok"] = pe < median
            else:
                result["pe_vs_sector"] = "n/a"
                result["pe_vs_sector_ok"] = None  # can't score if no P/E

            # ── Metric 5: Debt/Equity ─────────────────────────────────────────
            de = info.get("debtToEquity")
            if de is not None:
                result["debt_equity"] = round(float(de) / 100, 2)  # yfinance returns as %, convert to ratio
                threshold = SECTOR_DE_THRESHOLD.get(sector, SECTOR_DE_THRESHOLD["Default"])
                result["debt_equity_ok"] = result["debt_equity"] < threshold
            else:
                result["debt_equity_ok"] = None  # can't score

            # ── Calculate score ───────────────────────────────────────────────
            scored_metrics = [
                result["revenue_growth"],
                result["earnings_growth"],
                result["fcf_positive"],
                result["pe_vs_sector_ok"],
                result["debt_equity_ok"],
            ]

            valid   = [m for m in scored_metrics if m is not None]
            passing = [m for m in valid if m is True]

            if len(valid) == 0:
                result["error"] = "No scoreable metrics"
                return result

            # Scale to 0-100 based on valid metrics only
            raw_score = (len(passing) / len(valid)) * 100
            result["score"] = round(raw_score)

            # Grade
            if   raw_score >= 80: result["grade"] = "Strong"
            elif raw_score >= 60: result["grade"] = "Solid"
            elif raw_score >= 40: result["grade"] = "Mixed"
            elif raw_score >= 20: result["grade"] = "Weak"
            else:                 result["grade"] = "Poor"

            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            result["error"] = str(e)[:100]
            return result


def grade_color(grade: str) -> str:
    return {
        "Strong": "#15803d",
        "Solid":  "#1d4ed8",
        "Mixed":  "#92620a",
        "Weak":   "#b91c1c",
        "Poor":   "#6b7280",
    }.get(grade, "#6b7280")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"CallingMarkets Fundamentals Engine — {today}\n")

    # Load signals.json to get ticker + sector list
    with open("signals.json", "r") as f:
        data = json.load(f)

    # Build unique ticker → sector map (first occurrence wins)
    ticker_sector: dict[str, str] = {}
    for row in data.get("signals", []):
        t = row["ticker"]
        s = row.get("sector", "Unknown")
        if t not in ticker_sector:
            ticker_sector[t] = s

    # Filter — skip ETFs and crypto
    scoreable = {
        t: s for t, s in ticker_sector.items()
        if t not in ETF_TICKERS
        and "/" not in t                    # skip crypto (BTC/USD etc)
        and not t.endswith("USD")           # extra safety
        and len(t) <= 5                     # sanity check
    }

    print(f"Total tickers: {len(ticker_sector)}")
    print(f"Scoreable (stocks only): {len(scoreable)}")
    print(f"Skipped (ETFs/crypto): {len(ticker_sector) - len(scoreable)}\n")

    results   = {}
    errors    = []
    processed = 0

    for ticker, sector in sorted(scoreable.items()):
        processed += 1
        print(f"  [{processed:3}/{len(scoreable)}] {ticker:<8} ({sector})", end="... ")

        result = score_ticker(ticker, sector)

        if result.get("error"):
            print(f"✗ {result['error']}")
            errors.append(ticker)
        else:
            grade = result.get("grade", "?")
            score = result.get("score", "?")
            print(f"✓ {score}/100 — {grade}")
            results[ticker] = result

        time.sleep(RATE_DELAY)

    # ── Summary stats ──────────────────────────────────────────────────────────
    graded = [r for r in results.values() if r.get("grade")]
    grade_counts = {}
    for r in graded:
        g = r["grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1

    print(f"\n{'='*50}")
    print(f"Scored:  {len(results)}/{len(scoreable)}")
    print(f"Errors:  {len(errors)}")
    if errors:
        print(f"  Failed: {', '.join(errors)}")
    print("\nGrade distribution:")
    for grade in ["Strong","Solid","Mixed","Weak","Poor"]:
        count = grade_counts.get(grade, 0)
        bar = "█" * count
        print(f"  {grade:<8} {count:3}  {bar}")

    # ── Save output ────────────────────────────────────────────────────────────
    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "scored":    len(results),
        "skipped":   len(ticker_sector) - len(scoreable),
        "errors":    len(errors),
        "grades": {
            "Strong": grade_counts.get("Strong", 0),
            "Solid":  grade_counts.get("Solid",  0),
            "Mixed":  grade_counts.get("Mixed",  0),
            "Weak":   grade_counts.get("Weak",   0),
            "Poor":   grade_counts.get("Poor",   0),
        },
        "scores": results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ {OUTPUT_FILE} written — {len(results)} stocks scored")


if __name__ == "__main__":
    run()
