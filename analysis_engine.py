"""
CallingMarkets Analysis Engine
Runs weekly (Monday). Reads signals.json, fetches recent news per sector,
calls Claude API to generate sector synopses, saves analysis.json.
"""

import json
import os
import re
from datetime import datetime, timedelta
import requests

# ── API KEYS ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NEWSAPI_KEY       = os.environ["NEWSAPI_KEY"]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
NEWSAPI_URL   = "https://newsapi.org/v2/everything"

# ── SECTOR → SEARCH TERMS ─────────────────────────────────────────────────────
# Maps sector labels to news search queries
SECTOR_QUERIES = {
    "ETF - US Market":       "S&P 500 stock market",
    "ETF - Technology":      "technology sector stocks",
    "ETF - Financials":      "financial sector banks",
    "ETF - Energy":          "energy sector oil gas",
    "ETF - Healthcare":      "healthcare sector biotech pharma",
    "ETF - Industrials":     "industrial sector manufacturing",
    "ETF - Materials":       "materials sector mining metals",
    "ETF - Utilities":       "utilities sector electricity",
    "ETF - Real Estate":     "real estate REIT sector",
    "ETF - Staples":         "consumer staples sector",
    "ETF - Consumer":        "consumer discretionary retail",
    "ETF - Communications":  "media communications telecom sector",
    "ETF - Biotech":         "biotech biotechnology sector",
    "ETF - Semiconductors":  "semiconductor chip sector",
    "ETF - Innovation":      "innovation disruptive technology ARK",
    "ETF - Genomics":        "genomics gene editing biotech",
    "ETF - Fintech":         "fintech financial technology payments",
    "ETF - Cybersecurity":   "cybersecurity sector stocks",
    "ETF - Robotics":        "robotics automation sector",
    "ETF - AI & Robotics":   "artificial intelligence AI sector",
    "ETF - International":   "international developed markets stocks",
    "ETF - Emerging Markets":"emerging markets stocks",
    "ETF - Developed Markets":"developed markets Europe Japan stocks",
    "ETF - China":           "China stock market economy",
    "ETF - Japan":           "Japan stock market economy",
    "ETF - Brazil":          "Brazil stock market economy",
    "ETF - India":           "India stock market economy",
    "ETF - Germany":         "Germany stock market economy",
    "ETF - UK":              "UK Britain stock market economy",
    "ETF - Canada":          "Canada stock market economy",
    "ETF - Australia":       "Australia stock market economy",
    "ETF - South Korea":     "South Korea stock market economy",
    "ETF - Taiwan":          "Taiwan stock market economy",
    "ETF - Bonds":           "treasury bonds interest rates Fed",
    "ETF - High Yield":      "high yield junk bonds credit",
    "ETF - Corp Bonds":      "corporate bonds investment grade",
    "ETF - EM Bonds":        "emerging market bonds",
    "ETF - TIPS":            "inflation TIPS treasury",
    "ETF - Municipal":       "municipal bonds muni",
    "ETF - Volatility":      "market volatility VIX fear",
    "ETF - Bitcoin":         "Bitcoin ETF crypto market",
    "ETF - Ethereum":        "Ethereum ETF crypto",
    "Commodity":             "commodities gold silver oil",
    "Crypto":                "cryptocurrency Bitcoin Ethereum market",
    "Technology":            "technology stocks earnings",
    "Financials":            "bank financial stocks earnings",
    "Healthcare":            "healthcare pharma biotech stocks",
    "Energy":                "oil gas energy stocks",
    "Consumer":              "consumer retail stocks spending",
    "Industrials":           "industrial manufacturing stocks",
    "Real Estate":           "real estate housing REIT",
    "Materials":             "materials mining metals stocks",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def fetch_news(sector: str, days_back: int = 7) -> list[str]:
    """Fetch recent headlines for a sector."""
    query = SECTOR_QUERIES.get(sector, sector)
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        r = requests.get(NEWSAPI_URL, params={
            "q":        query,
            "from":     from_date,
            "sortBy":   "relevancy",
            "pageSize": 8,
            "language": "en",
            "apiKey":   NEWSAPI_KEY,
        }, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        headlines = [
            f"- {a['title']} ({a['source']['name']})"
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
        return headlines[:6]
    except Exception as e:
        print(f"  NEWS WARNING [{sector}]: {e}")
        return []

def call_claude(prompt: str) -> str:
    """Call Claude API and return the text response."""
    r = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      "claude-opus-4-5",
            "max_tokens": 600,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()

def build_sector_prompt(sector: str, tickers: list, signals: dict, headlines: list) -> str:
    """Build the Claude prompt for a sector synopsis."""

    # Summarize signals
    daily_buy   = sum(1 for t in tickers if signals.get(t, {}).get("daily")   == "BUY")
    weekly_buy  = sum(1 for t in tickers if signals.get(t, {}).get("weekly")  == "BUY")
    monthly_buy = sum(1 for t in tickers if signals.get(t, {}).get("monthly") == "BUY")
    total       = len(tickers)

    signal_summary = (
        f"Daily: {daily_buy}/{total} BUY | "
        f"Weekly: {weekly_buy}/{total} BUY | "
        f"Monthly: {monthly_buy}/{total} BUY"
    )

    # Key tickers with full signal state
    ticker_lines = []
    for t in tickers[:12]:  # cap at 12 for prompt brevity
        d = signals.get(t, {}).get("daily",   "N/A")
        w = signals.get(t, {}).get("weekly",  "N/A")
        m = signals.get(t, {}).get("monthly", "N/A")
        ticker_lines.append(f"  {t}: D={d} W={w} M={m}")

    news_section = "\n".join(headlines) if headlines else "No recent headlines available."

    today = datetime.utcnow().strftime("%B %d, %Y")

    return f"""You are writing the weekly market analysis for CallingMarkets, a momentum-based signals publication for institutional and sophisticated retail investors.

Write a 1-2 paragraph sector synopsis for: {sector}

Today's date: {today}

SIGNAL DATA (momentum signals across Daily / Weekly / Monthly timeframes):
{signal_summary}

Key tickers:
{chr(10).join(ticker_lines)}

Recent news headlines:
{news_section}

INSTRUCTIONS:
- Lead with the overall sector bias (bullish, bearish, or mixed) based on the signal data
- Note any notable divergence between timeframes (e.g. daily selling off but monthly still bullish)
- Weave in 1-2 relevant news items to provide market context
- Call out 2-3 specific tickers worth watching and why
- Tone: balanced, data-driven, with light editorial conviction — like a sharp analyst, not a robot
- Do NOT use bullet points. Write in flowing prose only.
- Do NOT include disclaimers or mention that this is AI-generated.
- Length: 2 concise paragraphs, approximately 120-180 words total."""

def generate_takeaways(analyses: list) -> list:
    """Generate 4-6 key takeaways summarizing the week across all sectors."""
    # Build a compact summary of all sector biases
    bullish = [a["sector"] for a in analyses if a["bias"] == "Bullish"]
    bearish = [a["sector"] for a in analyses if a["bias"] == "Bearish"]
    mixed   = [a["sector"] for a in analyses if a["bias"] == "Mixed"]

    today = datetime.utcnow().strftime("%B %d, %Y")

    prompt = f"""You are writing the weekly market summary for CallingMarkets, a momentum signals publication.

Date: {today}

SECTOR BIAS SUMMARY:
Bullish ({len(bullish)}): {", ".join(bullish[:8]) if bullish else "None"}
Mixed ({len(mixed)}):   {", ".join(mixed[:8]) if mixed else "None"}
Bearish ({len(bearish)}): {", ".join(bearish[:8]) if bearish else "None"}

SECTOR SYNOPSES (sample):
{chr(10).join([f"- {a['sector']}: {a['synopsis'][:200]}..." for a in analyses[:6]])}

Write exactly 5 key takeaways that summarize the most important market themes this week.
Each takeaway should be one concise sentence (max 20 words), actionable and specific.
Focus on: overall market direction, sector rotation, notable divergences, what traders should watch.

Respond ONLY with a JSON array of 5 strings:
["takeaway 1", "takeaway 2", "takeaway 3", "takeaway 4", "takeaway 5"]"""

    try:
        r = requests.post(ANTHROPIC_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model":      "claude-opus-4-5",
                "max_tokens": 400,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"].strip()
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  Takeaways error: {e}")

    return []

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    # Load signals.json
    with open("signals.json", "r") as f:
        data = json.load(f)

    # Build lookup: {ticker: {daily, weekly, monthly}}
    signals = {}
    for row in data["signals"]:
        signals[row["ticker"]] = {
            "daily":   row["timeframes"]["daily"]["signal"],
            "weekly":  row["timeframes"]["weekly"]["signal"],
            "monthly": row["timeframes"]["monthly"]["signal"],
            "price":   row.get("price"),
        }

    # Group tickers by sector
    sectors: dict[str, list] = {}
    for row in data["signals"]:
        sec = row.get("sector", "Other")
        sectors.setdefault(sec, []).append(row["ticker"])

    print(f"Found {len(sectors)} sectors, {len(signals)} tickers\n")

    analyses = []
    for sector, tickers in sorted(sectors.items()):
        print(f"  Analyzing: {sector} ({len(tickers)} tickers)…")

        headlines = fetch_news(sector)
        prompt    = build_sector_prompt(sector, tickers, signals, headlines)

        try:
            synopsis = call_claude(prompt)
        except Exception as e:
            print(f"    Claude ERROR: {e}")
            synopsis = "Analysis unavailable this week."

        # Compute bias label
        total       = len(tickers)
        weekly_buy  = sum(1 for t in tickers if signals.get(t, {}).get("weekly") == "BUY")
        pct         = weekly_buy / total if total else 0
        bias        = "Bullish" if pct >= 0.6 else "Bearish" if pct <= 0.4 else "Mixed"

        analyses.append({
            "sector":   sector,
            "bias":     bias,
            "tickers":  tickers,
            "signals": {
                "daily_buy":   sum(1 for t in tickers if signals.get(t, {}).get("daily")   == "BUY"),
                "weekly_buy":  weekly_buy,
                "monthly_buy": sum(1 for t in tickers if signals.get(t, {}).get("monthly") == "BUY"),
                "total":       total,
            },
            "synopsis": synopsis,
        })

        print(f"    Bias: {bias} | Done")

    # Generate overall market takeaways
    print("\nGenerating key takeaways…")
    takeaways = generate_takeaways(analyses)

    output = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "week_of":   datetime.utcnow().strftime("%B %d, %Y"),
        "takeaways": takeaways,
        "sectors":   analyses,
    }

    with open("analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ analysis.json written — {len(analyses)} sectors")

if __name__ == "__main__":
    print("CallingMarkets Analysis Engine\n")
    run()
