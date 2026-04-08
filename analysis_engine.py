"""
CallingMarkets Analysis Engine
Runs weekly (Monday). Reads signals.json, determines sector bias using
sector ETF signals, generates a full market intelligence article,
posts to WordPress, and saves analysis.json for widget consumption.
"""

import json
import os
import re
from datetime import datetime, timedelta
import requests

# ── API KEYS ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NEWSAPI_KEY       = os.environ["NEWSAPI_KEY"]
WP_URL            = os.environ["WP_URL"]
WP_USERNAME       = os.environ["WP_USERNAME"]
WP_APP_PASSWORD   = os.environ["WP_APP_PASSWORD"]

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
NEWSAPI_URL   = "https://newsapi.org/v2/everything"

# ── SECTOR ETF MAP ─────────────────────────────────────────────────────────────
SECTOR_ETF_MAP = {
    "Technology":              "XLK",
    "Communications":          "XLC",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Materials":               "XLB",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Semiconductors":          "SMH",
    "Biotech":                 "XBI",
    "Cybersecurity":           "CIBR",
    "AI & Robotics":           "BOTZ",
    "Fintech":                 "FINX",
    "Crypto":                  "BTC/USD",
    "Commodities":             "PDBC",
    "Volatility":              "VXX",
    "Fixed Income":            "TLT",
    "International Developed": "EFA",
    "International Emerging":  "EEM",
    "Broad Market":            "SPY",
    # Legacy
    "Healthcare": "XLV", "Consumer": "XLY", "ETF - Technology": "XLK",
    "ETF - Financials": "XLF", "ETF - Energy": "XLE", "ETF - Healthcare": "XLV",
    "ETF - Industrials": "XLI", "ETF - Materials": "XLB", "ETF - Utilities": "XLU",
    "ETF - Real Estate": "XLRE", "ETF - Staples": "XLP", "ETF - Consumer": "XLY",
    "ETF - Communications": "XLC", "ETF - Biotech": "XBI", "ETF - Semiconductors": "SMH",
    "ETF - Volatility": "VXX", "ETF - Bitcoin": "BTC/USD", "ETF - Ethereum": "ETH/USD",
    "ETF - International": "EFA", "ETF - Emerging Markets": "EEM",
    "ETF - Developed Markets": "EFA", "ETF - Bonds": "TLT", "ETF - High Yield": "HYG",
    "ETF - Corp Bonds": "LQD", "ETF - EM Bonds": "EMB", "ETF - TIPS": "TIP",
    "ETF - Municipal": "MUB", "ETF - US Market": "SPY", "ETF - Innovation": "ARKK",
    "ETF - Genomics": "ARKG", "ETF - Fintech": "FINX", "ETF - Cybersecurity": "CIBR",
    "ETF - Robotics": "ROBO", "ETF - AI & Robotics": "BOTZ", "ETF - China": "FXI",
    "ETF - Japan": "EWJ", "ETF - Brazil": "EWZ", "ETF - India": "INDA",
    "ETF - Germany": "EWG", "ETF - UK": "EWU", "ETF - Canada": "EWC",
    "ETF - Australia": "EWA", "ETF - South Korea": "EWY", "ETF - Taiwan": "EWT",
    "Commodity": "PDBC",
}

SECTOR_QUERIES = {
    "Technology": "technology stocks earnings AI chips",
    "Communications": "media communications telecom sector",
    "Consumer Discretionary": "consumer spending retail travel stocks",
    "Consumer Staples": "consumer staples grocery defensive stocks",
    "Financials": "bank financial stocks earnings rates",
    "Health Care": "healthcare pharma biotech stocks",
    "Energy": "oil gas energy stocks crude",
    "Industrials": "industrial manufacturing defense aerospace stocks",
    "Materials": "materials mining metals chemicals stocks",
    "Utilities": "utilities electricity power grid stocks",
    "Real Estate": "real estate REIT housing stocks",
    "Semiconductors": "semiconductor chip stocks AI demand",
    "Biotech": "biotech biotechnology FDA drug stocks",
    "Cybersecurity": "cybersecurity hacking breach security stocks",
    "AI & Robotics": "artificial intelligence robotics automation stocks",
    "Fintech": "fintech payments digital finance stocks",
    "Crypto": "cryptocurrency Bitcoin Ethereum market",
    "Commodities": "commodities gold oil agriculture metals",
    "Volatility": "market volatility VIX fear index",
    "Fixed Income": "treasury bonds interest rates Fed policy",
    "International Developed": "international developed markets Europe Japan stocks",
    "International Emerging": "emerging markets China India Brazil stocks",
    "Broad Market": "S&P 500 stock market outlook",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def call_claude(prompt: str, max_tokens: int = 1200) -> str:
    r = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":    "claude-opus-4-5",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def fetch_news(query: str, days_back: int = 7) -> list[str]:
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        r = requests.get(NEWSAPI_URL, params={
            "q": query, "from": from_date, "sortBy": "relevancy",
            "pageSize": 6, "language": "en", "apiKey": NEWSAPI_KEY,
        }, timeout=10)
        r.raise_for_status()
        return [
            f"- {a['title']} ({a['source']['name']})"
            for a in r.json().get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ][:5]
    except Exception as e:
        print(f"  NEWS WARNING: {e}")
        return []


def get_etf_bias(sector: str, signals: dict) -> tuple[str, str]:
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf or etf not in signals:
        return None, etf
    s = signals[etf]
    weekly  = s.get("weekly",  "SELL")
    monthly = s.get("monthly", "SELL")
    if weekly  in ("ERR", "N/A", None): weekly  = "SELL"
    if monthly in ("ERR", "N/A", None): monthly = "SELL"
    if   monthly == "BUY"  and weekly == "BUY":  return "Bullish", etf
    elif monthly == "BUY"  and weekly == "SELL": return "Distribution", etf
    elif monthly == "SELL" and weekly == "BUY":  return "Accumulation", etf
    else:                                         return "Bearish", etf


def get_fallback_bias(tickers: list, signals: dict) -> str:
    total = len(tickers)
    if total == 0: return "Bearish"
    wb = sum(1 for t in tickers if signals.get(t, {}).get("weekly")  == "BUY")
    mb = sum(1 for t in tickers if signals.get(t, {}).get("monthly") == "BUY")
    if mb/total >= 0.55 and wb/total >= 0.55: return "Bullish"
    if mb/total >= 0.55: return "Distribution"
    if wb/total >= 0.55: return "Accumulation"
    return "Bearish"


def get_flips(signals_data: list, signals: dict, tf: str = "weekly") -> tuple[list, list]:
    """Return (bullish_flips, bearish_flips) for a timeframe."""
    bull, bear = [], []
    seen = set()
    for row in signals_data:
        ticker = row["ticker"]
        if ticker in seen: continue
        t = row.get("timeframes", {}).get(tf, {})
        curr, prev = t.get("signal"), t.get("previous")
        if not curr or not prev or curr == prev: continue
        if curr == "BUY"  and prev == "SELL":
            bull.append(row)
            seen.add(ticker)
        elif curr == "SELL" and prev == "BUY":
            bear.append(row)
            seen.add(ticker)
    return bull, bear


# ── ARTICLE GENERATION ────────────────────────────────────────────────────────
def generate_article(analyses: list, signals_data: list, signals: dict,
                     market_news: list, today: str, week_of: str,
                     fundamentals: dict = {}) -> str:
    """Generate the full HTML article to post to WordPress."""

    # ── Stats for market state ────────────────────────────────────────────────
    bias_counts = {}
    for a in analyses:
        bias_counts[a["bias"]] = bias_counts.get(a["bias"], 0) + 1

    bullish_sectors = [a["sector"] for a in analyses if a["bias"] == "Bullish"]
    accum_sectors   = [a["sector"] for a in analyses if a["bias"] == "Accumulation"]
    dist_sectors    = [a["sector"] for a in analyses if a["bias"] == "Distribution"]
    bearish_sectors = [a["sector"] for a in analyses if a["bias"] == "Bearish"]

    # ── Weekly flips cross-referenced with sector phase ───────────────────────
    bull_flips, bear_flips = get_flips(signals_data, signals, "weekly")

    # Categorize flips by sector context
    high_conv_longs  = [r for r in bull_flips if r.get("sector_bias") in ("Bullish", "Accumulation")]
    high_conv_shorts = [r for r in bear_flips if r.get("sector_bias") in ("Bearish", "Distribution")]
    contrarian_watch = [r for r in bull_flips if r.get("sector_bias") in ("Bearish", "Distribution")]

    # ── Top setups — tickers with all 3 timeframes aligned ───────────────────
    top_longs, top_shorts = [], []
    seen_l, seen_s = set(), set()
    for row in signals_data:
        ticker = row["ticker"]
        if "/" in ticker: continue  # skip crypto for setups
        tf = row.get("timeframes", {})
        d = tf.get("daily",   {}).get("signal")
        w = tf.get("weekly",  {}).get("signal")
        m = tf.get("monthly", {}).get("signal")
        bias = row.get("sector_bias", "")
        if d == "BUY" and w == "BUY" and m == "BUY" and bias in ("Bullish", "Accumulation"):
            if ticker not in seen_l:
                top_longs.append(row)
                seen_l.add(ticker)
        if d == "SELL" and w == "SELL" and m == "SELL" and bias in ("Bearish", "Distribution"):
            if ticker not in seen_s:
                top_shorts.append(row)
                seen_s.add(ticker)

    # ── Build prompts ─────────────────────────────────────────────────────────
    sector_summary = "\n".join([
        f"- {a['sector']}: {a['bias']} (ETF: {a.get('bias_etf','?')})"
        for a in analyses
    ])

    flip_summary = ""
    if high_conv_longs:
        flip_summary += "High conviction BUY flips (sector is Bullish/Accumulation):\n"
        flip_summary += "\n".join([f"  ${r['ticker']} in {r['sector']} ({r.get('sector_bias','')})" for r in high_conv_longs[:8]])
    if high_conv_shorts:
        flip_summary += "\n\nHigh conviction SELL flips (sector is Bearish/Distribution):\n"
        flip_summary += "\n".join([f"  ${r['ticker']} in {r['sector']} ({r.get('sector_bias','')})" for r in high_conv_shorts[:8]])
    if contrarian_watch:
        flip_summary += "\n\nContrarian BUY flips (sector is Bearish — watch carefully):\n"
        flip_summary += "\n".join([f"  ${r['ticker']} in {r['sector']} ({r.get('sector_bias','')})" for r in contrarian_watch[:5]])

    long_setups  = ", ".join([f"${r['ticker']}" for r in top_longs[:8]])
    short_setups = ", ".join([f"${r['ticker']}" for r in top_shorts[:8]])

    news_text = "\n".join(market_news) if market_news else "No major headlines this week."

    sector_synopses = "\n\n".join([
        f"{a['sector']} ({a['bias']}): {a['synopsis']}"
        for a in sorted(analyses, key=lambda x: ["Bullish","Accumulation","Distribution","Bearish"].index(x["bias"]) if x["bias"] in ["Bullish","Accumulation","Distribution","Bearish"] else 4)
    ])

    # ── Call Claude to write each section ─────────────────────────────────────
    print("  Writing market state section...")
    market_state = call_claude(f"""You write the weekly market intelligence report for CallingMarkets.ai — a momentum signal platform for serious traders.

Date: {today} | Week of: {week_of}

SECTOR BIAS SUMMARY (driven by sector ETF signals):
Bullish ({len(bullish_sectors)}): {', '.join(bullish_sectors) or 'None'}
Accumulation ({len(accum_sectors)}): {', '.join(accum_sectors) or 'None'}
Distribution ({len(dist_sectors)}): {', '.join(dist_sectors) or 'None'}
Bearish ({len(bearish_sectors)}): {', '.join(bearish_sectors) or 'None'}

RECENT MARKET HEADLINES:
{news_text}

Write the opening "State of the Market" section — 3 punchy paragraphs.
- Paragraph 1: Overall market posture this week. What does the sector distribution tell us?
- Paragraph 2: The dominant theme — what's rotating, what's holding, what's breaking down?
- Paragraph 3: The key risk and the key opportunity heading into this week.
Tone: confident, analytical, direct. Like a senior PM talking to their team Monday morning.
No bullet points. No headers. Just sharp prose.""", max_tokens=600)

    print("  Writing signal flips section...")
    flips_section = call_claude(f"""You write for CallingMarkets.ai — a momentum signal platform.

Date: {today}

WEEKLY SIGNAL FLIPS + SECTOR CONTEXT:
{flip_summary if flip_summary else 'No significant weekly flips this week.'}

Write the "Signal Flips Worth Watching" section — 3-4 paragraphs.
- Open with what the flip activity tells us about market internals
- Discuss the high conviction longs (BUY flip + bullish/accumulation sector) — which are most interesting and why
- Discuss the high conviction shorts (SELL flip + bearish/distribution sector) — which are most dangerous
- If any contrarian flips exist (BUY in bearish sector), note them as either early recovery or potential bull trap
- Be specific about tickers — use $TICKER format
- Tone: trader-grade analysis, not financial advice boilerplate
No bullet points. Flowing prose only.""", max_tokens=700)

    # Add fundamentals context for setups
    def fmt_setup(rows, max_n=8):
        lines = []
        for r in rows[:max_n]:
            t  = r["ticker"]
            fd = fundamentals.get(t, {})
            score_val = fd.get("score", "?")
            fs = f"  Fundamentals: {score_val}/100 ({fd.get('grade','N/A')})" if fd.get("score") is not None else ""
            lines.append(f"  ${t} ({r.get('sector','')}) — Sector: {r.get('sector_bias','')}{fs}")
        return "\n".join(lines) if lines else "None this week"

    long_setups_detail  = fmt_setup(top_longs)
    short_setups_detail = fmt_setup(top_shorts)

    print("  Writing setups section...")
    setups_section = call_claude(f"""You write for CallingMarkets.ai — a momentum signal platform.

Date: {today}

STOCKS WITH FULL TIMEFRAME ALIGNMENT:
Long setups (Daily BUY + Weekly BUY + Monthly BUY + Bullish/Accumulation sector):
{long_setups_detail}

Short setups (Daily SELL + Weekly SELL + Monthly SELL + Bearish/Distribution sector):
{short_setups_detail}

Write the "Setups to Watch" section — 2-3 paragraphs.
- Explain what full timeframe alignment means for conviction
- Where fundamentals score is provided, weave it in — a 80+/100 long is higher conviction than a 40/100 long
- Highlight 3-5 most interesting long setups with brief reasoning
- Highlight 3-5 most interesting short setups with brief reasoning
- Use $TICKER format
- End with one sentence on overall positioning bias
Tone: sharp, actionable. Like a morning note from a real desk.""", max_tokens=600)

    # ── Assemble HTML article ──────────────────────────────────────────────────
    bias_colors = {
        "Bullish":      ("#15803d", "#dcfce7"),
        "Accumulation": ("#1d4ed8", "#eff6ff"),
        "Distribution": ("#92620a", "#fef9ec"),
        "Bearish":      ("#b91c1c", "#fef2f2"),
    }

    sector_html = ""
    for a in sorted(analyses, key=lambda x: ["Bullish","Accumulation","Distribution","Bearish"].index(x["bias"]) if x["bias"] in ["Bullish","Accumulation","Distribution","Bearish"] else 4):
        color, bg = bias_colors.get(a["bias"], ("#374151", "#f9fafb"))
        def ticker_badge(t):
            fd = fundamentals.get(t, {})
            grade = fd.get("grade")
            gcolor = {"Strong":"#15803d","Solid":"#1d4ed8","Mixed":"#92620a","Weak":"#b91c1c","Poor":"#6b7280"}.get(grade,"")
            score_val2 = fd.get("score", "?")
            dot = (f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{gcolor};margin-left:4px;vertical-align:middle;" title="Fundamentals: {score_val2} ({grade})"></span>' if grade else "")
            return f'<span style="background:#f3f4f6;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;margin:2px;">{t}{dot}</span>'
        tickers_html = " ".join([ticker_badge(t) for t in a.get("tickers", [])[:12] if not t.startswith("ETF") and len(t) <= 5])
        sector_html += f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <span style="font-size:14px;font-weight:700;color:#111827;">{a['sector']}</span>
    <span style="background:{bg};color:{color};font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;padding:3px 10px;border-radius:20px;">{a['bias']}</span>
    <span style="font-size:11px;color:#9ca3af;margin-left:auto;">ETF: {a.get('bias_etf','—')}</span>
  </div>
  <p style="font-size:14px;line-height:1.6;color:#374151;margin:0 0 10px;">{a['synopsis']}</p>
  <div style="margin-top:8px;">{tickers_html}</div>
</div>"""

    # Overview stats bar
    stats_bar = f"""<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:32px;">
  <div style="background:#dcfce7;border-radius:8px;padding:16px;text-align:center;">
    <div style="font-size:28px;font-weight:700;color:#15803d;">{len(bullish_sectors)}</div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;color:#15803d;">Bullish</div>
  </div>
  <div style="background:#eff6ff;border-radius:8px;padding:16px;text-align:center;">
    <div style="font-size:28px;font-weight:700;color:#1d4ed8;">{len(accum_sectors)}</div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;color:#1d4ed8;">Accumulation</div>
  </div>
  <div style="background:#fef9ec;border-radius:8px;padding:16px;text-align:center;">
    <div style="font-size:28px;font-weight:700;color:#92620a;">{len(dist_sectors)}</div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;color:#92620a;">Distribution</div>
  </div>
  <div style="background:#fef2f2;border-radius:8px;padding:16px;text-align:center;">
    <div style="font-size:28px;font-weight:700;color:#b91c1c;">{len(bearish_sectors)}</div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;color:#b91c1c;">Bearish</div>
  </div>
</div>"""

    def paras(text):
        return "\n".join("<p>" + p.strip() + "</p>" for p in text.split("\n\n") if p.strip())

    article_html = (
        '<p style="font-size:13px;color:#6b7280;margin-bottom:24px;">Week of ' + week_of + ' &nbsp;·&nbsp; CallingMarkets Signal Intelligence</p>\n\n'
        + stats_bar + "\n\n"
        + "<h2>State of the Market</h2>\n"
        + paras(market_state) + "\n\n"
        + "<h2>Signal Flips Worth Watching</h2>\n"
        + paras(flips_section) + "\n\n"
        + "<h2>Setups to Watch</h2>\n"
        + paras(setups_section) + "\n\n"
        + "<h2>Sector Breakdown</h2>\n"
        + '<p style="color:#6b7280;font-size:14px;margin-bottom:20px;">Sector phase determined by ETF signal alignment (Monthly + Weekly). Ordered by conviction.</p>\n'
        + sector_html + "\n\n"
        + '<hr style="border:none;border-top:1px solid #e5e7eb;margin:32px 0;">\n'
        + '<p style="font-size:12px;color:#9ca3af;">Signals generated by the CallingMarkets momentum engine using EMA/RSI/MACD across daily, weekly, and monthly timeframes. This report is for informational purposes and does not constitute investment advice.</p>'
    )

    return article_html


def generate_seo_meta(analyses: list, bull_count: int, bear_count: int,
                      week_of: str, today: str) -> dict:
    """Generate SEO title, slug, excerpt for the WordPress post."""
    bullish = [a["sector"] for a in analyses if a["bias"] == "Bullish"]
    bearish = [a["sector"] for a in analyses if a["bias"] == "Bearish"]

    prompt = f"""Generate SEO metadata for a weekly market analysis article for CallingMarkets.ai.

Week of: {week_of}
Bullish sectors ({bull_count}): {', '.join(bullish[:4]) or 'None'}
Bearish sectors ({bear_count}): {', '.join(bearish[:4]) or 'None'}

Respond ONLY with valid JSON (no markdown):
{{
  "title": "SEO title under 65 chars — specific, includes week date",
  "slug": "url-slug-with-hyphens",
  "excerpt": "Meta description under 155 chars — specific market insight for this week"
}}"""

    try:
        text  = call_claude(prompt, max_tokens=200)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  SEO meta error: {e}")

    return {
        "title":   f"Weekly Market Analysis — {week_of}",
        "slug":    f"weekly-analysis-{datetime.utcnow().strftime('%Y-%m-%d')}",
        "excerpt": f"CallingMarkets weekly sector analysis. {bull_count} bullish, {bear_count} bearish sectors heading into this week.",
    }


def post_to_wordpress(title: str, slug: str, content: str, excerpt: str) -> str:
    """Post the article to WordPress and return the post URL."""
    endpoint = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/posts"

    # Get or create "Weekly Analysis" category
    cat_id = get_or_create_category("Weekly Analysis")

    payload = {
        "title":   title,
        "slug":    slug,
        "content": content,
        "excerpt": excerpt,
        "status":  "publish",
        "categories": [cat_id] if cat_id else [],
        "format":  "standard",
    }

    r = requests.post(
        endpoint,
        auth=(WP_USERNAME, WP_APP_PASSWORD),
        json=payload,
        timeout=30,
    )

    if r.status_code in (200, 201):
        post = r.json()
        url  = post.get("link", "")
        print(f"  ✓ Article published: {url}")
        return url
    else:
        print(f"  ✗ WordPress post failed: {r.status_code} {r.text[:200]}")
        return ""


def get_or_create_category(name: str) -> int | None:
    """Get existing category ID or create it."""
    endpoint = f"{WP_URL.rstrip('/')}/wp-json/wp/v2/categories"
    try:
        # Check if exists
        r = requests.get(endpoint, params={"search": name},
                         auth=(WP_USERNAME, WP_APP_PASSWORD), timeout=10)
        cats = r.json()
        if cats:
            return cats[0]["id"]
        # Create it
        r = requests.post(endpoint, auth=(WP_USERNAME, WP_APP_PASSWORD),
                          json={"name": name}, timeout=10)
        if r.status_code in (200, 201):
            return r.json()["id"]
    except Exception as e:
        print(f"  Category error: {e}")
    return None


def generate_takeaways(analyses: list) -> list:
    bullish = [a["sector"] for a in analyses if a["bias"] == "Bullish"]
    dist    = [a["sector"] for a in analyses if a["bias"] == "Distribution"]
    accum   = [a["sector"] for a in analyses if a["bias"] == "Accumulation"]
    bearish = [a["sector"] for a in analyses if a["bias"] == "Bearish"]
    today   = datetime.utcnow().strftime("%B %d, %Y")

    prompt = f"""Week of {today}. Sector bias: Bullish={', '.join(bullish[:6]) or 'None'}, Accumulation={', '.join(accum[:6]) or 'None'}, Distribution={', '.join(dist[:6]) or 'None'}, Bearish={', '.join(bearish[:6]) or 'None'}.

Write exactly 5 key market takeaways. Each: one actionable sentence under 20 words.
Respond ONLY with a JSON array: ["takeaway 1", "takeaway 2", "takeaway 3", "takeaway 4", "takeaway 5"]"""

    try:
        text  = call_claude(prompt, max_tokens=300)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  Takeaways error: {e}")
    return []


# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    today   = datetime.utcnow().strftime("%B %d, %Y")
    week_of = datetime.utcnow().strftime("%B %d, %Y")

    print(f"CallingMarkets Analysis Engine — {today}\n")

    # Load signals
    with open("signals.json", "r") as f:
        data = json.load(f)

    # Load fundamentals (optional — graceful if not present)
    fundamentals = {}
    try:
        with open("fundamentals.json", "r") as f:
            fd = json.load(f)
            fundamentals = fd.get("scores", {})
            print(f"Loaded fundamentals for {len(fundamentals)} stocks")
    except FileNotFoundError:
        print("fundamentals.json not found — skipping fundamental scores")
    except Exception as e:
        print(f"Fundamentals load warning: {e}")

    signals_data = data.get("signals", [])

    # Build signal lookup
    signals = {}
    for row in signals_data:
        signals[row["ticker"]] = {
            "daily":   row["timeframes"]["daily"]["signal"],
            "weekly":  row["timeframes"]["weekly"]["signal"],
            "monthly": row["timeframes"]["monthly"]["signal"],
            "price":   row.get("price"),
        }

    # Group tickers by sector (use first occurrence to avoid duplication)
    sectors: dict[str, list] = {}
    seen_in_sector: dict[str, set] = {}
    for row in signals_data:
        sec = row.get("sector", "Other")
        t   = row["ticker"]
        if t not in seen_in_sector.get(sec, set()):
            sectors.setdefault(sec, []).append(t)
            seen_in_sector.setdefault(sec, set()).add(t)

    print(f"Found {len(sectors)} sectors, {len(signals)} signal entries\n")

    # ── Compute sector biases ──────────────────────────────────────────────────
    analyses = []
    for sector, tickers in sorted(sectors.items()):
        bias, etf = get_etf_bias(sector, signals)
        if bias is None:
            bias = get_fallback_bias(tickers, signals)
            etf  = etf or "fallback"

        # Fetch news per sector
        query    = SECTOR_QUERIES.get(sector, sector)
        headlines = []
        try:
            headlines = fetch_news(query)
        except: pass

        # Generate sector synopsis
        etf_sig = signals.get(etf, {}) if etf else {}
        etf_line = f"{etf}: D={etf_sig.get('daily','?')} W={etf_sig.get('weekly','?')} M={etf_sig.get('monthly','?')}" if etf in signals else "N/A"
        non_etf  = [t for t in tickers if t != etf][:10]
        ticker_lines = [f"  {t}: D={signals.get(t,{}).get('daily','?')} W={signals.get(t,{}).get('weekly','?')} M={signals.get(t,{}).get('monthly','?')}" for t in non_etf]

        news_block = "\n".join(headlines) if headlines else "No headlines."
        tickers_block = "\n".join(ticker_lines)
        synopsis_prompt = (
            "CallingMarkets weekly sector synopsis. Date: " + today + "\n\n"
            + sector + " — Bias: " + bias + "\n"
            + "ETF Signal: " + etf_line + "\n"
            + "Key tickers:\n" + tickers_block + "\n\n"
            + "News:\n" + news_block + "\n\n"
            + "Write 2 tight paragraphs (100-150 words total). Lead with the bias, note ETF vs individual divergence, cite 1-2 news items, name 2-3 tickers to watch. Sharp prose, no bullets, no disclaimers."
        )

        try:
            synopsis = call_claude(synopsis_prompt, max_tokens=400)
        except Exception as e:
            print(f"  Synopsis error [{sector}]: {e}")
            synopsis = "Analysis unavailable."

        total       = len(tickers)
        daily_buy   = sum(1 for t in tickers if signals.get(t,{}).get("daily")   == "BUY")
        weekly_buy  = sum(1 for t in tickers if signals.get(t,{}).get("weekly")  == "BUY")
        monthly_buy = sum(1 for t in tickers if signals.get(t,{}).get("monthly") == "BUY")

        analyses.append({
            "sector":    sector,
            "bias":      bias,
            "bias_etf":  etf or "fallback",
            "bias_method": "etf" if etf and etf in signals else "average",
            "tickers":   tickers,
            "etf_signals": {
                "daily":   signals.get(etf,{}).get("daily",  "N/A"),
                "weekly":  signals.get(etf,{}).get("weekly", "N/A"),
                "monthly": signals.get(etf,{}).get("monthly","N/A"),
            } if etf else {},
            "signals": {
                "daily_buy":   daily_buy,
                "weekly_buy":  weekly_buy,
                "monthly_buy": monthly_buy,
                "total":       total,
            },
            "synopsis": synopsis,
        })

        print(f"  {sector}: {bias} (ETF: {etf})")

    # Attach sector_bias to each signal row for article generation
    sector_bias_map = {a["sector"]: a["bias"] for a in analyses}
    for row in signals_data:
        row["sector_bias"] = sector_bias_map.get(row.get("sector",""), "")

    # ── Generate takeaways ─────────────────────────────────────────────────────
    print("\nGenerating takeaways...")
    takeaways = generate_takeaways(analyses)

    # ── Fetch broad market news ───────────────────────────────────────────────
    print("Fetching market headlines...")
    market_news = fetch_news("stock market weekly outlook S&P 500 Fed", days_back=7)

    # ── Generate article ──────────────────────────────────────────────────────
    print("\nGenerating article...")
    article_html = generate_article(
        analyses, signals_data, signals, market_news, today, week_of, fundamentals
    )

    # ── Generate SEO meta ─────────────────────────────────────────────────────
    print("Generating SEO metadata...")
    bull_count = sum(1 for a in analyses if a["bias"] == "Bullish")
    bear_count = sum(1 for a in analyses if a["bias"] == "Bearish")
    meta = generate_seo_meta(analyses, bull_count, bear_count, week_of, today)

    # ── Post to WordPress ─────────────────────────────────────────────────────
    print(f"Posting to WordPress: {meta['title']}")
    post_url = post_to_wordpress(
        title   = meta["title"],
        slug    = meta["slug"],
        content = article_html,
        excerpt = meta["excerpt"],
    )

    # ── Save analysis.json ────────────────────────────────────────────────────
    output = {
        "generated":   datetime.utcnow().isoformat() + "Z",
        "week_of":     week_of,
        "bias_method": "sector_etf",
        "post_url":    post_url,
        "takeaways":   takeaways,
        "sectors":     analyses,
    }

    with open("analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ analysis.json written — {len(analyses)} sectors")
    print(f"✓ Article: {post_url or 'check WordPress'}")

    bias_counts = {}
    for a in analyses:
        bias_counts[a["bias"]] = bias_counts.get(a["bias"], 0) + 1
    print("\nBias summary:")
    for b, c in sorted(bias_counts.items()):
        print(f"  {b}: {c}")


if __name__ == "__main__":
    run()
