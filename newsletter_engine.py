#!/usr/bin/env python3
"""
CallingMarkets Newsletter Engine
Runs after weekly analysis — fetches WordPress users, generates email via Claude,
sends via Brevo API.
"""

import os
import json
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
WP_URL          = os.environ["WP_URL"]               # e.g. https://callingmarkets.ai
WP_USERNAME     = os.environ["WP_USERNAME"]
WP_APP_PASSWORD = os.environ["WP_APP_PASSWORD"]
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
BREVO_KEY       = os.environ["BREVO_API_KEY"]
SIGNALS_URL     = "https://raw.githubusercontent.com/callingmarkets/signals/main/signals.json"
ANALYSIS_URL    = "https://raw.githubusercontent.com/callingmarkets/signals/main/analysis.json"

FROM_EMAIL      = "newsletter@callingmarkets.ai"
FROM_NAME       = "Calling Markets"
SUBJECT_PREFIX  = "Market Outlook"

# ── Test mode ─────────────────────────────────────────────────────────────────
TEST_EMAIL      = os.environ.get("TEST_EMAIL", "").strip()
TEST_MODE       = bool(TEST_EMAIL)
REPLY_TO_EMAIL  = FROM_EMAIL

# ── Step 1: Fetch WordPress subscribers ───────────────────────────────────────
def get_wp_users():
    if TEST_MODE:
        print(f"TEST MODE — sending only to: {TEST_EMAIL}")
        return [{"email": TEST_EMAIL, "name": "Test User"}]

    print("Fetching WordPress users...")
    users   = []
    page    = 1
    auth    = (WP_USERNAME, WP_APP_PASSWORD)
    while True:
        res = requests.get(
            f"{WP_URL}/wp-json/wp/v2/users",
            params={"per_page": 100, "page": page},
            auth=auth
        )
        if res.status_code != 200:
            print(f"  WP users error: {res.status_code} {res.text[:200]}")
            break
        batch = res.json()
        if not batch:
            break
        for u in batch:
            email = u.get("email") or u.get("user_email")
            name  = u.get("name") or u.get("display_name") or ""
            if email:
                users.append({"email": email, "name": name})
        print(f"  Page {page}: {len(batch)} users")
        if len(batch) < 100:
            break
        page += 1
    print(f"Total subscribers: {len(users)}")
    return users

# ── Step 2: Load analysis data ─────────────────────────────────────────────────
def load_data():
    print("Loading analysis.json...")
    analysis = requests.get(ANALYSIS_URL + f"?t={int(datetime.now().timestamp())}").json()
    print("Loading signals.json...")
    signals  = requests.get(SIGNALS_URL  + f"?t={int(datetime.now().timestamp())}").json()
    return analysis, signals

# ── Step 3: Find flips from signals.json ───────────────────────────────────────
def get_flips(signals_data):
    flips = {"bullish": {"daily": [], "weekly": [], "monthly": []},
             "bearish":  {"daily": [], "weekly": [], "monthly": []}}
    for row in signals_data.get("signals", []):
        ticker = row["ticker"]
        for tf in ["daily", "weekly", "monthly"]:
            t    = row.get("timeframes", {}).get(tf, {})
            curr = t.get("signal")
            prev = t.get("previous")
            if not curr or not prev or curr == prev:
                continue
            if curr == "BUY"  and prev == "SELL":
                flips["bullish"][tf].append(ticker)
            if curr == "SELL" and prev == "BUY":
                flips["bearish"][tf].append(ticker)
    return flips

# ── Step 4: Generate email HTML via Claude ─────────────────────────────────────
def generate_email(analysis, flips):
    print("Generating email with Claude...")

    week_of  = analysis.get("week_of", "")
    takes    = analysis.get("takeaways", [])
    sectors  = analysis.get("sectors",  [])

    # Build sector summary grouped by bias
    bias_groups = {"Bullish": [], "Distribution": [], "Accumulation": [], "Bearish": []}
    for s in sectors:
        bias = s.get("bias", "")
        if bias in bias_groups:
            bias_groups[bias].append(s["sector"])

    # Build flip summary
    flip_summary = []
    for tf in ["weekly", "monthly"]:
        b = flips["bullish"][tf][:5]
        r = flips["bearish"][tf][:5]
        if b: flip_summary.append(f"Weekly Bullish Flips: {', '.join(b)}")
        if r: flip_summary.append(f"Weekly Bearish Flips: {', '.join(r)}")

    # Trim synopses to 2 sentences to stay under Gmail 102KB clip limit
    sector_highlights = []
    for s in sectors[:4]:
        synopsis_short = ". ".join(s["synopsis"].split(".")[:2]).strip()
        if not synopsis_short.endswith("."): synopsis_short += "."
        sector_highlights.append({"bias": s["bias"], "sector": s["sector"], "synopsis": synopsis_short})

    prompt = f"""You are writing the weekly CallingMarkets newsletter for the week of {week_of}.

CallingMarkets uses a momentum signal system (EMA, RSI, MACD) across Daily, Weekly, and Monthly timeframes.
Sectors are classified as: Bullish (monthly+weekly buy), Distribution (monthly buy + weekly sell),
Accumulation (monthly sell + weekly buy), or Bearish (monthly+weekly sell).

Here is this week's data:

KEY TAKEAWAYS:
{chr(10).join(f"- {t}" for t in takes)}

SECTOR BIAS BREAKDOWN:
- Bullish ({len(bias_groups["Bullish"])}): {", ".join(bias_groups["Bullish"]) or "None"}
- Distribution ({len(bias_groups["Distribution"])}): {", ".join(bias_groups["Distribution"]) or "None"}
- Accumulation ({len(bias_groups["Accumulation"])}): {", ".join(bias_groups["Accumulation"]) or "None"}
- Bearish ({len(bias_groups["Bearish"])}): {", ".join(bias_groups["Bearish"]) or "None"}

SIGNAL FLIPS THIS WEEK:
{chr(10).join(flip_summary) or "No significant flips this week"}

SECTOR HIGHLIGHTS (4 sectors, use exactly this data):
{chr(10).join(f'[{s["bias"]}] {s["sector"]}: {s["synopsis"]}' for s in sector_highlights)}

Write a complete, professional HTML email newsletter matching this EXACT design:

DESIGN SPEC:
- Subject line first line: SUBJECT: <subject>
- Inline CSS only. Max-width 600px. White background #ffffff. Font: Arial, sans-serif.
- Brand colors: accent blue #0041FE, green #1d7a3a, green-bg #eaf5ee, red #c0392b, red-bg #fdecea, amber #92620a, amber-bg #fef9ec, blue #1d4ed8, blue-bg #eff6ff

SECTION 1 — HEADER:
- Full-width block, background #0041FE
- "CallingMarkets" in white, bold, 28px, centered
- "Week of {week_of}" in white, 14px, centered, below title
- Padding 32px 24px

SECTION 2 — KEY TAKEAWAYS:
- White card, 1px border #e5e7eb, border-radius 8px, padding 24px, margin 24px auto
- "Key Takeaways" heading, 18px bold, color #111827, blue bottom border 2px #0041FE, padding-bottom 12px
- Each takeaway as a list item with a filled blue circle bullet (#0041FE), 13px, line-height 1.6, color #374151
- Bold the first 4-6 words of each takeaway

SECTION 3 — MARKET BIAS OVERVIEW:
- "Market Bias Overview" heading same style as above
- 4 side-by-side boxes in a table (25% each), border-radius 8px, padding 12px
- Bullish box: bg #eaf5ee, label "BULLISH" color #1d7a3a, count in large bold green
- Distribution box: bg #fef9ec, label "DISTRIBUTION" color #92620a, count in large bold amber  
- Accumulation box: bg #eff6ff, label "ACCUMULATION" color #1d4ed8, count in large bold blue
- Bearish box: bg #fdecea, label "BEARISH" color #c0392b, count in large bold red
- Show up to 5 sector names in small text (11px), then "+X more" if over 5

SECTION 4 — SIGNAL FLIPS:
- "Signal Flips This Week" heading same style
- Two side-by-side boxes in a table (50% each)
- Left box: bg #eaf5ee, "▲ WEEKLY BULLISH FLIPS" label in #1d7a3a, ticker list in bold #1d7a3a
- Right box: bg #fdecea, "▼ WEEKLY BEARISH FLIPS" label in #c0392b, ticker list in bold #c0392b
- If no flips: show "None this week" in muted text

SECTION 5 — SECTOR HIGHLIGHTS:
- "Sector Highlights" heading same style
- 4 sector cards stacked vertically, each: white bg, 1px border #e5e7eb, border-radius 8px, padding 16px, margin-bottom 12px
- Top row: sector name bold left, bias badge pill right (same colors as bias overview)
- Synopsis text below: 13px, color #6b7280, line-height 1.6
- This section is REQUIRED — do not omit it

SECTION 6 — FOOTER:
- Full-width block, background #111827, padding 24px, text-align center
- "CallingMarkets" in white bold, link to https://callingmarkets.ai
- "Momentum signals for smarter market decisions." in #9ca3af, 13px
- "You're receiving this as a CallingMarkets subscriber." in #6b7280, 12px, margin-top 12px
- Unsubscribe link: <a href="{{{{unsubscribe_url}}}}" style="color:#6b7280;font-size:12px;">Unsubscribe</a>

IMPORTANT: Keep total HTML under 80KB. Return ONLY the HTML, no markdown, no explanation."""

    res = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": "claude-opus-4-5", "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
    )
    text = res.json()["content"][0]["text"].strip()

    # Extract subject line
    subject = f"{SUBJECT_PREFIX} — Week of {week_of}"
    if text.startswith("SUBJECT:"):
        lines   = text.split("\n", 1)
        subject = lines[0].replace("SUBJECT:", "").strip()
        text    = lines[1].strip() if len(lines) > 1 else text

    if TEST_MODE:
        subject = f"[TEST] {subject}"
    print(f"  Subject: {subject}")
    return subject, text

# ── Step 5: Send via Brevo ─────────────────────────────────────────────────────
def send_newsletter(users, subject, html_body):
    print(f"Sending to {len(users)} subscribers via Brevo...")
    sent = 0
    errors = 0

    # Brevo supports batch sending up to 50 recipients per call
    batch_size = 50
    for i in range(0, len(users), batch_size):
        batch = users[i:i+batch_size]
        to    = [{"email": u["email"], "name": u["name"]} for u in batch]

        payload = {
            "sender":      {"name": FROM_NAME, "email": FROM_EMAIL},
            "replyTo":     {"email": REPLY_TO_EMAIL},
            "to":          to,
            "subject":     subject,
            "htmlContent": html_body,
        }

        res = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": BREVO_KEY, "content-type": "application/json"},
            json=payload
        )

        if res.status_code in (200, 201, 202):
            sent   += len(batch)
            print(f"  Batch {i//batch_size + 1}: sent to {len(batch)} recipients")
        else:
            errors += len(batch)
            print(f"  Batch {i//batch_size + 1} ERROR: {res.status_code} {res.text[:300]}")

    print(f"\nDone — Sent: {sent}, Errors: {errors}")
    return sent, errors

# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print(f"CallingMarkets Newsletter Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    users               = get_wp_users()
    if not users:
        print("No subscribers found. Exiting.")
        return

    analysis, signals   = load_data()
    flips               = get_flips(signals)
    subject, html_body  = generate_email(analysis, flips)
    sent, errors        = send_newsletter(users, subject, html_body)

    print(f"\nNewsletter complete. {sent} sent, {errors} errors.")

if __name__ == "__main__":
    run()
