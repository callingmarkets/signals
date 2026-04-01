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

    # Limit synopses to 2 sentences each to keep email under Gmail's 102KB clip limit
    sector_highlights = []
    for s in sectors[:4]:
        synopsis_short = ". ".join(s["synopsis"].split(".")[:2]).strip()
        if not synopsis_short.endswith("."): synopsis_short += "."
        sector_highlights.append(f"[{s['bias']}] {s['sector']}: {synopsis_short}")

    prompt = f"""Write a weekly CallingMarkets newsletter for the week of {week_of}.

CallingMarkets tracks momentum signals (EMA/RSI/MACD) across Daily, Weekly, Monthly timeframes.
Bias classifications: Bullish (monthly+weekly BUY), Distribution (monthly BUY + weekly SELL), Accumulation (monthly SELL + weekly BUY), Bearish (monthly+weekly SELL).

DATA:

TAKEAWAYS:
{chr(10).join(f"- {t}" for t in takes)}

SECTOR BIAS:
- Bullish ({len(bias_groups['Bullish'])}): {', '.join(bias_groups['Bullish']) or 'None'}
- Distribution ({len(bias_groups['Distribution'])}): {', '.join(bias_groups['Distribution']) or 'None'}
- Accumulation ({len(bias_groups['Accumulation'])}): {', '.join(bias_groups['Accumulation']) or 'None'}
- Bearish ({len(bias_groups['Bearish'])}): {', '.join(bias_groups['Bearish']) or 'None'}

FLIPS:
{chr(10).join(flip_summary) or 'No significant flips'}

SECTOR HIGHLIGHTS (use exactly these, keep each to 1-2 sentences):
{chr(10).join(sector_highlights)}

Write a COMPLETE HTML email. Rules:
- First line must be: SUBJECT: <subject line>
- Inline CSS only, max-width 600px
- Brand colors: green #1d7a3a, red #c0392b, amber #d97706, blue #1d4ed8, accent #0041FE
- MUST include ALL these sections in order:
  1. Header — dark blue (#0041FE) background, white "CallingMarkets" bold, week date subtitle
  2. Key Takeaways — white box, blue left border, bullet points
  3. Market Bias Overview — 4 side-by-side boxes (Bullish=green, Distribution=amber, Accumulation=blue, Bearish=red), show count + sector list, "+X more" if over 5
  4. Signal Flips — two side-by-side boxes, green bullish left / red bearish right, ticker list
  5. Sector Highlights — 3-4 sector cards, bias badge, 1-2 sentence synopsis
  6. Footer — dark background, callingmarkets.ai link, "You're receiving this as a CallingMarkets subscriber", unsubscribe link placeholder: <a href="{{{{unsubscribe_url}}}}">Unsubscribe</a>
- Keep total HTML under 80KB. Be concise in copy. No filler sentences.
- Return ONLY the HTML. No markdown. No explanation."""

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
