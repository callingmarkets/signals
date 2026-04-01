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

# ── Step 4: Claude generates CONTENT ONLY (not HTML) ─────────────────────────
def generate_content(analysis, flips):
    print("Generating content with Claude...")

    week_of  = analysis.get("week_of", "")
    takes    = analysis.get("takeaways", [])
    sectors  = analysis.get("sectors",  [])

    bias_groups = {"Bullish": [], "Distribution": [], "Accumulation": [], "Bearish": []}
    for s in sectors:
        bias = s.get("bias", "")
        if bias in bias_groups:
            bias_groups[bias].append(s["sector"])

    flip_summary = []
    for tf in ["weekly", "monthly"]:
        b = flips["bullish"][tf][:6]
        r = flips["bearish"][tf][:6]
        if b: flip_summary.append(("bull", tf.upper(), ", ".join(b)))
        if r: flip_summary.append(("bear", tf.upper(), ", ".join(r)))

    # Pick one sector per bias for highlights
    highlights = []
    for bias in ["Bullish", "Distribution", "Accumulation", "Bearish"]:
        match = next((s for s in sectors if s.get("bias") == bias), None)
        if match:
            synopsis = ". ".join(match["synopsis"].split(".")[:2]).strip()
            if not synopsis.endswith("."): synopsis += "."
            highlights.append({"bias": bias, "sector": match["sector"], "synopsis": synopsis})

    prompt = f"""You are writing content for the CallingMarkets weekly newsletter, week of {week_of}.

Provide ONLY these content fields as a JSON object, nothing else:

{{
  "subject": "<compelling subject line under 60 chars>",
  "takeaways": [
    {{"bold": "<first 4-6 words>", "rest": "<rest of sentence>"}}
    ... one object per takeaway
  ],
  "bullish_flips": "<comma separated tickers or 'None this week'>",
  "bearish_flips": "<comma separated tickers or 'None this week'>",
  "highlights": [
    {{"bias": "<bias>", "sector": "<sector name>", "synopsis": "<1-2 sentence synopsis, punchy and data-driven>"}}
    ... one per bias provided
  ]
}}

DATA:
Takeaways: {takes}
Bullish flips: {flips["bullish"]["weekly"][:6] + flips["bullish"]["monthly"][:3]}
Bearish flips: {flips["bearish"]["weekly"][:6] + flips["bearish"]["monthly"][:3]}
Highlights to write about:
{chr(10).join(f'- [{h["bias"]}] {h["sector"]}: {h["synopsis"]}' for h in highlights)}

Return ONLY valid JSON. No markdown. No explanation."""

    res = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": "claude-opus-4-5", "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]}
    )
    raw = res.json()["content"][0]["text"].strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    subject = data.get("subject", f"Market Outlook — Week of {week_of}")
    if TEST_MODE:
        subject = f"[TEST] {subject}"

    # ── Build HTML from locked template ───────────────────────────────────────
    week_of_a = analysis.get("week_of", "")

    # Takeaways
    takes_html = ""
    for t in data.get("takeaways", []):
        bold = t.get("bold", "")
        rest = t.get("rest", "")
        takes_html += f'''<tr><td style="padding:6px 0 6px 12px;font-size:13px;color:#374151;line-height:1.6;border-left:3px solid #0041FE;margin-bottom:8px;display:block;">
          <strong>{bold}</strong> {rest}</td></tr>'''

    # Bias columns
    def bias_col(label, cls_color, bg, count, names):
        shown = names[:5]
        more  = len(names) - 5
        items = "".join(f'<div style="font-size:11px;color:#374151;padding:2px 0;">{n}</div>' for n in shown)
        if more > 0: items += f'<div style="font-size:11px;color:#9ca3af;">+{more} more</div>'
        return f'''<td width="25%" style="padding:4px;">
          <div style="background:{bg};border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:22px;font-weight:700;color:{cls_color};">{count}</div>
            <div style="font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{cls_color};margin-bottom:8px;">{label}</div>
            {items}
          </div></td>'''

    bias_html = (
        bias_col("Bullish",      "#1d7a3a", "#eaf5ee", len(bias_groups["Bullish"]),      bias_groups["Bullish"])     +
        bias_col("Distribution", "#92620a", "#fef9ec", len(bias_groups["Distribution"]), bias_groups["Distribution"]) +
        bias_col("Accumulation", "#1d4ed8", "#eff6ff", len(bias_groups["Accumulation"]), bias_groups["Accumulation"]) +
        bias_col("Bearish",      "#c0392b", "#fdecea", len(bias_groups["Bearish"]),      bias_groups["Bearish"])
    )

    # Flips
    bull_flips = data.get("bullish_flips", "None this week")
    bear_flips = data.get("bearish_flips", "None this week")

    # Highlights
    bias_badge_style = {
        "Bullish":      "background:#eaf5ee;color:#1d7a3a;",
        "Distribution": "background:#fef9ec;color:#92620a;",
        "Accumulation": "background:#eff6ff;color:#1d4ed8;",
        "Bearish":      "background:#fdecea;color:#c0392b;",
    }
    highlights_html = ""
    for h in data.get("highlights", []):
        badge = bias_badge_style.get(h["bias"], "background:#f3f4f6;color:#374151;")
        highlights_html += f'''
        <tr><td style="padding:0 0 12px 0;">
          <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;">
            <table width="100%" cellpadding="0" cellspacing="0"><tr>
              <td style="font-size:14px;font-weight:700;color:#111827;">{h["sector"]}</td>
              <td align="right"><span style="font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;{badge}">{h["bias"]}</span></td>
            </tr></table>
            <p style="margin:10px 0 0;font-size:13px;color:#6b7280;line-height:1.6;">{h["synopsis"]}</p>
          </div>
        </td></tr>'''

    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:24px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

  <!-- HEADER -->
  <tr><td style="background:#0041FE;padding:32px 24px;text-align:center;border-radius:8px 8px 0 0;">
    <div style="font-size:28px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">CallingMarkets</div>
    <div style="font-size:14px;color:rgba(255,255,255,0.85);margin-top:6px;">Week of {week_of_a}</div>
  </td></tr>

  <!-- BODY -->
  <tr><td style="background:#ffffff;padding:28px 24px;">

    <!-- Key Takeaways -->
    <div style="font-size:18px;font-weight:700;color:#111827;padding-bottom:12px;border-bottom:2px solid #0041FE;margin-bottom:16px;">Key Takeaways</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
      {takes_html}
    </table>

    <!-- Market Bias Overview -->
    <div style="font-size:18px;font-weight:700;color:#111827;padding-bottom:12px;border-bottom:2px solid #0041FE;margin-bottom:16px;">Market Bias Overview</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
      <tr>{bias_html}</tr>
    </table>

    <!-- Signal Flips -->
    <div style="font-size:18px;font-weight:700;color:#111827;padding-bottom:12px;border-bottom:2px solid #0041FE;margin-bottom:16px;">Signal Flips This Week</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
      <tr>
        <td width="50%" style="padding-right:6px;">
          <div style="background:#eaf5ee;border-radius:8px;padding:16px;">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#1d7a3a;margin-bottom:8px;">▲ WEEKLY BULLISH FLIPS</div>
            <div style="font-size:13px;font-weight:700;color:#1d7a3a;">{bull_flips}</div>
          </div>
        </td>
        <td width="50%" style="padding-left:6px;">
          <div style="background:#fdecea;border-radius:8px;padding:16px;">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#c0392b;margin-bottom:8px;">▼ WEEKLY BEARISH FLIPS</div>
            <div style="font-size:13px;font-weight:700;color:#c0392b;">{bear_flips}</div>
          </div>
        </td>
      </tr>
    </table>

    <!-- Sector Highlights -->
    <div style="font-size:18px;font-weight:700;color:#111827;padding-bottom:12px;border-bottom:2px solid #0041FE;margin-bottom:16px;">Sector Highlights</div>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:8px;">
      {highlights_html}
    </table>

  </td></tr>

  <!-- FOOTER -->
  <tr><td style="background:#111827;padding:24px;text-align:center;border-radius:0 0 8px 8px;">
    <a href="https://callingmarkets.ai" style="font-size:16px;font-weight:700;color:#ffffff;text-decoration:none;">CallingMarkets</a>
    <div style="font-size:13px;color:#9ca3af;margin-top:6px;">Momentum signals for smarter market decisions.</div>
    <div style="margin-top:16px;font-size:12px;color:#6b7280;">You're receiving this as a CallingMarkets subscriber.</div>
    <div style="margin-top:6px;"><a href="{{unsubscribe_url}}" style="font-size:12px;color:#6b7280;">Unsubscribe</a></div>
  </td></tr>

</table>
</td></tr></table>
</body></html>'''

    print(f"  Subject: {subject}")
    print(f"  HTML size: {len(html.encode())/1024:.1f}KB")
    return subject, html


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
    subject, html_body  = generate_content(analysis, flips)
    sent, errors        = send_newsletter(users, subject, html_body)

    print(f"\nNewsletter complete. {sent} sent, {errors} errors.")

if __name__ == "__main__":
    run()
