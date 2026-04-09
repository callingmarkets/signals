"""
CallingMarkets Alert Engine
Runs after signal_engine.py completes.
Reads signals.json, finds flips, queries WordPress for watchlist subscribers,
sends personalized flip alert emails via Brevo.
"""

import json
import os
import requests
import base64
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
SIGNALS_URL     = "https://raw.githubusercontent.com/callingmarkets/signals/main/signals.json"
WP_URL          = os.environ["WP_URL"].rstrip("/")
WP_USERNAME     = os.environ["WP_USERNAME"]
WP_APP_PASSWORD = os.environ["WP_APP_PASSWORD"]
BREVO_API_KEY   = os.environ["BREVO_API_KEY"]
SITE_URL        = WP_URL
SIGNALS_PAGE    = f"{SITE_URL}/signals"

BREVO_SEND_URL  = "https://api.brevo.com/v3/smtp/email"
FROM_EMAIL      = "alerts@callingmarkets.ai"
FROM_NAME       = "CallingMarkets"

TIMEFRAME_LABELS = {
    "daily":   "Daily",
    "weekly":  "Weekly",
    "monthly": "Monthly",
}

# ── AUTH ──────────────────────────────────────────────────────────────────────
def wp_auth_headers():
    token = base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


# ── FETCH SIGNALS ─────────────────────────────────────────────────────────────
def fetch_signals():
    r = requests.get(SIGNALS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {s["ticker"]: s for s in data.get("signals", [])}


# ── FIND FLIPS ────────────────────────────────────────────────────────────────
def find_flips(signals: dict) -> list:
    """Return list of {ticker, timeframe, from, to, sector} for every flip."""
    flips = []
    for ticker, row in signals.items():
        for tf, label in TIMEFRAME_LABELS.items():
            tf_data = row.get("timeframes", {}).get(tf, {})
            curr = tf_data.get("signal")
            prev = tf_data.get("previous")
            if curr and prev and curr != prev and prev not in ("N/A", "ERR") and curr not in ("N/A", "ERR"):
                flips.append({
                    "ticker":    ticker,
                    "timeframe": label,
                    "tf_key":    tf,
                    "from":      prev,
                    "to":        curr,
                    "sector":    row.get("sector", ""),
                    "price":     row.get("price"),
                })
    return flips


# ── FETCH ALL WATCHLISTS ──────────────────────────────────────────────────────
def fetch_all_watchlists() -> list:
    """Returns [{email, name, tickers, alerts_enabled}] from WordPress."""
    url = f"{WP_URL}/wp-json/cm/v1/watchlist/all"
    try:
        r = requests.get(url, headers=wp_auth_headers(), timeout=15)
        if r.status_code == 200:
            return r.json().get("watchlists", [])
        else:
            print(f"  ✗ Watchlist fetch failed: {r.status_code}")
            return []
    except Exception as e:
        print(f"  ✗ Watchlist fetch error: {e}")
        return []


# ── BUILD EMAIL ───────────────────────────────────────────────────────────────
def build_email_html(name: str, user_flips: list) -> str:
    today = datetime.utcnow().strftime("%B %d, %Y")

    flip_rows = ""
    for f in user_flips:
        direction = "▲ BUY" if f["to"] == "BUY" else "▼ SELL"
        dir_color = "#22c55e" if f["to"] == "BUY" else "#ef4444"
        dir_bg    = "rgba(34,197,94,.1)" if f["to"] == "BUY" else "rgba(239,68,68,.1)"
        prev_color = "#ef4444" if f["from"] == "BUY" else "#22c55e"
        flip_rows += f"""
        <tr>
          <td style="padding:10px 12px;font-weight:700;font-size:14px;color:#f9fafb;">{f['ticker']}</td>
          <td style="padding:10px 12px;font-size:12px;color:#9ca3af;">{f['sector']}</td>
          <td style="padding:10px 12px;font-size:12px;color:#9ca3af;">{f['timeframe']}</td>
          <td style="padding:10px 12px;">
            <span style="font-size:11px;font-weight:600;color:{prev_color};background:rgba(156,163,175,.1);padding:3px 8px;border-radius:4px;">{f['from']}</span>
            <span style="font-size:11px;color:#4b5563;margin:0 6px;">→</span>
            <span style="font-size:11px;font-weight:700;color:{dir_color};background:{dir_bg};padding:3px 8px;border-radius:4px;">{direction}</span>
          </td>
          <td style="padding:10px 12px;font-size:12px;color:#9ca3af;">{f'${f["price"]:,.2f}' if f.get('price') else '—'}</td>
        </tr>"""

    count = len(user_flips)
    summary = f"{count} signal flip{'s' if count != 1 else ''} on your watchlist"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0a0a0a;font-family:'DM Sans',-apple-system,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a;">
  <tr><td align="center" style="padding:40px 20px;">
    <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

      <!-- Header -->
      <tr><td style="padding:0 0 28px;">
        <p style="margin:0 0 4px;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#f97316;">CallingMarkets</p>
        <h1 style="margin:0 0 6px;font-size:22px;font-weight:700;color:#f9fafb;letter-spacing:-0.4px;">Signal Flip Alert</h1>
        <p style="margin:0;font-size:13px;color:#9ca3af;">{today} &nbsp;·&nbsp; {summary}</p>
      </td></tr>

      <!-- Table -->
      <tr><td style="background:#121212;border-radius:10px;border:1px solid #1f1f1f;overflow:hidden;">
        <table width="100%" cellpadding="0" cellspacing="0">
          <thead>
            <tr style="border-bottom:1px solid #1f1f1f;">
              <th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#4b5563;">Ticker</th>
              <th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#4b5563;">Sector</th>
              <th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#4b5563;">Timeframe</th>
              <th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#4b5563;">Flip</th>
              <th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#4b5563;">Price</th>
            </tr>
          </thead>
          <tbody style="border-collapse:collapse;">
            {flip_rows}
          </tbody>
        </table>
      </td></tr>

      <!-- CTA -->
      <tr><td style="padding:24px 0 0;text-align:center;">
        <a href="{SIGNALS_PAGE}" style="display:inline-block;background:#f97316;color:#fff;font-size:13px;font-weight:700;padding:12px 28px;border-radius:6px;text-decoration:none;">View Full Dashboard →</a>
      </td></tr>

      <!-- Footer -->
      <tr><td style="padding:28px 0 0;text-align:center;font-size:11px;color:#4b5563;line-height:1.6;">
        You're receiving this because you have alerts enabled on your CallingMarkets watchlist.<br>
        <a href="{SITE_URL}/account" style="color:#4b5563;">Manage alerts</a>
      </td></tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""


# ── SEND EMAIL ────────────────────────────────────────────────────────────────
def send_alert(email: str, name: str, user_flips: list) -> bool:
    count  = len(user_flips)
    tickers = ", ".join(f["ticker"] for f in user_flips[:3])
    extra  = f" +{count - 3} more" if count > 3 else ""
    subject = f"Signal Alert: {tickers}{extra} flipped"

    payload = {
        "sender":   {"name": FROM_NAME, "email": FROM_EMAIL},
        "to":       [{"email": email, "name": name or email}],
        "subject":  subject,
        "htmlContent": build_email_html(name or "there", user_flips),
    }
    try:
        r = requests.post(
            BREVO_SEND_URL,
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        if r.status_code in (200, 201):
            return True
        else:
            print(f"  ✗ Email to {email} failed: {r.status_code} {r.text[:100]}")
            return False
    except Exception as e:
        print(f"  ✗ Email error: {e}")
        return False


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"CallingMarkets Alert Engine — {today}\n")

    print("Fetching signals...")
    signals = fetch_signals()
    print(f"  Loaded {len(signals)} tickers")

    print("Finding flips...")
    flips = find_flips(signals)
    print(f"  Found {len(flips)} flip(s) today")

    if not flips:
        print("  No flips — no alerts to send.")
        return

    # Print flip summary
    for f in flips:
        print(f"  {f['ticker']:10} {f['timeframe']:8} {f['from']} → {f['to']}")

    print("\nFetching watchlists from WordPress...")
    watchlists = fetch_all_watchlists()
    print(f"  Found {len(watchlists)} users with watchlists")

    if not watchlists:
        print("  No watchlists — no alerts to send.")
        return

    # Build flip lookup: ticker → list of flips
    flip_by_ticker: dict[str, list] = {}
    for f in flips:
        flip_by_ticker.setdefault(f["ticker"], []).append(f)

    # Match flips to users
    sent = 0
    skipped = 0
    for user in watchlists:
        email   = user.get("email", "")
        name    = user.get("name", "")
        tickers = user.get("tickers", [])
        alerts  = user.get("alerts_enabled", True)  # default true if field missing

        if not email or not alerts:
            skipped += 1
            continue

        # Find flips for this user's watchlist
        user_flips = []
        for ticker in tickers:
            if ticker in flip_by_ticker:
                user_flips.extend(flip_by_ticker[ticker])

        if not user_flips:
            continue

        print(f"  Sending {len(user_flips)} flip(s) to {email}...")
        ok = send_alert(email, name, user_flips)
        if ok:
            sent += 1
            print(f"    ✓ Sent")

    print(f"\n✓ Alert engine complete — {sent} emails sent, {skipped} skipped (alerts disabled)")


if __name__ == "__main__":
    main()
