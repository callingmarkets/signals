#!/usr/bin/env python3
"""
CallingMarkets Tweet Engine
Runs after signal engine — posts daily market update to X (Twitter).
"""

import os
import json
import hmac
import hashlib
import time
import random
import string
import urllib.parse
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_KEY        = os.environ["ANTHROPIC_API_KEY"]
TWITTER_API_KEY      = os.environ["TWITTER_API_KEY"]
TWITTER_API_SECRET   = os.environ["TWITTER_API_SECRET"]
TWITTER_ACCESS_TOKEN = os.environ["TWITTER_ACCESS_TOKEN"]
TWITTER_ACCESS_SECRET= os.environ["TWITTER_ACCESS_SECRET"]

SIGNALS_URL  = "https://raw.githubusercontent.com/callingmarkets/signals/main/signals.json"
ANALYSIS_URL = "https://raw.githubusercontent.com/callingmarkets/signals/main/analysis.json"
SITE_URL     = "https://callingmarkets.ai"

# ── OAuth 1.0a signing ────────────────────────────────────────────────────────
def oauth_header(method, url, params={}):
    nonce     = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    timestamp = str(int(time.time()))

    oauth_params = {
        "oauth_consumer_key":     TWITTER_API_KEY,
        "oauth_nonce":            nonce,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp":        timestamp,
        "oauth_token":            TWITTER_ACCESS_TOKEN,
        "oauth_version":          "1.0",
    }

    # Combine all params for signature base
    all_params = {**params, **oauth_params}
    sorted_params = "&".join(
        f"{urllib.parse.quote(k, safe='')  }={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted(all_params.items())
    )
    base_string = "&".join([
        method.upper(),
        urllib.parse.quote(url, safe=""),
        urllib.parse.quote(sorted_params, safe=""),
    ])
    signing_key = f"{urllib.parse.quote(TWITTER_API_SECRET, safe='')}&{urllib.parse.quote(TWITTER_ACCESS_SECRET, safe='')}"
    signature   = hmac.new(
        signing_key.encode(), base_string.encode(), hashlib.sha1
    ).digest()
    import base64
    oauth_params["oauth_signature"] = base64.b64encode(signature).decode()

    header = "OAuth " + ", ".join(
        f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(str(v), safe="")}"'
        for k, v in sorted(oauth_params.items())
    )
    return header

# ── Post tweet ────────────────────────────────────────────────────────────────
def post_tweet(text):
    url     = "https://api.twitter.com/2/tweets"
    payload = {"text": text}
    header  = oauth_header("POST", url)
    res     = requests.post(
        url,
        headers={"Authorization": header, "Content-Type": "application/json"},
        json=payload
    )
    if res.status_code in (200, 201):
        tweet_id = res.json().get("data", {}).get("id")
        print(f"  ✓ Tweet posted: https://x.com/i/web/status/{tweet_id}")
        return True
    else:
        print(f"  ✗ Tweet failed: {res.status_code} {res.text}")
        return False

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    signals  = requests.get(SIGNALS_URL  + f"?t={int(time.time())}").json()
    try:
        analysis = requests.get(ANALYSIS_URL + f"?t={int(time.time())}").json()
    except:
        analysis = {}
    return signals, analysis

# ── Find notable flips ────────────────────────────────────────────────────────
def get_flips(signals_data):
    bull, bear = [], []
    for row in signals_data.get("signals", []):
        ticker = row["ticker"]
        # Weekly and monthly flips only — daily is noise
        for tf in ["weekly", "monthly"]:
            t    = row.get("timeframes", {}).get(tf, {})
            curr = t.get("signal")
            prev = t.get("previous")
            if not curr or not prev or curr == prev:
                continue
            if curr == "BUY"  and prev == "SELL": bull.append((ticker, tf))
            if curr == "SELL" and prev == "BUY":  bear.append((ticker, tf))
    return bull, bear

# ── Generate tweet via Claude ─────────────────────────────────────────────────
def generate_tweet(signals_data, analysis):
    bull_flips, bear_flips = get_flips(signals_data)

    # Bias counts from analysis
    sectors      = analysis.get("sectors", [])
    bias_counts  = {"Bullish": 0, "Distribution": 0, "Accumulation": 0, "Bearish": 0}
    for s in sectors:
        b = s.get("bias", "")
        if b in bias_counts:
            bias_counts[b] += 1

    total_tickers = len(signals_data.get("signals", []))
    daily_buy     = sum(1 for r in signals_data.get("signals", [])
                        if r.get("timeframes", {}).get("daily", {}).get("signal") == "BUY")
    daily_pct     = round(daily_buy / total_tickers * 100) if total_tickers else 0

    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You write sharp, opinionated market tweets for @CallingMarkets — a momentum signal platform tracking 214 tickers using EMA/RSI/MACD signals.

Today is {today}.

SIGNAL DATA:
- Daily: {daily_buy}/{total_tickers} tickers ({daily_pct}%) on BUY signal
- Sector bias — Bullish: {bias_counts['Bullish']}, Distribution: {bias_counts['Distribution']}, Accumulation: {bias_counts['Accumulation']}, Bearish: {bias_counts['Bearish']}
- Bullish flips today: {[f[0] for f in bull_flips[:5]] or 'None'}
- Bearish flips today: {[f[0] for f in bear_flips[:5]] or 'None'}

Write ONE tweet (max 240 chars including the link). Rules:
- Give ONE clear, actionable take — not a summary of data points
- The take should tell traders what to DO or WATCH, not just what happened
- Examples of good takes: "Energy is the only sector worth being long right now", "Momentum says stay defensive until weekly signals recover", "Gold miners just flipped — this is where capital is hiding"
- Use $ before tickers sparingly — max 2 tickers per tweet
- Use emojis sparingly — 1 max, only where meaningful
- End with: {SITE_URL}
- No hashtags
- Do NOT start with "Today" or "The market" or list multiple data points
- Sound like a sharp trader with conviction, not a data terminal
- Return ONLY the tweet text, nothing else"""

    res = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": "claude-opus-4-5", "max_tokens": 300,
              "messages": [{"role": "user", "content": prompt}]}
    )
    tweet = res.json()["content"][0]["text"].strip()

    # Safety check — truncate if over 280
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    print(f"  Generated tweet ({len(tweet)} chars):\n  {tweet}")
    return tweet

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print(f"CallingMarkets Tweet Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    print("Loading signal data...")
    signals_data, analysis = load_data()
    print(f"  {len(signals_data.get('signals', []))} tickers loaded")

    print("Generating tweet...")
    tweet = generate_tweet(signals_data, analysis)

    print("Posting to X...")
    post_tweet(tweet)

if __name__ == "__main__":
    run()
