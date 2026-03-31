"""
One-time backfill script.
Forces previous = opposite of current for ALL tickers/timeframes.
Run once, then the rolling logic in signal_engine.py takes over.
"""

import json

with open("signals.json", "r") as f:
    data = json.load(f)

updated = 0
for row in data["signals"]:
    for label in ["daily", "weekly", "monthly"]:
        tf = row["timeframes"].get(label, {})
        if tf.get("signal") in ("BUY", "SELL"):
            tf["previous"] = "SELL" if tf["signal"] == "BUY" else "BUY"
            updated += 1

with open("signals.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Backfilled {updated} signals with opposite previous.")
