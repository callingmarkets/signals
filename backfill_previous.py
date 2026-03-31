"""
One-time backfill script.
Sets previous = current for all tickers/timeframes as a clean baseline.
Future real flips will update previous naturally via signal_engine.py.
"""

import json

with open("signals.json", "r") as f:
    data = json.load(f)

updated = 0
for row in data["signals"]:
    for label in ["daily", "weekly", "monthly"]:
        tf = row["timeframes"].get(label, {})
        if tf.get("signal") in ("BUY", "SELL"):
            tf["previous"] = tf["signal"]
            updated += 1

with open("signals.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Backfilled {updated} signals — previous set to match current.")
