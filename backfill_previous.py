"""
One-time backfill script.
Sets previous = opposite of current for any ticker/timeframe where previous is null.
Run once manually, then the rolling logic takes over from signal_engine.py.
"""

import json

with open("signals.json", "r") as f:
    data = json.load(f)

updated = 0
for row in data["signals"]:
    for label in ["daily", "weekly", "monthly"]:
        tf = row["timeframes"].get(label, {})
        if tf.get("previous") is None and tf.get("signal") in ("BUY", "SELL"):
            tf["previous"] = "SELL" if tf["signal"] == "BUY" else "BUY"
            updated += 1

with open("signals.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Backfilled {updated} null previous signals.")
