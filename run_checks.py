import argparse
from collections import Counter
import json
from pathlib import Path

from alert_system.data import load_data
from alert_system.detectors import AlertEngine


def run(sample):
    data = load_data({"source": "bundled", "sample": sample})
    engine = AlertEngine()
    alerts = []
    for event in data.events:  # Source arrival order, one record at a time
        alerts += engine.process(event)
    return data, alerts


def main():
    parser = argparse.ArgumentParser(description="Run the data-quality checks on a bundled Databento sample")
    parser.add_argument("--sample", choices=("quotes", "trades"), default="quotes")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the alert list")
    args = parser.parse_args()
    data, alerts = run(args.sample)
    print(data.metadata["label"])
    print(f"Records checked: {len(data.events):,}")
    print(f"Symbols: {', '.join(data.metadata['symbols'])}")
    print(f"Alerts: {len(alerts)}")
    for rule_id, count in sorted(Counter(alert["rule_id"] for alert in alerts).items()):
        print(f"  {rule_id}: {count}")
    if args.output:
        args.output.write_text(json.dumps(alerts, indent=2, allow_nan=False) + "\n")
        print(f"Saved alerts to {args.output}")


if __name__ == "__main__":
    main()
