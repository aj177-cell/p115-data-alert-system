"""Run the Milestone 1 alert engine and the v3 detectors on the same data.

python run_combined.py                        # bundled AAPL quotes (same input as run_checks.py)
python run_combined.py --sample trades
python run_combined.py --local 2026-11-12/153000_123456_abcdef/data.parquet --symbol AAPL
python run_combined.py --stress               # v3 synthetic stream + labeled faults, both scored
python run_combined.py --output-dir out/      # native outputs of both systems + unified table

run_checks.py is unchanged and still runs Milestone 1 alone.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from integration.combine import BATCH, ENGINE, bundled, local, run_combined
from integration.stress import stress_test

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def counts(unified):
    if unified.empty:
        return "  (none)"
    table = unified.assign(subtype=unified["subtype"].fillna("")).groupby(["source", "rule", "subtype", "severity"]).size()
    return table.rename("findings").to_frame().to_string()


def save(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{ENGINE}_alerts.json").write_text(json.dumps(result.alerts, indent=2, allow_nan=False) + "\n")
    result.flags.to_parquet(folder / f"{BATCH}_flags.parquet")
    result.unified.to_csv(folder / "combined_findings.csv", index=False)
    result.overlap.to_csv(folder / "overlap.csv", index=False)
    print(f"Saved native outputs, combined_findings.csv and overlap.csv to {folder}")


def main():
    parser = argparse.ArgumentParser(description="Milestone 1 engine + detectors v3 on the same records")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", choices=("quotes", "trades"), default="quotes")
    source.add_argument("--local", help="Parquet file under data/, e.g. written by download_and_show()")
    source.add_argument("--stress", action="store_true", help="v3 synthetic stream with injected faults")
    parser.add_argument("--symbol", help="--local only: keep one symbol")
    parser.add_argument("--start", help="--local only: first ts_recv (naive = New York time)")
    parser.add_argument("--end", help="--local only: end of ts_recv window (exclusive)")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.stress:
        result = stress_test()
        print(f"Stress test: v3 synthetic mbp-1 stream, {result.records:,} records, {len(result.labels)} labeled faults")
        print("Engine scored with cooldown off; findings at severity 'info' ignored; ±1 s matching.\n")
        print("Labels detected per fault family (None = that system has no rule for it):")
        table = result.detection.astype("Int64").astype(object)
        print(table.where(result.detection.notna(), "—").to_string())
        print("\nFindings per rule (faulted stream) and the same rules on the clean stream:")
        print(result.breakdown.to_string())
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            result.unified.to_csv(args.output_dir / "stress_findings.csv", index=False)
            result.labels.to_csv(args.output_dir / "stress_labels.csv", index=False)
            result.detection.to_csv(args.output_dir / "stress_detection.csv")
            result.breakdown.to_csv(args.output_dir / "stress_breakdown.csv")
            print(f"\nSaved stress_*.csv to {args.output_dir}")
        return

    frame, metadata = (local(args.local, args.symbol, args.start, args.end) if args.local else bundled(args.sample))
    result = run_combined(frame, metadata)
    print(metadata.get("label", ""))
    print(f"Records checked: {result.records:,}  ({metadata.get('dataset')} / {metadata.get('schema')})")
    if result.engine_note:
        print(result.engine_note)
    print(f"{ENGINE}: {len(result.alerts)} alerts   {BATCH}: {len(result.flags)} flags\n")
    print(counts(result.unified))
    print("\nChecks implemented by both systems:")
    print(result.overlap.to_string(index=False))
    if args.output_dir:
        save(result, args.output_dir)


if __name__ == "__main__":
    main()
