import argparse
from collections import Counter
import json
from pathlib import Path

from alert_system.data import load_data
from alert_system.detectors import AlertEngine
from alert_system.quality.config import DetectorConfig, cme_futures_config, equities_config
from alert_system.quality.detectors import run_all
from alert_system.quality.inputs import pipeline_frame, pipeline_parquets, read_pipeline_parquet


PRESETS = {"equities": equities_config, "cme": cme_futures_config, "base": DetectorConfig}


def run(sample):
    data = load_data({"source": "bundled", "sample": sample})
    engine = AlertEngine()
    alerts = []
    for event in data.events:  # Source arrival order, one record at a time
        alerts += engine.process(event)
    return data, alerts


def run_batch_frame(frame, preset="equities"):
    """Run the seven batch checks and the vendor-flag check on one pipeline table."""
    if preset not in PRESETS:
        raise ValueError("preset must be equities, cme, or base")
    return run_all(frame, PRESETS[preset]())


def run_batch_sample(sample, preset="equities"):
    """Batch checks on the same bundled file the loader reads."""
    frame = pipeline_frame({"source": "bundled", "sample": sample})
    flags, summary = run_batch_frame(frame, preset)
    return frame, flags, summary


def _print_streaming(data, alerts):
    print(data.metadata["label"])
    print(f"Records checked: {len(data.events):,}")
    print(f"Symbols: {', '.join(data.metadata['symbols'])}")
    print(f"Streaming alerts: {len(alerts)}")
    for rule_id, count in sorted(Counter(alert["rule_id"] for alert in alerts).items()):
        print(f"  {rule_id}: {count}")


def _print_batch(label, frame, flags):
    print(label)
    print(f"Records checked: {len(frame):,}")
    print(f"Batch flags: {len(flags)}")
    counts = Counter(flags["check"]) if len(flags) else {}
    for check, count in sorted(counts.items()):
        print(f"  {check}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Run streaming and batch data-quality checks")
    parser.add_argument("--sample", choices=("quotes", "trades"), default="quotes")
    parser.add_argument("--mode", choices=("streaming", "batch", "both"), default="streaming")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="equities",
                        help="Batch threshold preset. equities matches the XNAS pilot (sequence gaps skipped, tick 0.01).")
    parser.add_argument("--path", type=Path,
                        help="Parquet file, or a directory of pipeline data.parquet files. Batch mode only. No download.")
    parser.add_argument("--output", type=Path,
                        help="Streaming mode writes alerts JSON. Batch or both mode writes one flags CSV.")
    args = parser.parse_args()
    if args.path is not None and args.mode != "batch":
        parser.error("--path is for --mode batch. Combined sample runs use --mode both.")

    alerts = None
    flags = None
    if args.mode in ("streaming", "both"):
        data, alerts = run(args.sample)
        _print_streaming(data, alerts)
    if args.mode == "batch" and args.path is not None:
        paths = pipeline_parquets(args.path)
        if args.output and len(paths) != 1:
            parser.error("--output writes one flags table. Pass a single Parquet file.")
        flags = None
        for path in paths:
            frame = read_pipeline_parquet(path)
            flags, _summary = run_batch_frame(frame, args.preset)
            _print_batch(str(path), frame, flags)
        if len(paths) != 1:
            flags = None
    elif args.mode in ("batch", "both"):
        frame, flags, _summary = run_batch_sample(args.sample, args.preset)
        _print_batch(f"Batch checks on the {args.sample} sample ({args.preset} preset)", frame, flags)

    if args.output and args.mode == "streaming":
        args.output.write_text(json.dumps(alerts, indent=2, allow_nan=False) + "\n")
        print(f"Saved alerts to {args.output}")
    elif args.output and flags is not None:
        flags.to_csv(args.output, index=False)
        print(f"Saved flags to {args.output}")


if __name__ == "__main__":
    main()
