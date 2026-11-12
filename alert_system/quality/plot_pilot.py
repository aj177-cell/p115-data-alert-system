"""Chart the committed 95 symbol-day pilot summaries.

The per-record flag tables are not in this repo. This script plots
``docs/results/flags_by_check.csv`` and the symbol-day totals in
``docs/results/run_index.csv``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"


def plot_summaries(results=RESULTS):
    """Return a figure with flag totals and flags per symbol-day."""
    results = Path(results)
    by_check = pd.read_csv(results / "flags_by_check.csv").sort_values("n")
    index = pd.read_csv(results / "run_index.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(by_check["check"], by_check["n"])
    axes[0].set_xlabel("Flags")
    axes[0].set_title("Pilot flags by check (95 symbol-days)")
    labels = index["venue"] + " " + index["symbol"] + " " + index["day"]
    axes[1].scatter(index["records"], index["n_flags"])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Records")
    axes[1].set_ylabel("Flags")
    axes[1].set_title("Flags per symbol-day")
    for name, row in zip(labels, index.itertuples(index=False)):
        if row.n_flags >= index["n_flags"].quantile(0.9):
            axes[1].annotate(name, (row.records, row.n_flags), fontsize=6)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot the committed pilot summaries")
    parser.add_argument("--output", type=Path, required=True, help="PNG path to write")
    args = parser.parse_args()
    figure = plot_summaries()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
