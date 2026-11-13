"""Bar charts comparing policies from simulate.py's CSV.

uv run python plots.py ../docs/results/baselines.csv --out ../docs/results/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = {
    "peak_temp_c": "Peak GPU temperature (°C)",
    "mean_busy_temp_c": "Mean busy-GPU temperature (°C)",
    "throttled_gpu_s": "Throttled GPU-seconds",
    "mean_wait_s": "Mean queue wait (s)",
}


def plot(results_csv: Path, out_dir: Path) -> list[Path]:
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    policies = [r["policy"] for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(policies, [float(r[key]) for r in rows], color="#4C72B0")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    for path in plot(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
