"""Charts comparing policies: bars from simulate.py's CSV, load curves from sweep.py's summary.

uv run python plots.py ../docs/results/baselines.csv --out ../docs/results/
uv run python plots.py ../docs/results/sweep_summary.csv --sweep --out ../docs/results/
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


# Load-sweep panels: steady-state metric (mean over seeds, bars = std) against target load.
SWEEP_METRICS = {
    "achieved_utilisation": "Achieved utilisation",
    "mean_wait_s": "Mean queue wait (s)",
    "peak_temp_c": "Peak GPU temperature (°C)",
    "min_headroom_c": "Minimum thermal headroom (°C)",
}
# Categorical slots in fixed order (validated for CVD separation); markers are the
# secondary encoding, so identity never rests on colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED = "#1f1f1e", "#6b6a65"


def plot_sweep(summary_csv: Path, out_dir: Path) -> Path:
    """Small multiples, one per steady-state metric, one line per policy.

    Every number is simulated, and the figure says so.
    """
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    policies = list(dict.fromkeys(r["policy"] for r in rows))  # first-seen order, stable
    if len(policies) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} policies per sweep plot")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, (key, label) in zip(axes.flat, SWEEP_METRICS.items(), strict=True):
        for i, policy in enumerate(policies):
            pts = sorted(
                (float(r["target_load"]), float(r[f"{key}_mean"]), float(r[f"{key}_std"]))
                for r in rows
                if r["policy"] == policy
            )
            x, y, err = zip(*pts, strict=True)
            ax.errorbar(
                x,
                y,
                yerr=err,
                label=policy,
                color=SERIES_COLORS[i],
                marker=SERIES_MARKERS[i],
                markersize=6,
                linewidth=2,
                capsize=3,
                elinewidth=1,
            )
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Target offered load", color=INK, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(policies), frameon=False, fontsize=9)
    seeds = sorted({int(r["seeds"]) for r in rows})
    fig.text(
        0.01,
        0.01,
        f"Simulated results. Mean over {'-'.join(map(str, seeds))} seeds, bars are one "
        "standard deviation. Steady-state window only.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path = out_dir / "sweep.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sweep", action="store_true", help="results is sweep.py's summary CSV")
    args = p.parse_args()
    if args.sweep:
        print(plot_sweep(args.results, args.out))
        return
    for path in plot(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
