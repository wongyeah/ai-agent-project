"""
Plot a search run's "best metric found so far" against search iteration,
optionally overlaid with a one-shot baseline reference line (see
scripts/run_one_shot_baseline.py).

Usage:
    python scripts/plot_search_trajectory.py \\
        --journal runs/eval_california_housing_journal.json \\
        --baseline runs/eval_california_housing_baseline.json \\
        --out runs/search_vs_baseline.png

--baseline is optional; omit it to plot the search trajectory alone.

NOTE on metric direction: this assumes LOWER is better (e.g. MSE), same
assumption Journal.get_best_node() and Agent._node_value() make elsewhere
in this project (see the TODO comments there). Pass --higher-is-better
for a metric like Accuracy where the opposite holds — running this
unmodified against an accuracy-based run (e.g. eval_titanic.yaml) without
that flag will plot a nonsensical "best so far" curve that goes the wrong
direction.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.agent.journal import Journal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--out", type=str, default="runs/search_vs_baseline.png")
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Set for metrics like Accuracy; default assumes lower is better (e.g. MSE).",
    )
    args = parser.parse_args()

    with open(args.journal) as f:
        journal = Journal.from_dict(json.load(f))

    better = (lambda a, b: a > b) if args.higher_is_better else (lambda a, b: a < b)
    running_best = None

    steps, best_so_far, raw_metric = [], [], []
    for n in journal.nodes:
        steps.append(n.step)
        if n.metric is not None:
            if running_best is None or better(n.metric, running_best):
                running_best = n.metric
        best_so_far.append(running_best)
        raw_metric.append(n.metric)

    baseline_metric = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        baseline_metric = baseline_data.get("metric", baseline_data.get("mse"))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    ax.plot(
        steps, raw_metric, marker="o", markersize=5, linewidth=1, alpha=0.45,
        color="#7f8fa6", label="Each step's own metric (raw attempt)", zorder=2,
    )
    ax.step(
        steps, best_so_far, where="post", linewidth=2.4, color="#1f6feb",
        label="Best-so-far metric (search trajectory)", zorder=3,
    )
    ax.scatter(steps, best_so_far, color="#1f6feb", s=22, zorder=4)

    if baseline_metric is not None:
        ax.axhline(
            baseline_metric, color="#d1242f", linestyle="--", linewidth=1.8,
            label=f"One-shot baseline: {baseline_metric:,.3e}", zorder=1,
        )

    direction = "higher is better" if args.higher_is_better else "lower is better"
    ax.set_xlabel("Search iteration (step)")
    ax.set_ylabel(f"Validation metric ({direction})")
    ax.set_title("Search Trajectory vs. One-Shot Baseline")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.2e}"))
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
