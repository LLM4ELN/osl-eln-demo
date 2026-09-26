"""Compare benchmark result files cell by cell.

``benchmark_report.py`` and ``merge_benchmarks.py`` both key on the model name
alone, so two arms of the same model collapse into one row. This keys on
(model, agent_type, enforcement) instead, which is the condition a run was
actually executed under.

Usage:
    uv run python compare_arms.py results/a.yaml results/b.yaml
    uv run python compare_arms.py --label A2 results/a.yaml --label A1 results/b.yaml
"""

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

RESULTS_DIR = Path(__file__).parent / "results"


def load(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def condition(entry: Dict[str, Any]) -> str:
    """One-line description of the cell an entry belongs to."""
    enforcement = entry.get("enforcement") or {}
    if not enforcement:
        return entry.get("agent_type", "?")
    return (
        f"{entry.get('agent_type', '?')}"
        f"/{enforcement.get('catalogue_mode', '?')}"
        f"/{enforcement.get('decode_constraint', '?')}"
        f"/n={enforcement.get('catalogue_size', '?')}"
    )


def summarise(entry: Dict[str, Any]) -> Dict[str, Any]:
    runs = entry.get("runs") or []
    passed = sum(1 for r in runs if r.get("passed"))
    errors = [r.get("total_errors") or 0 for r in runs]
    tokens = [r.get("total_tokens") or 0 for r in runs]
    cost = sum(r.get("cost") or 0 for r in runs)
    return {
        "runs": len(runs),
        "passed": passed,
        "errors": statistics.mean(errors) if errors else 0.0,
        "tokens": statistics.median(tokens) if tokens else 0,
        "cost": cost,
    }


def error_types(entry: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in entry.get("runs") or []:
        for chapter in run.get("chapters") or []:
            for error in chapter.get("errors") or []:
                key = error.get("error_type", "?")
                counts[key] = counts.get(key, 0) + 1
    return counts


def index(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {e["model"]: e for e in entries}


def compare(
    left: Tuple[str, List[Dict[str, Any]]],
    right: Tuple[str, List[Dict[str, Any]]],
) -> None:
    left_label, left_entries = left
    right_label, right_entries = right
    left_by_model = index(left_entries)
    right_by_model = index(right_entries)

    left_condition = condition(left_entries[0]) if left_entries else "?"
    right_condition = condition(right_entries[0]) if right_entries else "?"
    print(f"{left_label}: {left_condition}")
    print(f"{right_label}: {right_condition}")
    print()

    shared = [m for m in left_by_model if m in right_by_model]
    only_left = sorted(set(left_by_model) - set(right_by_model))
    only_right = sorted(set(right_by_model) - set(left_by_model))
    if only_left:
        print(f"only in {left_label}: {', '.join(only_left)}")
    if only_right:
        print(f"only in {right_label}: {', '.join(only_right)}")

    header = (
        f"{'model':<32} {left_label:>12} {right_label:>12} "
        f"{'tokens ' + left_label:>14} {'tokens ' + right_label:>14}"
    )
    print(header)
    print("-" * len(header))

    totals = {left_label: 0, right_label: 0}
    favours_left: List[str] = []
    favours_right: List[str] = []
    ties: List[str] = []
    for model in shared:
        a = summarise(left_by_model[model])
        b = summarise(right_by_model[model])
        totals[left_label] += a["passed"]
        totals[right_label] += b["passed"]
        if a["passed"] > b["passed"]:
            favours_left.append(model)
        elif b["passed"] > a["passed"]:
            favours_right.append(model)
        else:
            ties.append(model)
        print(
            f"{model:<32} {a['passed']:>4}/{a['runs']:<7} "
            f"{b['passed']:>4}/{b['runs']:<7} "
            f"{a['tokens']:>14,} {b['tokens']:>14,}"
        )

    runs_left = sum(summarise(left_by_model[m])["runs"] for m in shared)
    runs_right = sum(summarise(right_by_model[m])["runs"] for m in shared)
    print("-" * len(header))
    print(
        f"{'total':<32} {totals[left_label]:>4}/{runs_left:<7} "
        f"{totals[right_label]:>4}/{runs_right:<7}"
    )
    print(
        f"\nper model: {len(favours_left)} favour {left_label}, "
        f"{len(favours_right)} favour {right_label}, {len(ties)} tied"
    )
    if favours_left:
        print(f"  {left_label}: {', '.join(favours_left)}")
    if favours_right:
        print(f"  {right_label}: {', '.join(favours_right)}")

    print("\nerror types")
    all_types = set()
    left_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for model in shared:
        for key, count in error_types(left_by_model[model]).items():
            left_counts[key] = left_counts.get(key, 0) + count
        for key, count in error_types(right_by_model[model]).items():
            right_counts[key] = right_counts.get(key, 0) + count
    all_types = set(left_counts) | set(right_counts)
    for key in sorted(all_types):
        print(
            f"  {key:<28} {left_counts.get(key, 0):>6} "
            f"{right_counts.get(key, 0):>6}"
        )

    cost_left = sum(summarise(left_by_model[m])["cost"] for m in shared)
    cost_right = sum(summarise(right_by_model[m])["cost"] for m in shared)
    print(f"\ncost  {left_label}: ${cost_left:.2f}   {right_label}: ${cost_right:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs=2, type=Path)
    parser.add_argument(
        "--labels",
        default="left,right",
        help="Comma-separated labels for the two files.",
    )
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    if len(labels) != 2:
        raise SystemExit("--labels needs exactly two comma-separated names")

    compare(
        (labels[0], load(args.files[0])),
        (labels[1], load(args.files[1])),
    )


if __name__ == "__main__":
    main()
