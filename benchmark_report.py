"""Render benchmark YAML results as a Markdown report with tables and Mermaid charts."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_result() -> Path:
    """Return the most recently modified .yaml file in results/."""
    yamls = sorted(RESULTS_DIR.glob("benchmark_*.yaml"), key=lambda p: p.stat().st_mtime)
    if not yamls:
        print("No benchmark result files found in", RESULTS_DIR, file=sys.stderr)
        sys.exit(1)
    return yamls[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _collect_chapter_names(data: list) -> list[str]:
    """Collect ordered chapter names across all models and runs."""
    seen = {}
    for model_result in data:
        for run in model_result["runs"]:
            for ch in run["chapters"]:
                name = ch["chapter_name"]
                if name not in seen:
                    seen[name] = len(seen)
    return list(seen.keys())


def _short_chapter(name: str) -> str:
    """Shorten 'Chapter 1: Experiment #1' to 'Ch1'."""
    if name.startswith("Chapter "):
        num = name.split(":")[0].replace("Chapter ", "")
        return f"Ch{num}"
    return name[:8]


def _build_chart_labels(data: list) -> list[str]:
    """Return sanitized model names for Mermaid x-axis labels."""
    # Use non-breaking hyphen so Mermaid doesn't parse hyphens as minus
    return [m["model"].replace("-", "‑") for m in data]


MERMAID_CHART = "xychart-beta horizontal"


def _mermaid_x_axis(labels: list[str]) -> str:
    """Build a Mermaid x-axis line with quoted labels."""
    quoted = ['"' + l + '"' for l in labels]
    return "  x-axis [" + ", ".join(quoted) + "]"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def render_header(data: list, source_path: Path) -> str:
    n_models = len(data)
    run_counts = [len(m["runs"]) for m in data]
    max_runs = max(run_counts) if run_counts else 0
    return (
        f"# Benchmark Report\n\n"
        f"- **Source:** `{source_path.name}`\n"
        f"- **Models:** {n_models}\n"
        f"- **Runs per model:** {max_runs}\n"
    )


def render_model_config_table(data: list) -> str:
    """Render a table of model configurations (provider, temperature, etc.)."""
    # Collect all litellm_params keys across models (excluding api_base which is just a URL)
    interesting_keys = set()
    for m in data:
        cfg = m.get("model_config", {})
        params = cfg.get("litellm_params", {})
        for k in params:
            if k not in ("model", "api_base", "api_key"):
                interesting_keys.add(k)

    if not interesting_keys:
        return ""

    sorted_keys = sorted(interesting_keys)
    header_parts = ["Model", "Provider"] + sorted_keys
    sep_parts = ["---"] * len(header_parts)
    lines = [
        "## Model Configuration\n",
        "| " + " | ".join(header_parts) + " |",
        "| " + " | ".join(sep_parts) + " |",
    ]
    for m in data:
        model = m["model"]
        cfg = m.get("model_config", {})
        provider = cfg.get("provider", "—")
        params = cfg.get("litellm_params", {})
        cells = [model, provider]
        for k in sorted_keys:
            val = params.get(k)
            cells.append(str(val) if val is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _has_cost_data(data: list) -> bool:
    """Check if any model has cost > 0."""
    for m in data:
        for run in m.get("runs", []):
            if run.get("cost", 0) > 0:
                return True
    return False


def render_summary_table(data: list) -> str:
    show_cost = _has_cost_data(data)
    header = (
        "| Model | Passed | Avg Errors | Avg Time (s) "
        "| Avg In Tokens | Avg Out Tokens | Avg Tok/s |"
    )
    sep = "|---|---|---|---|---|---|---|"
    if show_cost:
        header += " Avg Cost |"
        sep += "---|"
    lines = [
        "## Summary\n",
        header,
        sep,
    ]
    for m in data:
        model = m["model"]
        runs = m["runs"]
        if not runs:
            empty = f"| {model} | 0/0 | — | — | — | — | — |"
            if show_cost:
                empty += " — |"
            lines.append(empty)
            continue
        n = len(runs)
        pass_count = sum(1 for r in runs if r["passed"])
        avg_errors = _avg([r["total_errors"] for r in runs])
        avg_time = _avg([r["total_time"] for r in runs])
        avg_in = _avg([r.get("input_tokens", 0) for r in runs])
        avg_out = _avg([r.get("output_tokens", 0) for r in runs])
        avg_tps = _avg([
            r.get("tokens_per_second", 0) for r in runs
        ])
        row = (
            f"| {model} | {pass_count}/{n} "
            f"| {avg_errors:.1f} | {avg_time:.1f} "
            f"| {avg_in:.0f} | {avg_out:.0f} "
            f"| {avg_tps:.1f} |"
        )
        if show_cost:
            avg_cost = _avg([r.get("cost", 0) for r in runs])
            if avg_cost > 0:
                row += f" ${avg_cost:.4f} |"
            else:
                row += " — |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def render_chapter_table(data: list, chapter_names: list[str]) -> str:
    short_names = [_short_chapter(n) for n in chapter_names]
    header_parts = ["Model"] + [f"{s} Errors" for s in short_names] + [f"{s} Time" for s in short_names]
    sep_parts = ["---"] * len(header_parts)
    lines = [
        "## Per-Chapter Breakdown\n",
        "| " + " | ".join(header_parts) + " |",
        "| " + " | ".join(sep_parts) + " |",
    ]
    for m in data:
        model = m["model"]
        runs = m["runs"]
        if not runs:
            lines.append("| " + model + " | " + " | ".join(["—"] * (len(chapter_names) * 2)) + " |")
            continue
        err_cells = []
        time_cells = []
        for ch_name in chapter_names:
            ch_errors = []
            ch_times = []
            for run in runs:
                for ch in run["chapters"]:
                    if ch["chapter_name"] == ch_name:
                        ch_errors.append(len(ch["errors"]))
                        ch_times.append(ch["elapsed_time"])
                        break
            if ch_errors:
                err_cells.append(f"{_avg(ch_errors):.1f}")
                time_cells.append(f"{_avg(ch_times):.1f}")
            else:
                err_cells.append("—")
                time_cells.append("—")
        row = [model] + err_cells + time_cells
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_error_type_table(data: list) -> str:
    type_counts: dict[str, int] = defaultdict(int)
    type_models: dict[str, set] = defaultdict(set)
    for m in data:
        for run in m["runs"]:
            for ch in run["chapters"]:
                for err in ch["errors"]:
                    et = err["error_type"]
                    type_counts[et] += 1
                    type_models[et].add(m["model"])
    if not type_counts:
        return "## Error Type Distribution\n\nNo errors recorded.\n"
    lines = [
        "## Error Type Distribution\n",
        "| Error Type | Count | Models Affected |",
        "|---|---|---|",
    ]
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        models = ", ".join(sorted(type_models[et]))
        lines.append(f"| `{et}` | {count} | {models} |")
    return "\n".join(lines) + "\n"


def render_errors_chart(data: list, labels: list[str]) -> str:
    values = []
    for m in data:
        runs = m["runs"]
        values.append(_avg([r["total_errors"] for r in runs]) if runs else 0)
    lines = [
        "## Errors per Model\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Errors per Model"',
        _mermaid_x_axis(labels),
        '  y-axis "Errors"',
        f'  bar [{", ".join(f"{v:.1f}" for v in values)}]',
        "```\n",
    ]
    return "\n".join(lines)


def render_time_chart(data: list, labels: list[str]) -> str:
    values = []
    for m in data:
        runs = m["runs"]
        values.append(_avg([r["total_time"] for r in runs]) if runs else 0)
    lines = [
        "## Time per Model\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Total Time per Model (s)"',
        _mermaid_x_axis(labels),
        '  y-axis "Seconds"',
        f'  bar [{", ".join(f"{v:.1f}" for v in values)}]',
        "```\n",
    ]
    return "\n".join(lines)


def render_tokens_chart(data: list, labels: list[str]) -> str:
    values = []
    for m in data:
        runs = m["runs"]
        values.append(
            _avg([r.get("total_tokens", 0) for r in runs])
            if runs else 0
        )
    lines = [
        "## Tokens per Model\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Total Tokens per Model"',
        _mermaid_x_axis(labels),
        '  y-axis "Tokens"',
        f'  bar [{", ".join(f"{v:.0f}" for v in values)}]',
        "```\n",
    ]
    return "\n".join(lines)


def render_token_rate_chart(data: list, labels: list[str]) -> str:
    values = []
    for m in data:
        runs = m["runs"]
        values.append(
            _avg([r.get("tokens_per_second", 0) for r in runs])
            if runs else 0
        )
    lines = [
        "## Token Rate per Model\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Tokens per Second"',
        _mermaid_x_axis(labels),
        '  y-axis "Tok/s"',
        f'  bar [{", ".join(f"{v:.1f}" for v in values)}]',
        "```\n",
    ]
    return "\n".join(lines)


def render_cost_chart(data: list, labels: list[str]) -> str:
    if not _has_cost_data(data):
        return ""
    values = []
    for m in data:
        runs = m["runs"]
        values.append(
            _avg([r.get("cost", 0) for r in runs])
            if runs else 0
        )
    lines = [
        "## Cost per Model\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Avg Cost per Model ($)"',
        _mermaid_x_axis(labels),
        '  y-axis "USD"',
        f'  bar [{", ".join(f"{v:.4f}" for v in values)}]',
        "```\n",
    ]
    return "\n".join(lines)


def render_chapter_errors_chart(
    data: list, chapter_names: list[str], labels: list[str]
) -> str:
    short_names = [_short_chapter(n) for n in chapter_names]
    lines = [
        "## Errors per Chapter\n",
        "```mermaid",
        MERMAID_CHART,
        '  title "Errors per Chapter"',
        _mermaid_x_axis(labels),
        '  y-axis "Errors"',
    ]
    for ch_idx, ch_name in enumerate(chapter_names):
        values = []
        for m in data:
            ch_errors = []
            for run in m["runs"]:
                for ch in run["chapters"]:
                    if ch["chapter_name"] == ch_name:
                        ch_errors.append(len(ch["errors"]))
                        break
            values.append(_avg(ch_errors) if ch_errors else 0)
        lines.append(f'  bar [{", ".join(f"{v:.1f}" for v in values)}]')
    lines.append("```\n")
    # Add legend since xychart-beta doesn't label multiple bars
    bar_legend = " / ".join(f"Bar {i+1} = {s}" for i, s in enumerate(short_names))
    lines.append(f"*{bar_legend}*\n")
    return "\n".join(lines)


def render_detailed_errors(data: list) -> str:
    sections = ["## Detailed Errors\n"]
    for m in data:
        model = m["model"]
        total_errors = 0
        for run in m["runs"]:
            total_errors += run["total_errors"]
        if total_errors == 0:
            sections.append(f"### {model}\n\nNo errors.\n")
            continue
        sections.append(f"### {model}\n")
        for run_idx, run in enumerate(m["runs"]):
            if len(m["runs"]) > 1:
                sections.append(f"**Run {run_idx + 1}** — {'PASS' if run['passed'] else 'FAIL'}, "
                                f"{run['total_errors']} errors, {run['total_time']:.1f}s\n")
            if run.get("error_message"):
                sections.append(f"> Run-level error: {run['error_message']}\n")
            for ch in run["chapters"]:
                if not ch["errors"]:
                    continue
                sections.append(f"**{ch['chapter_name']}** ({ch['elapsed_time']:.1f}s)\n")
                for err in ch["errors"]:
                    sections.append(f"- `{err['error_type']}`: {err['message']}")
                sections.append("")
    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(data: list, source_path: Path) -> str:
    chapter_names = _collect_chapter_names(data)
    labels = _build_chart_labels(data)
    parts = [
        render_header(data, source_path),
        render_model_config_table(data),
        render_summary_table(data),
        render_errors_chart(data, labels),
        render_time_chart(data, labels),
        render_tokens_chart(data, labels),
        render_token_rate_chart(data, labels),
        render_cost_chart(data, labels),
        render_chapter_errors_chart(data, chapter_names, labels),
        render_chapter_table(data, chapter_names),
        render_error_type_table(data),
        render_detailed_errors(data),
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown report from benchmark YAML results.")
    parser.add_argument("result_file", nargs="?", type=Path, default=None,
                        help="Path to benchmark YAML file (default: latest in results/)")
    args = parser.parse_args()

    source = args.result_file or find_latest_result()
    data = load_results(source)
    report = generate_report(data, source)

    out_path = source.with_suffix(".md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(out_path)


if __name__ == "__main__":
    main()
