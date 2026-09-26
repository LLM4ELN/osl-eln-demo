"""Run one enforcement condition of the benchmark.

The enforcement knobs are environment variables read by the agents, so they
have to be set before ``benchmark`` is imported: that module snapshots the
environment to resolve ``os.environ/VAR`` references in the model config.

Usage:
    uv run python run_arm.py A2
    uv run python run_arm.py A1 --runs 3
    uv run python run_arm.py A2 --models claude-sonnet-4-5,gpt-5-mini
"""

import argparse
import os

from dotenv import load_dotenv

ARMS = {
    "A1": {"CATALOGUE_MODE": "full", "DECODE_CONSTRAINT": "none"},
    "A2": {"CATALOGUE_MODE": "full", "DECODE_CONSTRAINT": "enum"},
    "A1-lean": {"CATALOGUE_MODE": "paths", "DECODE_CONSTRAINT": "none"},
    "A2-lean": {"CATALOGUE_MODE": "paths", "DECODE_CONSTRAINT": "enum"},
}

# The nine models of the 2026-09-26 runs, so a re-baseline stays comparable.
BASELINE_MODELS = [
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-5",
    "deepseek-v3.2",
    "cohere-command-a",
    "gpt-5.2-chat",
    "gpt-5-nano",
    "gpt-5-mini",
    "mistral-large-3",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument(
        "--models",
        default=",".join(BASELINE_MODELS),
        help="Comma-separated model names, or 'config' to use the YAML list.",
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--agent-type", default="multi_step")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra enforcement setting, repeatable. Applied after the arm.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    os.environ.update(ARMS[args.arm])
    for setting in args.env:
        if "=" not in setting:
            raise SystemExit(f"--env needs KEY=VALUE, got {setting!r}")
        key, value = setting.split("=", 1)
        os.environ[key.strip()] = value.strip()

    import benchmark

    config = benchmark.load_benchmark_config()
    config["agent_type"] = [args.agent_type]
    if args.runs is not None:
        config["runs"] = args.runs
    if args.models != "config":
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        by_name = {m["model_name"]: m for m in config["models"]}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise SystemExit(f"not in model_list: {', '.join(missing)}")
        config["models"] = [by_name[n] for n in wanted]

    print(
        f"arm={args.arm} {ARMS[args.arm]} agent_type={args.agent_type} "
        f"models={len(config['models'])} runs={config['runs']}"
    )
    print(f"enforcement={benchmark.get_enforcement()}")

    results = benchmark.run_model_comparison(config, verbose=args.verbose)
    benchmark.print_comparison(results)
    benchmark.export_results(results)


if __name__ == "__main__":
    main()
