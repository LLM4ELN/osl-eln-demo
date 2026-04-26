"""Benchmark for process chains with physical quantities.

Tests LLM extraction of:
- MixingProcess with duration + rotational frequency
- HeatingProcess with duration + temperature
- Material inputs/outputs with mass
- Process chain linkage (output of mixing = input of curing)
"""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import process_models  # noqa: F401
import dummy_backend
import schema_catalog
from schema_catalog import build_inventory
from benchmark import (
    BenchmarkResult,
    ChapterExpectation,
    ChapterResult,
    EntityExpectation,
    ModelBenchmarkResult,
    ValidationError,
    _apply_model_config,
    _collect_future_result,
    create_benchmark_agent,
    export_results,
    get_entity_class_path,
    is_class_match,
    load_benchmark_config,
    normalize_string,
    print_comparison,
)

dummy_backend.register()

# Register custom process models in the schema inventory so the agent
# can discover MixingProcess, HeatingProcess, etc.
# Hide the original opensemantic mixing classes to avoid confusion.
_inv = build_inventory(
    include_properties=False,
    include_property_def=False,
    extra_modules=[process_models],
)
for _path in process_models.HIDDEN_SCHEMA_PATHS:
    _inv.items.pop(_path, None)
schema_catalog._cached_inventory = _inv

# ---------------------------------------------------------------------------
# Process-chain test cases
# ---------------------------------------------------------------------------

CHAPTERS: List[ChapterExpectation] = [
    ChapterExpectation(
        name="Chapter 1: Mixing and Curing",
        prompt=(
            "2g of Material A and 1g of Material B were mixed "
            "for 30 s @ 1000 rpm. The mixture was cured for "
            "10 h at 120\u00b0C."
        ),
        expected_new=[
            EntityExpectation(
                class_path="opensemantic.core.v1.Material",
                fields={"name": "Material A"},
            ),
            EntityExpectation(
                class_path="opensemantic.core.v1.Material",
                fields={"name": "Material B"},
            ),
            EntityExpectation(
                class_path="process_models.MixingProcess",
                fields={
                    "duration": {
                        "value": 30,
                        "unit": "Item:OSW85302b21cf045998b80f38c9fdb88f84",  # second
                    },
                    "mixing_speed": {
                        "value": 1000,
                        "unit": "Item:OSWed94e8968e8855bb9130785f981b0549",  # per_minute (rpm)
                    },
                    "input_materials": [
                        {"mass": {
                            "value": 2,
                            "unit": "Item:OSW3c9bf4c3682f5a52b6e99f7ad7949903",  # gram
                        }},
                        {"mass": {
                            "value": 1,
                            "unit": "Item:OSW3c9bf4c3682f5a52b6e99f7ad7949903",  # gram
                        }},
                    ],
                },
            ),
            EntityExpectation(
                class_path="process_models.HeatingProcess",
                fields={
                    "duration": {
                        "value": 10,
                        "unit": "Item:OSWa58b950af5d658e7b8bc2e1736817e43",  # hour
                    },
                    "temperature": {
                        "value": 120,
                        "unit": "Item:OSWad04992cecb75af4be6f2205e791b42e",  # Celsius
                    },
                },
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Enhanced field matching for nested / quantity fields
# ---------------------------------------------------------------------------


def nested_field_matches(actual: Any, expected: Any) -> bool:
    """Recursively check whether *actual* satisfies *expected*.

    Supports:
    - dict-in-dict partial matching
    - numeric comparison with tolerance
    - list matching (each expected item must match an actual item)
    - string substring matching (case-insensitive)
    """
    if expected is None:
        return True
    if actual is None:
        return False

    # Dict: partial match — every key in expected must match in actual
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if not nested_field_matches(act_val, exp_val):
                return False
        return True

    # List: each expected element must find a match in actual
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        used = set()
        for exp_item in expected:
            found = False
            for idx, act_item in enumerate(actual):
                if idx in used:
                    continue
                if nested_field_matches(act_item, exp_item):
                    used.add(idx)
                    found = True
                    break
            if not found:
                return False
        return True

    # Numeric: compare with tolerance
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) < 0.01
        except (ValueError, TypeError):
            return False

    # String: case-insensitive substring
    if isinstance(expected, str):
        return normalize_string(expected) in normalize_string(str(actual))

    return actual == expected


def find_matching_entity_nested(
    entities: Dict[str, Any],
    expectation: EntityExpectation,
    exclude_ids: set = None,
) -> tuple[str, Dict] | None:
    """Find an entity matching the expectation using nested field matching."""
    exclude_ids = exclude_ids or set()

    for entity_id, entity in entities.items():
        if entity_id in exclude_ids:
            continue

        actual_class = get_entity_class_path(entity)
        if not is_class_match(actual_class, expectation.class_path):
            continue

        entity_dict = json.loads(entity.json(exclude_none=True))
        if nested_field_matches(entity_dict, expectation.fields):
            return (entity_id, entity_dict)

    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chapter(
    chapter: ChapterExpectation,
    agent: Any,
    previous_entity_ids: set,
) -> ChapterResult:
    """Validate a chapter's results with nested quantity matching."""
    result = ChapterResult(chapter_name=chapter.name)

    current_entities = agent.get_entities()
    new_entity_ids = set(current_entities.keys()) - previous_entity_ids
    matched_new_ids: set = set()

    # Validate expected NEW entities
    for expectation in chapter.expected_new:
        new_entities = {
            eid: current_entities[eid]
            for eid in new_entity_ids
            if eid not in matched_new_ids
        }

        match = find_matching_entity_nested(
            new_entities, expectation, matched_new_ids
        )

        if match:
            entity_id, entity_dict = match
            matched_new_ids.add(entity_id)
            result.entities_created.append({
                "id": entity_id,
                "class": get_entity_class_path(current_entities[entity_id]),
                "expected_class": expectation.class_path,
                "fields": expectation.fields,
                "actual": entity_dict,
            })
        else:
            result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="MISSING_NEW_ENTITY",
                message=(
                    f"Expected new entity of type '{expectation.class_path}' "
                    f"with fields {expectation.fields} not found"
                ),
                details={"expected": expectation.fields},
            ))

    # Check for unexpected new entities
    unexpected_new = new_entity_ids - matched_new_ids
    if unexpected_new:
        for eid in unexpected_new:
            entity = current_entities[eid]
            entity_class = get_entity_class_path(entity)
            result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="UNEXPECTED_NEW_ENTITY",
                message=f"Unexpected new entity created: {entity_class}",
                details={"entity_id": eid, "class": entity_class},
            ))

    expected_new_count = len(chapter.expected_new)
    actual_new_count = len(new_entity_ids)
    if actual_new_count != expected_new_count:
        result.errors.append(ValidationError(
            chapter=chapter.name,
            error_type="WRONG_NEW_COUNT",
            message=(
                f"Expected {expected_new_count} new entities, "
                f"got {actual_new_count}"
            ),
            details={
                "expected": expected_new_count,
                "actual": actual_new_count,
                "new_ids": list(new_entity_ids),
            },
        ))

    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    verbose: bool = True,
    agent_type: str = "iterative",
) -> BenchmarkResult:
    """Run the process-chain benchmark."""
    result = BenchmarkResult()
    agent = create_benchmark_agent(agent_type)

    all_entity_ids: set = set()
    total_start = time.time()

    for chapter in CHAPTERS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {chapter.name}")
            print(f"Prompt: {chapter.prompt}")
            print(f"Expected new: {len(chapter.expected_new)}")
            print("=" * 60)

        chapter_start = time.time()

        try:
            agent.invoke(prompt=chapter.prompt)
            chapter_elapsed = time.time() - chapter_start

            if not agent.entities:
                chapter_result = ChapterResult(chapter_name=chapter.name)
                chapter_result.errors.append(ValidationError(
                    chapter=chapter.name,
                    error_type="INVOKE_FAILED",
                    message="Failed to create entity",
                ))
                chapter_result.elapsed_time = chapter_elapsed
                result.chapters.append(chapter_result)
                continue

            chapter_result = validate_chapter(
                chapter, agent, all_entity_ids
            )
            chapter_result.elapsed_time = chapter_elapsed

            if verbose:
                print(f"\nCreated: {len(chapter_result.entities_created)}")
                for e in chapter_result.entities_created:
                    print(f"  - {e['class']}: {e['fields']}")
                if chapter_result.errors:
                    print(f"\nErrors: {len(chapter_result.errors)}")
                    for err in chapter_result.errors:
                        print(f"  [{err.error_type}] {err.message}")

            agent.store_entities()
            all_entity_ids.update(agent.entities.keys())
            result.chapters.append(chapter_result)
            agent.clear()

        except Exception:
            chapter_elapsed = time.time() - chapter_start
            tb = traceback.format_exc()
            print(f"\n  EXCEPTION in {chapter.name}:\n{tb}")
            chapter_result = ChapterResult(chapter_name=chapter.name)
            chapter_result.elapsed_time = chapter_elapsed
            chapter_result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="EXCEPTION",
                message=tb.splitlines()[-1],
                details={"traceback": tb},
            ))
            result.chapters.append(chapter_result)
            break

    for ch in result.chapters:
        result.total_errors += len(ch.errors)
    result.passed = result.total_errors == 0
    result.total_time = time.time() - total_start

    if hasattr(agent, "token_usage"):
        result.input_tokens = agent.token_usage.get("input_tokens", 0)
        result.output_tokens = agent.token_usage.get("output_tokens", 0)
        result.total_tokens = agent.token_usage.get("total_tokens", 0)

    if result.total_tokens > 0 and result.total_time > 0:
        result.tokens_per_second = result.total_tokens / result.total_time

    if verbose:
        print(f"\nTotal benchmark time: {result.total_time:.2f}s")
        print(
            f"Tokens: {result.input_tokens} in / "
            f"{result.output_tokens} out / "
            f"{result.total_tokens} total"
        )

    return result


def print_summary(result: BenchmarkResult):
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 60)
    print("PROCESS CHAIN BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nChapters processed: {len(result.chapters)}")

    total_created = sum(len(c.entities_created) for c in result.chapters)
    print(f"Total entities created: {total_created}")

    print("\nPer-chapter breakdown:")
    for chapter in result.chapters:
        status = "PASS" if not chapter.errors else "FAIL"
        print(
            f"\n  [{status}] {chapter.chapter_name} "
            f"({chapter.elapsed_time:.2f}s)"
        )
        print(
            f"    New: {len(chapter.entities_created)}, "
            f"Errors: {len(chapter.errors)}"
        )
        if chapter.errors:
            for err in chapter.errors:
                print(f"      [{err.error_type}] {err.message}")

    print("\n" + "=" * 60)
    if result.passed:
        print("RESULT: PASSED")
    else:
        print(f"RESULT: FAILED ({result.total_errors} errors)")
    print("=" * 60)


def _run_benchmarks_for_model(
    model_config: Dict[str, Any],
    n_runs: int,
    verbose: bool,
    run_timeout: float = 600,
    agent_type: str = "iterative",
) -> ModelBenchmarkResult:
    """Run benchmark repetitions for a single model."""
    _apply_model_config(model_config)
    model = model_config["model_name"]

    safe_config = {
        k: v for k, v in model_config.items() if k != "litellm_params"
    }
    params = model_config.get("litellm_params", {})
    safe_config["litellm_params"] = {
        k: v for k, v in params.items() if k not in ("api_key",)
    }

    model_result = ModelBenchmarkResult(
        model=model, agent_type=agent_type, model_config=safe_config
    )

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} for {model} ---")
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            run_benchmark, verbose=verbose, agent_type=agent_type
        )
        result = _collect_future_result(future, run_idx, run_timeout)
        executor.shutdown(wait=False, cancel_futures=True)
        model_result.runs.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"--- Run {run_idx + 1} finished: {status}, "
            f"{result.total_errors} errors, "
            f"{result.total_time:.2f}s ---"
        )

    price_in = model_config.get("price_per_million_token_in")
    price_out = model_config.get("price_per_million_token_out")
    if price_in is not None and price_out is not None:
        for run in model_result.runs:
            run.cost = (
                run.input_tokens * price_in / 1_000_000
                + run.output_tokens * price_out / 1_000_000
            )

    return model_result


def run_model_comparison(config=None, verbose=False):
    """Run the process benchmark for each model defined in the config."""
    if config is None:
        config = load_benchmark_config()

    model_configs = config["models"]
    agent_types = config.get("agent_type", ["iterative"])
    n_runs = config.get("runs", 3)
    run_timeout = config.get("timeout", 600)

    combinations = [
        (mc, at) for mc in model_configs for at in agent_types
    ]

    all_results = []
    for mc, at in combinations:
        model = mc["model_name"]
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model}  AGENT: {at}")
        print(f"{'#'*60}")
        try:
            model_result = _run_benchmarks_for_model(
                mc, n_runs, verbose, run_timeout, agent_type=at,
            )
        except Exception:
            tb = traceback.format_exc()
            print(f"\n### {model}/{at} CRASHED:\n{tb}")
            model_result = ModelBenchmarkResult(
                model=model, agent_type=at
            )
        all_results.append(model_result)

    return all_results


if __name__ == "__main__":
    all_results = run_model_comparison()
    print_comparison(all_results)
    export_results(all_results)
