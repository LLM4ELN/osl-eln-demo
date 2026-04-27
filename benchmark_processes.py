"""Benchmark for process chains with physical quantities.

Tests LLM extraction of:
- MixingProcess with duration + rotational frequency
- HeatingProcess with duration + temperature
- Material inputs/outputs with mass
- Process chain linkage (output of mixing = input of curing)

Reuses the benchmark infrastructure from benchmark.py by patching
CHAPTERS and validate_chapter with process-chain specific versions.
"""

import json
from typing import Any, Dict, List

import process_models  # noqa: F401
import dummy_backend
import benchmark
import schema_catalog
from schema_catalog import build_inventory
from benchmark import (
    ChapterExpectation,
    ChapterResult,
    EntityExpectation,
    ValidationError,
    export_results,
    get_entity_class_path,
    is_class_match,
    normalize_string,
    print_comparison,
    run_model_comparison,
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
# Custom validate_chapter with nested quantity matching
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
# Monkey-patch benchmark.py to use our CHAPTERS and validator
# ---------------------------------------------------------------------------

benchmark.CHAPTERS = CHAPTERS
benchmark.validate_chapter = validate_chapter


if __name__ == "__main__":
    all_results = run_model_comparison()
    print_comparison(all_results)
    export_results(all_results)
