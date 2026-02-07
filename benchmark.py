"""Benchmark for OoldAgent using the demo playbook test cases."""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from oold_agent import OoldAgent
from schema_catalog import get_cached_inventory


@dataclass
class EntityExpectation:
    """Expected properties for an entity."""
    class_path: str  # e.g., "opensemantic.lab.v1.LaboratoryProcess"
    fields: Dict[str, Any] = field(default_factory=dict)
    reused: bool = False  # If True, expect this entity to be reused
    expected_fields_after: Dict[str, Any] = field(default_factory=dict)
    """Fields expected AFTER modification (for expected_modified)."""


@dataclass
class ChapterExpectation:
    """Expected results for a chapter."""
    name: str
    prompt: str
    expected_new: List[EntityExpectation] = field(default_factory=list)
    expected_reused: List[EntityExpectation] = field(default_factory=list)
    expected_modified: List[EntityExpectation] = field(default_factory=list)


@dataclass
class ValidationError:
    """A single validation error."""
    chapter: str
    error_type: str
    message: str
    details: Optional[Dict] = None


@dataclass
class ChapterResult:
    """Results from processing a chapter."""
    chapter_name: str
    entities_created: List[Dict] = field(default_factory=list)
    entities_reused: List[str] = field(default_factory=list)
    entities_modified: List[Dict] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    elapsed_time: float = 0.0


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    chapters: List[ChapterResult] = field(default_factory=list)
    total_errors: int = 0
    passed: bool = True


# Test cases from the playbook
CHAPTERS: List[ChapterExpectation] = [
    ChapterExpectation(
        name="Chapter 1: Experiment #1",
        prompt=(
            "A tensile test experiment #1 conducted by Dr. Jane Doe, "
            "employed at Example Lab Corp."
        ),
        expected_new=[
            EntityExpectation(
                class_path="opensemantic.lab.v1.LaboratoryProcess",
                fields={
                    # "name": "tensile test experiment #1",
                }
            ),
            EntityExpectation(
                class_path="opensemantic.core.v1.Person",
                fields={
                    "first_name": "Jane",
                    "surname": "Doe",
                }
            ),
            EntityExpectation(
                class_path="opensemantic.base.v1.Organization",
                fields={
                    # "name": "Example Lab Corp",
                }
            ),
        ],
        expected_reused=[],
    ),
    ChapterExpectation(
        name="Chapter 2: Experiment #2",
        prompt=(
            "A tensile test experiment #2 conducted by Dr. John Doe, "
            "employed at Example Lab Corp."
        ),
        expected_new=[
            EntityExpectation(
                class_path="opensemantic.lab.v1.LaboratoryProcess",
                fields={
                    # "name": "tensile test experiment #2",
                }
            ),
            EntityExpectation(
                class_path="opensemantic.core.v1.Person",
                fields={
                    "first_name": "John",
                    "surname": "Doe",
                }
            ),
        ],
    ),
    ChapterExpectation(
        name="Chapter 3: Experiment #3",
        prompt=(
            "A tensile test experiment #3 conducted by Dr. John Doe "
            "(john.doe@example-lab.com)"
        ),
        expected_new=[
            EntityExpectation(
                class_path="opensemantic.lab.v1.LaboratoryProcess",
                fields={
                    # "name": "tensile test experiment #3",
                }
            ),
        ],
        expected_modified=[
            EntityExpectation(
                class_path="opensemantic.core.v1.Person",
                fields={
                    "first_name": "John",
                    "surname": "Doe",
                },
                expected_fields_after={
                    "first_name": "John",
                    "surname": "Doe",
                    "email": ["john.doe@example-lab.com"],
                },
            ),
        ],
    ),
]


def get_entity_class_path(entity) -> str:
    """Get the full class path for an entity."""
    module = entity.__class__.__module__.replace("._model", "")
    return f"{module}.{entity.__class__.__name__}"


def is_class_match(actual_path: str, expected_path: str) -> bool:
    """Check if actual type matches expected (including subclasses)."""
    if actual_path == expected_path:
        return True

    inventory = get_cached_inventory()
    return inventory.is_subclass_of(actual_path, expected_path)


def normalize_string(s: str) -> str:
    """Normalize a string for comparison (lowercase, strip, etc.)."""
    if s is None:
        return ""
    return str(s).lower().strip()


def field_matches(actual_value: Any, expected_value: Any) -> bool:
    """Check if an actual field value matches the expected value."""
    if expected_value is None:
        return True  # None means "don't care"

    if actual_value is None:
        return False

    # Normalize strings for comparison
    if isinstance(expected_value, str):
        norm_expected = normalize_string(expected_value)
        norm_actual = normalize_string(actual_value)
        return norm_expected in norm_actual

    # Direct comparison for other types
    return actual_value == expected_value


def find_matching_entity(
    entities: Dict[str, Any],
    expectation: EntityExpectation,
    exclude_ids: set = None
) -> Optional[tuple[str, Dict]]:
    """Find an entity matching the expectation.

    Returns (entity_id, entity_dict) or None if not found.
    """
    exclude_ids = exclude_ids or set()

    for entity_id, entity in entities.items():
        if entity_id in exclude_ids:
            continue

        # Check class
        actual_class = get_entity_class_path(entity)
        if not is_class_match(actual_class, expectation.class_path):
            continue

        # Check fields
        entity_dict = json.loads(entity.json(exclude_none=True))
        all_fields_match = True

        for field_name, expected_value in expectation.fields.items():
            actual_value = entity_dict.get(field_name)
            if not field_matches(actual_value, expected_value):
                all_fields_match = False
                break

        if all_fields_match:
            return (entity_id, entity_dict)

    return None


def validate_chapter(
    chapter: ChapterExpectation,
    agent: OoldAgent,
    previous_entity_ids: set
) -> ChapterResult:
    """Validate a chapter's results."""
    result = ChapterResult(chapter_name=chapter.name)

    # Get current entities
    current_entities = agent.get_entities()
    new_entity_ids = set(current_entities.keys()) - previous_entity_ids

    # Track matched entities to avoid double-matching
    matched_new_ids: set = set()
    matched_reused_ids: set = set()

    # Validate expected NEW entities
    for expectation in chapter.expected_new:
        # Only look in new entities
        new_entities = {
            eid: current_entities[eid]
            for eid in new_entity_ids
            if eid not in matched_new_ids
        }

        match = find_matching_entity(
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
                details={"expected": expectation.fields}
            ))

    # Validate expected REUSED entities
    for expectation in chapter.expected_reused:
        # Look in previous entities (those that were stored in vector store)
        # They should have been returned as reused
        reused_entities = {
            eid: current_entities[eid]
            for eid in previous_entity_ids
            if eid in current_entities and eid not in matched_reused_ids
        }

        match = find_matching_entity(
            reused_entities, expectation, matched_reused_ids
        )

        if match:
            entity_id, entity_dict = match
            matched_reused_ids.add(entity_id)
            result.entities_reused.append(entity_id)
        else:
            # Check if it was created as new instead of reused
            all_current = {
                eid: current_entities[eid]
                for eid in new_entity_ids
            }
            new_match = find_matching_entity(all_current, expectation)

            if new_match:
                result.errors.append(ValidationError(
                    chapter=chapter.name,
                    error_type="NOT_REUSED",
                    message=(
                        f"Entity of type '{expectation.class_path}' "
                        f"was created as NEW instead of being REUSED"
                    ),
                    details={
                        "expected_fields": expectation.fields,
                        "new_entity_id": new_match[0]
                    }
                ))
            else:
                result.errors.append(ValidationError(
                    chapter=chapter.name,
                    error_type="MISSING_REUSED_ENTITY",
                    message=(
                        f"Expected reused entity of type "
                        f"'{expectation.class_path}' not found"
                    ),
                    details={"expected": expectation.fields}
                ))

    # Validate expected MODIFIED entities
    matched_modified_ids: set = set()
    for expectation in chapter.expected_modified:
        # Look in previous entities that are still present
        reusable_entities = {
            eid: current_entities[eid]
            for eid in previous_entity_ids
            if eid in current_entities and eid not in matched_modified_ids
        }

        # Match by class and expected_fields_after (the final state)
        modified_expectation = EntityExpectation(
            class_path=expectation.class_path,
            fields=expectation.expected_fields_after or expectation.fields
        )
        match = find_matching_entity(reusable_entities, modified_expectation)

        if match:
            entity_id, entity_dict = match
            # Verify NOT in new_entity_ids (was reused, not created)
            if entity_id in new_entity_ids:
                result.errors.append(ValidationError(
                    chapter=chapter.name,
                    error_type="MODIFIED_CREATED_AS_NEW",
                    message=(
                        f"Entity '{expectation.class_path}' was created new "
                        f"instead of being modified from existing"
                    ),
                    details={"entity_id": entity_id}
                ))
            else:
                matched_modified_ids.add(entity_id)
                modified_entity_class = get_entity_class_path(
                    current_entities[entity_id]
                )
                result.entities_modified.append({
                    "id": entity_id,
                    "class": modified_entity_class,
                    "expected_fields_after": (
                        expectation.expected_fields_after
                    ),
                    "actual": entity_dict,
                })
        else:
            result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="MISSING_MODIFIED_ENTITY",
                message=(
                    f"Expected modified entity of type "
                    f"'{expectation.class_path}' "
                    f"with fields {expectation.expected_fields_after} "
                    f"not found"
                ),
                details={"expected": expectation.expected_fields_after}
            ))

    # Check for unexpected new entities (exclude modified from new)
    unexpected_new = new_entity_ids - matched_new_ids - matched_modified_ids
    if unexpected_new:
        for eid in unexpected_new:
            entity = current_entities[eid]
            entity_class = get_entity_class_path(entity)
            result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="UNEXPECTED_NEW_ENTITY",
                message=(
                    f"Unexpected new entity created: "
                    f"{entity_class}"
                ),
                details={
                    "entity_id": eid,
                    "class": entity_class
                }
            ))

    # Check count of new entities
    # (modified entities should NOT be in new_entity_ids)
    expected_new_count = len(chapter.expected_new)
    actual_new_count = len(new_entity_ids - matched_modified_ids)
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
                "new_ids": list(new_entity_ids)
            }
        ))

    return result


def run_benchmark(verbose: bool = True) -> BenchmarkResult:
    """Run the benchmark using the playbook test cases."""
    result = BenchmarkResult()

    # Create agent with fresh vector store
    agent = OoldAgent()

    all_entity_ids: set = set()
    total_start = time.time()

    for chapter in CHAPTERS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {chapter.name}")
            print(f"Prompt: {chapter.prompt}")
            print(f"Expected new: {len(chapter.expected_new)}")
            print(f"Expected reused: {len(chapter.expected_reused)}")
            print(f"Expected modified: {len(chapter.expected_modified)}")
            print("="*60)

        chapter_start = time.time()

        # Invoke the agent
        entity_id = agent.invoke(prompt=chapter.prompt)

        chapter_elapsed = time.time() - chapter_start

        if entity_id is None:
            chapter_result = ChapterResult(chapter_name=chapter.name)
            chapter_result.errors.append(ValidationError(
                chapter=chapter.name,
                error_type="INVOKE_FAILED",
                message="Failed to create entity"
            ))
            chapter_result.elapsed_time = chapter_elapsed
            result.chapters.append(chapter_result)
            continue

        # Validate chapter
        chapter_result = validate_chapter(
            chapter, agent, all_entity_ids
        )
        chapter_result.elapsed_time = chapter_elapsed

        if verbose:
            print(f"\nCreated: {len(chapter_result.entities_created)}")
            for e in chapter_result.entities_created:
                print(f"  - {e['class']}: {e['fields']}")

            print(f"Reused: {len(chapter_result.entities_reused)}")
            for eid in chapter_result.entities_reused:
                print(f"  - {eid}")

            print(f"Modified: {len(chapter_result.entities_modified)}")
            for e in chapter_result.entities_modified:
                print(f"  - {e['class']}: {e['expected_fields_after']}")

            if chapter_result.errors:
                print(f"\nErrors: {len(chapter_result.errors)}")
                for err in chapter_result.errors:
                    print(f"  [{err.error_type}] {err.message}")

        # Store entities in vector store for next chapter
        agent.store_entities()

        # Update tracking
        all_entity_ids.update(agent.entities.keys())
        result.chapters.append(chapter_result)

        # Clear agent state for next chapter
        agent.clear()

    # Calculate totals
    for chapter in result.chapters:
        result.total_errors += len(chapter.errors)

    result.passed = result.total_errors == 0

    total_elapsed = time.time() - total_start
    if verbose:
        print(f"\nTotal benchmark time: {total_elapsed:.2f}s")

    return result


def print_summary(result: BenchmarkResult):
    """Print a summary of the benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"\nChapters processed: {len(result.chapters)}")

    total_created = sum(len(c.entities_created) for c in result.chapters)
    total_reused = sum(len(c.entities_reused) for c in result.chapters)
    total_modified = sum(len(c.entities_modified) for c in result.chapters)
    print(f"Total entities created: {total_created}")
    print(f"Total entities reused: {total_reused}")
    print(f"Total entities modified: {total_modified}")

    print("\nPer-chapter breakdown:")
    for chapter in result.chapters:
        status = (
            "PASS" if not chapter.errors else "FAIL"
        )
        print(
            f"\n  [{status}] {chapter.chapter_name} "
            f"({chapter.elapsed_time:.2f}s)"
        )
        print(f"    New: {len(chapter.entities_created)}, "
              f"Reused: {len(chapter.entities_reused)}, "
              f"Modified: {len(chapter.entities_modified)}, "
              f"Errors: {len(chapter.errors)}")

        if chapter.errors:
            for err in chapter.errors:
                print(f"      [{err.error_type}] {err.message}")

    print("\n" + "="*60)
    if result.passed:
        print("RESULT: PASSED")
    else:
        print(f"RESULT: FAILED ({result.total_errors} errors)")
    print("="*60)


if __name__ == "__main__":
    result = run_benchmark(verbose=True)
    print_summary(result)
