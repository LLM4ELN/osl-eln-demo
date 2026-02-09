"""OoldAgent2 - Graph-aware entity segmentation agent.

Takes a prompt and all available schemas, then plans the full entity graph
in a single LLM call (segmentation step).
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from pydantic import BaseModel, Field

from opensemantic.v1 import OswBaseModel
import opensemantic.core.v1  # noqa: F401 needed for eval
import opensemantic.base.v1  # noqa: F401 needed for eval
import opensemantic.lab.v1   # noqa: F401 needed for eval

from llm_init import get_llm, model_supports_structured_output
from oold_agent import _schema_build_cache
from schema_catalog import (
    get_cached_inventory,
    get_data_schema_inventory_markdown,
)
from util import modify_schema, post_process_llm_json_response, deep_copy


# ---------------------------------------------------------------------------
# Variant A
# ---------------------------------------------------------------------------

class EntityDescription(BaseModel):
    """Variant A: Simple schema + text description."""
    schema_path: str = Field(
        ...,
        description=(
            "Full path of the selected data model, "
            "e.g. 'opensemantic.base.v1.Event'"
        ),
    )
    description: str = Field(
        ...,
        description=(
            "The part of the initial prompt where this entity is the "
            "subject, containing all relevant information"
        ),
    )


class SegmentationResultA(BaseModel):
    """Structured output wrapper for Variant A."""
    entities: List[EntityDescription] = Field(
        ..., description="List of all entities identified in the prompt"
    )


# ---------------------------------------------------------------------------
# Variant B
# ---------------------------------------------------------------------------

class EntityPropertyDescription(BaseModel):
    """Variant B: Schema + property key-value map."""
    schema_path: str = Field(
        ...,
        description=(
            "Full path of the selected data model, "
            "e.g. 'opensemantic.base.v1.Event'"
        ),
    )
    properties: Dict[str, str] = Field(
        ...,
        description=(
            "Key-value map where keys are property names from the schema "
            "and values are either literal values or textual descriptions "
            "of linked entities"
        ),
    )


class SegmentationResultB(BaseModel):
    """Structured output wrapper for Variant B."""
    entities: List[EntityPropertyDescription] = Field(
        ..., description="List of all entities identified in the prompt"
    )


# ---------------------------------------------------------------------------
# Variant C
# ---------------------------------------------------------------------------

class EntityPropertyMap(BaseModel):
    """Variant C: Schema + separated literal/range properties with IDs."""
    id: str = Field(
        ...,
        description=(
            "A unique identifier within this context, "
            "e.g. 'entity_1', 'event_xyz'"
        ),
    )
    schema_path: str = Field(
        ...,
        description=(
            "Full path of the selected data model, "
            "e.g. 'opensemantic.base.v1.Event'"
        ),
    )
    literal_properties: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key-value map for properties with literal values "
            "(strings, numbers, dates)"
        ),
    )
    range_properties: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key-value map for range properties (links to other entities). "
            "Values are the 'id' fields of other entities in this plan."
        ),
    )


class SegmentationResultC(BaseModel):
    """Structured output wrapper for Variant C."""
    entities: List[EntityPropertyMap] = Field(
        ...,
        description=(
            "List of all entities identified in the prompt "
            "with their properties and links"
        ),
    )


# ---------------------------------------------------------------------------
# Variant enum
# ---------------------------------------------------------------------------

class SegmentationVariant(str, Enum):
    A = "A"
    B = "B"
    C = "C"


_VARIANT_CONFIG = {
    SegmentationVariant.A: (SegmentationResultA, False),
    SegmentationVariant.B: (SegmentationResultB, True),
    SegmentationVariant.C: (SegmentationResultC, True),
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert data modeling assistant for electronic laboratory notebooks.

Your task: Given a natural language description of a laboratory activity, \
identify ALL distinct entities mentioned and map each to the most appropriate \
data model schema.

## Rules
1. Use the MOST SPECIFIC schema available (e.g. \
'opensemantic.core.v1.Person' for a person, not 'opensemantic.core.v1.Entity').
2. Use the MINIMAL number of entities needed — do not create redundant entities.
3. Place information in the most specific property available. For example, \
an address belongs in the 'postal_address' property, not in 'description'.
4. Entities should be linked via range properties where the schema defines them. \
For example, if an Event has a 'location' property with a range \
pointing to Site, the site should be a separate entity linked there.
5. You MUST use full schema paths from the valid list below \
(e.g. 'opensemantic.base.v1.Event').
6. Only create entities for which you have actual information in the prompt. \
Do not invent placeholder entities.

## Available Schemas

{schema_inventory}

## Valid Schema Paths (use one of these exactly)

{valid_paths_list}
"""

_USER_PROMPT = """\
Analyze the following description and identify all entities:

{prompt}

Return a structured plan of all entities found.\
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SegmentationAgent(BaseModel):
    """Agent that segments a prompt into a planned entity graph.

    Given a natural language description and the full schema catalog,
    produces a structured plan of all entities and their relationships
    in a single LLM call.
    """

    llm: Optional[Any] = None
    """Language model to use. If None, uses default from get_llm()."""

    entities: Dict[str, Any] = Field(default_factory=dict)
    """Constructed OswBaseModel instances keyed by IRI."""

    model_config = {"arbitrary_types_allowed": True}

    def _get_llm(self):
        """Get the LLM, creating one if needed."""
        if self.llm is None:
            object.__setattr__(self, "llm", get_llm())
        return self.llm

    # ------------------------------------------------------------------
    # Schema name validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_schema_names(
        entities: list,
        valid_paths: List[str],
    ) -> list:
        """Validate and correct schema names in the entity list.

        Attempts suffix matching when a schema name is not an exact match.
        """
        for entity in entities:
            if entity.schema_path in valid_paths:
                continue
            # Try suffix match
            candidates = [
                p for p in valid_paths if p.endswith("." + entity.schema_path)
            ]
            if len(candidates) == 1:
                print(
                    f"  Warning: Corrected schema "
                    f"'{entity.schema_path}' -> '{candidates[0]}'"
                )
                entity.schema_path = candidates[0]
            elif len(candidates) > 1:
                print(
                    f"  Warning: Ambiguous schema '{entity.schema_path}', "
                    f"using first match: {candidates[0]}"
                )
                entity.schema_path = candidates[0]
            else:
                print(
                    f"  Warning: Unknown schema '{entity.schema_path}', "
                    f"no match in inventory"
                )
        return entities

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment(
        self,
        prompt: str,
        variant: SegmentationVariant = SegmentationVariant.C,
    ) -> list:
        """Segment a prompt into a planned entity graph.

        Args:
            prompt: Natural language description of entities to create.
            variant: Which entity description variant to use (A, B, or C).

        Returns:
            List of entity descriptions (type depends on variant).
        """
        result_model, needs_full_properties = _VARIANT_CONFIG[variant]

        # Build schema inventory context
        schema_inventory = get_data_schema_inventory_markdown(
            include_properties=needs_full_properties,
            include_property_def=needs_full_properties,
        )

        # Valid paths for prompt reinforcement + post-validation
        inventory = get_cached_inventory()
        valid_paths = inventory.get_all_full_paths()
        valid_paths_str = "\n".join(f"- {p}" for p in valid_paths)

        # Prompts
        system_prompt = _SYSTEM_PROMPT.format(
            schema_inventory=schema_inventory,
            valid_paths_list=valid_paths_str,
        )
        user_prompt = _USER_PROMPT.format(prompt=prompt)

        # Structured output
        schema = result_model.model_json_schema()
        llm = self._get_llm()

        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(schema=schema, strict=True)
        else:
            response_format = ToolStrategy(schema=schema)

        agent = create_agent(
            model=llm,
            response_format=response_format,
            tools=[],
        )

        print(f"\n>> Segmenting prompt (variant {variant.value})...")
        t0 = time.time()
        result = agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        })

        elapsed = time.time() - t0

        if "structured_response" not in result:
            print(f"Error: No structured_response in result: {result}")
            return []

        parsed = result_model(**result["structured_response"])

        # Post-validate schema names
        self._validate_schema_names(parsed.entities, valid_paths)

        # Summary
        print(
            f"Segmentation result ({variant.value}): "
            f"{len(parsed.entities)} entities in {elapsed:.2f}s"
        )
        for entity in parsed.entities:
            print(f"  - {entity.schema_path}")

        return parsed.entities

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_schema(schema: dict, property_names: List[str]) -> dict:
        """Filter schema to only include specified properties + required."""
        filtered = deep_copy(schema)
        if "properties" in filtered:
            required = filtered.get("required", [])
            keep = set(property_names) | set(required)
            filtered["properties"] = {
                k: v for k, v in filtered["properties"].items()
                if k in keep
            }
        return filtered

    @staticmethod
    def _load_schema(schema_path: str) -> Tuple[Any, dict]:
        """Load and cache a schema class and its exported schema."""
        if schema_path in _schema_build_cache:
            cached = _schema_build_cache[schema_path]
            return cached["schema_cls"], cached["target_schema"]

        schema_cls: OswBaseModel = eval(schema_path)
        target_schema = schema_cls.export_schema()

        _schema_build_cache[schema_path] = {
            "schema_cls": schema_cls,
            "target_schema": target_schema,
        }
        return schema_cls, target_schema

    # ------------------------------------------------------------------
    # Single entity creation (LLM call with retries)
    # ------------------------------------------------------------------

    def _create_single_entity(
        self,
        entity: EntityPropertyMap,
        schema_cls,
        filtered_schema: dict,
        entity_uuid: uuid.UUID,
    ) -> Optional[dict]:
        """Create a single entity via LLM structured output with retries."""
        sys_prompt = (
            "You are an expert laboratory assistant. "
            "You always answer in valid JSON according to the provided schema. "
            "Fill ONLY the properties listed below with the given values. "
            "Do not invent any new information. "
            "Do not generate dummy or placeholder values. "
            "If you do not have enough information for a field, "
            "leave it empty or null."
        )

        schema_str = json.dumps(filtered_schema, indent=2)
        user_prompt = (
            f"Create a JSON document for the following entity.\n"
            f"Schema: {entity.schema_path}\n"
            f"Properties to fill:\n"
            f"{json.dumps(entity.literal_properties, indent=2)}\n\n"
            f"Use the following schema:\n{schema_str}"
        )

        llm = self._get_llm()
        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(
                schema=filtered_schema, strict=True
            )
        else:
            response_format = ToolStrategy(schema=filtered_schema)

        agent = create_agent(
            model=llm,
            response_format=response_format,
            tools=[],
        )

        max_retries = 3
        for attempt in range(max_retries):
            print(
                f"  Creating entity [{entity.id}] "
                f"{entity.schema_path} (attempt {attempt + 1})..."
            )

            try:
                result = agent.invoke({
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                })
            except Exception as e:
                print(f"  Error invoking agent for [{entity.id}]: {e}")
                return None

            if "structured_response" not in result:
                print(
                    f"  Error: No structured_response "
                    f"for [{entity.id}]: {result}"
                )
                return None

            result_dict = post_process_llm_json_response(
                result["structured_response"]
            )
            result_dict["uuid"] = str(entity_uuid)

            try:
                schema_cls(**result_dict)
                print(f"  Entity [{entity.id}] validated successfully")
                return result_dict
            except Exception as e:
                e_msg = str(e)
                print(f"  Validation error for [{entity.id}]: {e_msg}")
                user_prompt += (
                    f"\n\nThe previous response had an error: {e_msg}"
                )
                if attempt >= max_retries - 1:
                    print(
                        f"  Max retries reached for [{entity.id}], skipping"
                    )
                    return None

        return None

    # ------------------------------------------------------------------
    # Full entity construction pipeline
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """Run full entity construction pipeline.

        1. Segment prompt into entity graph (Variant C)
        2. Generate OSW-IDs
        3. Load/filter schemas
        4. Create entities in parallel via LLM
        5. Insert range property OSW-IDs
        6. Construct OswBaseModel instances

        Returns dict mapping entity IRI -> OswBaseModel instance.
        """
        self.entities = {}

        # Step 1: Segment
        segmentation = self.segment(prompt, SegmentationVariant.C)
        if not segmentation:
            print("Error: Segmentation returned no entities")
            return self.entities

        # Step 2: Generate OSW-IDs
        id_map: Dict[str, Tuple[str, uuid.UUID]] = {}
        for entity in segmentation:
            entity_uuid = uuid.uuid4()
            entity_id = "Item:OSW" + entity_uuid.hex
            id_map[entity.id] = (entity_id, entity_uuid)

        print(f"\n>> Generated IDs for {len(id_map)} entities:")
        for seg_id, (osw_id, _) in id_map.items():
            print(f"  {seg_id} -> {osw_id}")

        # Step 3: Load schemas and prepare filtered versions
        entity_preparations: List[
            Tuple[EntityPropertyMap, Any, dict, uuid.UUID]
        ] = []
        for entity in segmentation:
            try:
                schema_cls, target_schema = self._load_schema(
                    entity.schema_path
                )
            except Exception as e:
                print(f"  Error loading schema {entity.schema_path}: {e}")
                continue

            literal_keys = list(entity.literal_properties.keys())
            filtered = self._filter_schema(target_schema, literal_keys)

            try:
                filtered = modify_schema(filtered)
            except Exception as e:
                print(f"  Error modifying schema for [{entity.id}]: {e}")
                continue

            _, entity_uuid = id_map[entity.id]
            entity_preparations.append(
                (entity, schema_cls, filtered, entity_uuid)
            )

        # Step 4: Create entities in parallel
        print(f"\n>> Creating {len(entity_preparations)} entities...")
        t0 = time.time()
        results: Dict[str, Tuple[dict, Any]] = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._create_single_entity, ent, cls, schema, uid
                ): ent
                for ent, cls, schema, uid in entity_preparations
            }
            for future in as_completed(futures):
                entity = futures[future]
                try:
                    result_dict = future.result()
                except Exception as e:
                    print(f"  Exception for [{entity.id}]: {e}")
                    continue
                if result_dict is not None:
                    cls = next(
                        c for e, c, _, _ in entity_preparations
                        if e.id == entity.id
                    )
                    results[entity.id] = (result_dict, cls)

        elapsed = time.time() - t0
        print(f"  {len(results)} entities created in {elapsed:.2f}s")

        # Step 5: Insert range property OSW-IDs
        print("\n>> Inserting range property links...")
        for entity in segmentation:
            if entity.id not in results:
                continue
            result_dict, _ = results[entity.id]
            for prop_name, ref_id in entity.range_properties.items():
                if ref_id in id_map:
                    osw_id, _ = id_map[ref_id]
                    result_dict[prop_name] = osw_id
                    print(f"  [{entity.id}].{prop_name} -> {osw_id}")
                else:
                    print(
                        f"  Warning: [{entity.id}].{prop_name} "
                        f"references unknown '{ref_id}'"
                    )

        # Step 6: Construct OswBaseModel instances
        print("\n>> Constructing final entities...")
        for seg_id, (result_dict, schema_cls) in results.items():
            try:
                instance = schema_cls(**result_dict)
                iri = instance.get_iri()
                self.entities[iri] = instance
                print(f"  [{seg_id}] -> {iri}")
            except Exception as e:
                print(f"  Error constructing [{seg_id}]: {e}")

        print(
            f"\n>> Entity construction complete: "
            f"{len(self.entities)} entities"
        )
        return self.entities


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def segment_prompt(
    prompt: str,
    variant: SegmentationVariant = SegmentationVariant.C,
) -> list:
    """Convenience function to segment a prompt."""
    agent = SegmentationAgent()
    return agent.segment(prompt, variant)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def segmentation_test(test_prompt: str):
    """Quick test of segmentation."""
    for v in SegmentationVariant:
        print(f"\n{'='*60}")
        print(f"Variant {v.value}")
        print(f"{'='*60}")
        entities = segment_prompt(test_prompt, v)
        for e in entities:
            if isinstance(e, EntityPropertyMap):
                print(f"  [{e.id}] {e.schema_path}")
                print(f"    literals: {e.literal_properties}")
                print(f"    ranges:   {e.range_properties}")
            elif isinstance(e, EntityPropertyDescription):
                print(f"  {e.schema_path}: {e.properties}")
            else:
                print(f"  {e.schema_path}: {e.description}")

if __name__ == "__main__":
    test_prompt = (
        "A tensile test experiment #1 conducted by Dr. Jane Doe (jane.doe@example-lab.com), "
        "employed at Example Lab Corp."
    )

    _test_prompt = (
    """
The availability of stretchable conductive materials is a key requirement for the development of soft and wearable electronics. Although there are many promising materials, the characterization of these materials under realistic conditions is complex and a standardized and reliable procedure has not been etablished yet. We therefore introduce a comprehensive protocol for the practice-oriented dynamic electro-mechanical analysis of elastomer-particle composites. In addition to strain dependence (0–100% strain) and fatigue strength (10,000 cycles), this protocol aims in particular to clarify the influence of strain rate (0–100% s−1) on conductivity. Samples with the commonly used filler representatives carbon black and silver flakes with 20 vol% each were prepared and investigated. Silicone elastomers of different stiffness were used as matrix in order to determine its influence. We found that while the conductivity of the carbon black composites of about 1 × 102 S m−1 proved to be fatigue resistant and largely independent of the strain rate, the silver flake composites lost their initially higher conductivity of 1 × 104 S m−1 at high strain rates and increasing numbers of cycles. In addition, the use of a softer silicone matrix improved the performance of both particle composites, which was also demonstrated on an exemplary wearable electronic device.
    """
    )

    segmentation_test(test_prompt)
