"""OoldAgent3 - Multi-step entity creation with dynamic schema generation.

Uses a 5-step pipeline:
1. Schema detection - identify all entities and their schemas from prompt
2. Fillable property definition - per entity, identify which properties to fill
3A. Property extraction - extract values with range properties as entity ID enums
4. Entity construction - per entity, build final JSON via LLM with OSW-IDs
5. Deduplication - check against existing entities via RAG + LLM judging
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from opensemantic.v1 import OswBaseModel
import opensemantic.core.v1  # noqa: F401 needed for eval
import opensemantic.base.v1  # noqa: F401 needed for eval
import opensemantic.lab.v1   # noqa: F401 needed for eval

from llm_init import get_llm, model_supports_structured_output
from oold_agent import _schema_build_cache
from osl_init import lookup_excact_matching_entity, get_osl_client
from rag_init import get_vector_store
from schema_catalog import (
    get_cached_inventory,
    get_data_schema_inventory_markdown,
)
from util import (
    add_max_length_constraints,
    modify_schema,
    post_process_llm_json_response,
    deep_copy,
    remove_empty,
    remove_nulls,
)

# ---------------------------------------------------------------------------
# Step 1: Schema Detection Models (static JSON-SCHEMA)
# ---------------------------------------------------------------------------

class DetectedEntity(BaseModel):
    """A single entity detected in the prompt."""
    id: str = Field(
        ...,
        description=(
            "A unique identifier for this entity within the context, "
            "e.g. 'entity_11', 'entity_12'. "
            "Entities referring to the same real-world object must "
            "share the same id."
        ),
    )
    schema_path: str = Field(
        ...,
        description=(
            "Full path of the selected data model, "
            "e.g. 'opensemantic.base.v1.Event'"
        ),
    )
    description: str = Field(
        ...,
        max_length=1000,
        description=(
            "A lossless summary of the relevant information about this "
            "entity from the prompt, including key details that belong "
            "to this entity. Be brief -- do not exceed the original "
            "prompt's description length."
        ),
    )


class SchemaDetectionResult(BaseModel):
    """Result of step 1: all detected entities."""
    entities: List[DetectedEntity] = Field(
        ..., description="List of all entities identified in the prompt"
    )


# ---------------------------------------------------------------------------
# Prompt templates (generic, no domain-specific references)
# ---------------------------------------------------------------------------

_STEP1_SYSTEM_PROMPT = """\
You are an expert data modeling assistant.

Your task: Given a natural language description, identify ALL distinct entities \
mentioned and map each to the most appropriate data model schema.

## Rules
1. Use the SPECIFIC AS NEEDED BUT GENERIC AS POSSIBLE schema available. \
Pick the schema that best fits the entity described — not too generic \
(avoid the root entity type) and not too specific (avoid narrow subtypes \
unless the description warrants it).
2. Do not create redundant entities.
3. If the same real-world entity is mentioned multiple times, use the SAME id \
and consolidate all information about it into one description.
4. Entities should be linked via range properties where the schema defines them.
5. You MUST use full schema paths from the valid list below.
6. Only create entities for which you have actual information in the prompt. \
Do not invent placeholder entities.
7. Assign each entity a unique id like 'entity_11', 'entity_12', etc.

## Available Schemas

{schema_inventory}

## Valid Schema Paths (use one of these exactly)

{valid_paths_list}
"""

_STEP1_USER_PROMPT = """\
Analyze the following description and identify all distinct entities:

{prompt}

Return a structured list of all entities found with their ids, schema paths, \
and descriptions.\
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultiStepAgent(BaseModel):
    """Agent that creates linked entities using a multi-step pipeline.

    Steps:
    1. Schema detection - identify all entities and their schemas
    2. Fillable property definition - per entity, identify fillable properties
    3A. Property extraction - extract values with range refs as entity ID enums
    4. Entity construction - build final JSON per entity via LLM
    5. Deduplication - check against existing entities via RAG + LLM
    """

    vector_store: Optional[Any] = None
    """Vector store for entity lookup. If None, builds one on first use."""

    llm: Optional[Any] = None
    """Language model to use. If None, uses default from get_llm()."""

    use_llm_judge: bool = True
    """Whether to use LLM judge for entity comparison."""

    entities: Dict[str, Any] = Field(default_factory=dict)
    """Constructed OswBaseModel instances keyed by IRI."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        if self.vector_store is None:
            object.__setattr__(self, 'vector_store', get_vector_store())

    def _get_llm(self):
        """Get the LLM, creating one if needed."""
        if self.llm is None:
            object.__setattr__(self, "llm", get_llm())
        return self.llm

    # ------------------------------------------------------------------
    # Schema name validation (shared with oold_agent2)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_schema_names(
        entities: List[DetectedEntity],
        valid_paths: List[str],
    ) -> List[DetectedEntity]:
        """Validate and correct schema names via suffix matching."""
        for entity in entities:
            if entity.schema_path in valid_paths:
                continue
            candidates = [
                p for p in valid_paths
                if p.endswith("." + entity.schema_path)
            ]
            if len(candidates) == 1:
                print(
                    f"  Warning: Corrected schema "
                    f"'{entity.schema_path}' -> '{candidates[0]}'"
                )
                entity.schema_path = candidates[0]
            elif len(candidates) > 1:
                print(
                    f"  Warning: Ambiguous schema "
                    f"'{entity.schema_path}', using first: {candidates[0]}"
                )
                entity.schema_path = candidates[0]
            else:
                print(
                    f"  Warning: Unknown schema "
                    f"'{entity.schema_path}', no match in inventory"
                )
        return entities

    # ------------------------------------------------------------------
    # Schema helpers (shared with oold_agent2)
    # ------------------------------------------------------------------

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
    def _extract_range_properties(schema: dict) -> Dict[str, str]:
        """Extract properties that have a 'range' annotation.

        Returns dict mapping property_name -> range_schema_id.
        """
        range_props = {}
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if "range" in prop_schema:
                range_props[prop_name] = prop_schema["range"]
        return range_props

    # ------------------------------------------------------------------
    # Step 1: Schema Detection
    # ------------------------------------------------------------------

    def _step1_detect_entities(
        self, prompt: str
    ) -> List[DetectedEntity]:
        """Step 1: Detect entities and their schemas from the prompt.

        Uses a static JSON schema to identify entities with:
        - id: unique identifier within this context
        - schema_path: full path of the data model
        - description: relevant information from the prompt
        """
        print("\n" + "=" * 60)
        print("Step 1: Schema Detection")
        print("=" * 60)

        schema_inventory = get_data_schema_inventory_markdown(
            include_properties=False,
            include_property_def=False,
        )

        inventory = get_cached_inventory()
        valid_paths = inventory.get_all_full_paths()
        valid_paths_str = "\n".join(f"- {p}" for p in valid_paths)

        system_prompt = _STEP1_SYSTEM_PROMPT.format(
            schema_inventory=schema_inventory,
            valid_paths_list=valid_paths_str,
        )
        user_prompt = _STEP1_USER_PROMPT.format(prompt=prompt)

        schema = SchemaDetectionResult.model_json_schema()
        schema = modify_schema(schema)

        llm = self._get_llm()
        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(schema=schema, strict=True)
        else:
            response_format = ToolStrategy(schema=schema)

        agent = create_agent(
            model=llm, response_format=response_format, tools=[]
        )

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

        parsed = SchemaDetectionResult(**result["structured_response"])
        self._validate_schema_names(parsed.entities, valid_paths)

        print(
            f"Detected {len(parsed.entities)} entities "
            f"in {elapsed:.2f}s:"
        )
        for entity in parsed.entities:
            print(
                f"  {entity.id}: {entity.schema_path} "
                f"\"{entity.description}\""
            )

        return parsed.entities

    # ------------------------------------------------------------------
    # Step 2: Fillable Property Definition
    # ------------------------------------------------------------------

    def _step2_identify_fillable_properties(
        self,
        detected_entities: List[DetectedEntity],
        entity_schemas: Dict[str, dict],
        original_prompt: str,
    ) -> Dict[str, List[str]]:
        """Step 2: For each entity, identify which properties can be filled.

        Uses a single LLM call with an auto-generated JSON schema where
        each entity has an array of strings with enum values being all
        possible properties of that entity's schema.
        """
        print("\n" + "=" * 60)
        print("Step 2: Fillable Property Definition")
        print("=" * 60)

        # Build per-entity property enum schema
        entity_property_schemas = {}
        for entity in detected_entities:
            schema = entity_schemas.get(entity.id)
            if not schema:
                continue
            props = list(schema.get("properties", {}).keys())
            if not props:
                continue
            entity_property_schemas[entity.id] = {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": props,
                },
                "description": (
                    f"List of fillable property names for {entity.id} "
                    f"({entity.schema_path})"
                ),
            }

        if not entity_property_schemas:
            print("Error: No entity properties to identify")
            return {}

        # Wrap in fillable_properties object
        fillable_schema = {
            "title": "FillablePropertiesSchema",
            "description": (
                "For each detected entity, list which properties can be filled "
                "with actual information from the description or by linking to "
                "other entities in the plan. Do not include properties where "
                "you would need to invent or hallucinate data."
            ),
            "type": "object",
            "properties": {
                "fillable_properties": {
                    "type": "object",
                    "properties": entity_property_schemas,
                    "required": list(entity_property_schemas.keys()),
                    "additionalProperties": False,
                }
            },
            "required": ["fillable_properties"],
            "additionalProperties": False,
        }

        # Build context with entity descriptions and available properties
        entity_detail_lines = []
        for entity in detected_entities:
            schema = entity_schemas.get(entity.id)
            if not schema:
                continue
            properties = schema.get("properties", {})
            prop_info = {
                prop: {
                    "type": prop_def.get("type", "unknown"),
                    "description": prop_def.get("description", ""),
                    "range": prop_def.get("range", None),
                }
                for prop, prop_def in properties.items()
            }
            entity_detail_lines.append(
                f"\n### {entity.id} ({entity.schema_path})\n"
                f"Description: \"{entity.description}\"\n"
                f"Available properties:\n"
                f"{json.dumps(prop_info, indent=2)}"
            )
        entity_detail = "\n".join(entity_detail_lines)

        # Build entity reference for range property awareness
        entity_ref_lines = []
        for entity in detected_entities:
            short_type = entity.schema_path.split(".")[-1]
            entity_ref_lines.append(
                f"  {entity.id}: {short_type} ({entity.schema_path})"
            )
        entity_ref = "\n".join(entity_ref_lines)

        sys_prompt = (
            "You identify which schema properties can be filled from "
            "entity descriptions without hallucinating. "
            "For properties with a 'range' annotation, include them as "
            "fillable if a matching entity of the expected type exists "
            "in the entity plan."
        )

        user_prompt = (
            f"Given the following original prompt:\n"
            f"\"{original_prompt}\"\n\n"
            f"And the following entities detected from it:\n"
            f"{entity_ref}\n\n"
            f"For each entity below, list ONLY the property names that "
            f"can be filled with actual information from the description "
            f"or by linking to other entities in the plan. "
            f"Do not include properties where you would need to invent "
            f"or hallucinate data.\n"
            f"{entity_detail}"
        )

        fillable_schema = modify_schema(fillable_schema)

        llm = self._get_llm()
        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(
                schema=fillable_schema, strict=True
            )
        else:
            response_format = ToolStrategy(schema=fillable_schema)

        agent = create_agent(
            model=llm, response_format=response_format, tools=[]
        )

        t0 = time.time()
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            })
        except Exception as e:
            print(f"Error in fillable property identification: {e}")
            # Fallback: use all properties for each entity
            return {
                entity.id: list(
                    entity_schemas.get(entity.id, {})
                    .get("properties", {}).keys()
                )
                for entity in detected_entities
                if entity.id in entity_schemas
            }

        elapsed = time.time() - t0

        if "structured_response" not in result:
            print(f"Error: No structured_response: {result}")
            return {}

        parsed = result["structured_response"]
        fillable_map = parsed.get("fillable_properties", parsed)

        # Validate property names against actual schema properties
        for eid, props in fillable_map.items():
            schema = entity_schemas.get(eid)
            if schema:
                valid_props = set(schema.get("properties", {}).keys())
                fillable_map[eid] = [
                    p for p in props if p in valid_props
                ]

        print(f"Identified fillable properties in {elapsed:.2f}s:")
        for eid, props in fillable_map.items():
            print(f"  {eid}: {props}")

        return fillable_map

    # ------------------------------------------------------------------
    # Step 3A: Property Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_compatible_entity_ids(
        range_schema_id: str,
        detected_entities: List[DetectedEntity],
        inventory,
    ) -> List[str]:
        """Find entity IDs compatible with a range schema_id.

        Returns IDs of entities whose schema matches or is a subclass
        of the range target type.
        """
        compatible = []
        range_info = inventory.get_by_schema_id(range_schema_id)
        if not range_info:
            return compatible

        for entity in detected_entities:
            entity_info = None
            for item in inventory.items.values():
                if item.full_path == entity.schema_path:
                    entity_info = item
                    break

            if entity_info is None:
                continue

            # Compatible if same type or entity is a subclass of range type
            if (entity_info.full_path == range_info.full_path or
                    range_info.full_path in entity_info.subclass_of):
                compatible.append(entity.id)

        return compatible

    def _step3a_extract_properties(
        self,
        detected_entities: List[DetectedEntity],
        entity_schemas: Dict[str, dict],
        fillable_map: Dict[str, List[str]],
        original_prompt: str,
    ) -> Dict[str, Dict[str, str]]:
        """Step 3A: Extract property values for all entities in one LLM call.

        Builds an auto-generated JSON schema where:
        - Literal properties are type: string
        - Range properties are type: string with enum of compatible entity IDs
        - All fillable + required properties are mandatory

        Original prompt + step 1 result provided as context.
        """
        print("\n" + "=" * 60)
        print("Step 3A: Property Extraction")
        print("=" * 60)

        inventory = get_cached_inventory()

        # Build per-entity extraction schemas
        entity_extraction_schemas = {}
        for entity in detected_entities:
            schema = entity_schemas.get(entity.id)
            if not schema:
                continue

            fillable = fillable_map.get(entity.id, [])
            required = schema.get("required", [])
            all_needed = list(set(fillable) | set(required))

            range_props = self._extract_range_properties(schema)
            properties = schema.get("properties", {})

            prop_schemas = {}
            for prop_name in all_needed:
                if prop_name not in properties:
                    continue

                if prop_name in range_props:
                    # Range property: enum of compatible entity IDs
                    range_schema_id = range_props[prop_name]
                    compatible_ids = self._find_compatible_entity_ids(
                        range_schema_id, detected_entities, inventory
                    )
                    if compatible_ids:
                        prop_schemas[prop_name] = {
                            "type": "string",
                            "enum": compatible_ids,
                            "description": (
                                "Reference to a linked entity "
                                "(use one of the entity IDs)"
                            ),
                        }
                    else:
                        # No compatible entities, use plain string
                        prop_schemas[prop_name] = {
                            "type": "string",
                            "description": (
                                "Reference to a linked entity "
                                "(textual description)"
                            ),
                        }
                else:
                    # Literal property: type string with maxLength
                    prop_desc = (
                        properties[prop_name].get("description", "")
                        or f"Value for {prop_name}"
                    )
                    pn_lower = prop_name.lower()
                    if "name" in pn_lower or "label" in pn_lower:
                        max_len = 200
                    else:
                        max_len = 1000
                    prop_schemas[prop_name] = {
                        "type": "string",
                        "maxLength": max_len,
                        "description": prop_desc,
                    }

            if prop_schemas:
                entity_extraction_schemas[entity.id] = {
                    "type": "object",
                    "properties": prop_schemas,
                    "required": list(prop_schemas.keys()),
                    "additionalProperties": False,
                }

        if not entity_extraction_schemas:
            print("Error: No entity schemas for extraction")
            return {}

        extraction_schema = {
            "title": "EntityPropertyExtractionSchema",
            "description": (
                "For each detected entity, fill all listed properties based on "
                "the original description and entity plan. "
                "For literal properties, provide the actual value from the "
                "description. "
                "For range properties (those with enum options), select the "
                "entity ID of the linked entity from the enum. "
                "Do NOT invent information not present in the descriptions."
            ),
            "type": "object",
            "properties": {
                "entities": {
                    "type": "object",
                    "properties": entity_extraction_schemas,
                    "required": list(
                        entity_extraction_schemas.keys()
                    ),
                    "additionalProperties": False,
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        }

        # Build entity reference context
        ref_lines = []
        for entity in detected_entities:
            short_type = entity.schema_path.split(".")[-1]
            ref_lines.append(
                f"  {entity.id}: {short_type} "
                f"-- \"{entity.description}\""
            )
        entity_ref = "\n".join(ref_lines)

        sys_prompt = (
            "You are an expert data extraction assistant. "
            "Extract property values from entity descriptions. "
            "For literal properties, provide the actual value from the "
            "description. Use concise, factual phrasing -- do not "
            "elaborate, paraphrase at length beyond what is stated. "
            "For range properties (those with enum options), select the "
            "entity ID of the linked entity from the enum. "
            "You MUST fill ALL properties -- they are all required. "
            "Do NOT invent information not present in the descriptions."
        )

        user_prompt = (
            f"## Original Description\n\n{original_prompt}\n\n"
            f"## Detected Entities\n\n{entity_ref}\n\n"
            f"Fill all listed properties for each entity based on the "
            f"original description and entity plan above."
        )

        extraction_schema = modify_schema(extraction_schema)

        llm = self._get_llm()
        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(
                schema=extraction_schema, strict=True
            )
        else:
            response_format = ToolStrategy(schema=extraction_schema)

        agent = create_agent(
            model=llm, response_format=response_format, tools=[]
        )

        t0 = time.time()
        try:
            result = agent.invoke({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            })
        except Exception as e:
            print(f"Error extracting properties: {e}")
            return {}

        elapsed = time.time() - t0

        if "structured_response" not in result:
            print(f"Error: No structured_response: {result}")
            return {}

        parsed = result["structured_response"]
        extracted = parsed.get("entities", parsed)

        print(f"Extracted properties in {elapsed:.2f}s:")
        for eid, props in extracted.items():
            print(f"  [{eid}] {props}")

        return extracted

    # ------------------------------------------------------------------
    # Step 4: Entity Construction
    # ------------------------------------------------------------------

    def _step4_construct_entities(
        self,
        detected_entities: List[DetectedEntity],
        entity_schemas: Dict[str, dict],
        entity_classes: Dict[str, Any],
        fillable_map: Dict[str, List[str]],
        extracted_values: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Tuple[dict, Any, uuid.UUID]], Dict[str, Tuple[str, uuid.UUID]]]:
        """Step 4: Per entity, construct final JSON via LLM.

        - Generates OSW-IDs for all entities
        - Replaces readable entity IDs with OSW-IDs in extracted values
        - Per entity: asks LLM to build final JSON based on original schema
          (with non-fillable/non-required properties removed) and the
          filled properties from step 3A

        Returns:
            Tuple of (results_dict, id_map) where:
            - results_dict: entity_id -> (result_dict, schema_cls, uuid)
            - id_map: entity_id -> (osw_id, uuid)
        """
        print("\n" + "=" * 60)
        print("Step 4: Entity Construction")
        print("=" * 60)

        # Generate OSW-IDs for all entities
        id_map: Dict[str, Tuple[str, uuid.UUID]] = {}
        for entity in detected_entities:
            entity_uuid = uuid.uuid4()
            osw_id = "Item:OSW" + entity_uuid.hex
            id_map[entity.id] = (osw_id, entity_uuid)

        print("Generated OSW-IDs:")
        for seg_id, (osw_id, _) in id_map.items():
            print(f"  {seg_id} -> {osw_id}")

        # Replace entity IDs with OSW-IDs in extracted values
        resolved_values: Dict[str, Dict[str, str]] = {}
        for eid, props in extracted_values.items():
            resolved = {}
            for prop_name, value in props.items():
                if value in id_map:
                    resolved[prop_name] = id_map[value][0]
                else:
                    resolved[prop_name] = value
            resolved_values[eid] = resolved

        print("\n>> Resolved property values (entity IDs -> OSW-IDs):")
        for eid, props in resolved_values.items():
            print(f"  [{eid}] {props}")

        # Construct entities in parallel
        results: Dict[str, Tuple[dict, Any, uuid.UUID]] = {}

        def _construct_entity(
            entity: DetectedEntity,
        ) -> Optional[Tuple[str, dict, Any, uuid.UUID]]:
            schema = entity_schemas.get(entity.id)
            schema_cls = entity_classes.get(entity.id)
            if not schema or not schema_cls:
                return None

            fillable = fillable_map.get(entity.id, [])
            filtered = self._filter_schema(schema, fillable)

            try:
                filtered = modify_schema(filtered)
                filtered = add_max_length_constraints(filtered)
            except Exception as e:
                print(
                    f"  Error modifying schema for [{entity.id}]: {e}"
                )
                return None

            hints = resolved_values.get(entity.id, {})
            osw_id, entity_uuid = id_map[entity.id]

            result_dict = self._create_single_entity(
                entity_id=entity.id,
                schema_path=entity.schema_path,
                schema_cls=schema_cls,
                filtered_schema=filtered,
                entity_uuid=entity_uuid,
                property_hints=hints,
            )

            if result_dict is not None:
                return (entity.id, result_dict, schema_cls, entity_uuid)
            return None

        t0 = time.time()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_construct_entity, entity): entity
                for entity in detected_entities
            }
            for future in as_completed(futures):
                entity = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  Exception for [{entity.id}]: {e}")
                    continue
                if result is not None:
                    eid, result_dict, schema_cls, entity_uuid = result
                    results[eid] = (result_dict, schema_cls, entity_uuid)

        elapsed = time.time() - t0
        print(f"  {len(results)} entities constructed in {elapsed:.2f}s")

        return results, id_map

    def _create_single_entity(
        self,
        entity_id: str,
        schema_path: str,
        schema_cls,
        filtered_schema: dict,
        entity_uuid: uuid.UUID,
        property_hints: Optional[Dict[str, str]] = None,
    ) -> Optional[dict]:
        """Create a single entity via LLM structured output with retries.

        Uses the original schema (filtered to fillable + required properties)
        and the filled property values from step 3A as hints.
        """
        sys_prompt = (
            "You are an expert data modeling assistant. "
            "You always answer in valid JSON according to the provided "
            "schema. "
            "Fill ONLY the properties listed below with the given values. "
            "Keep all string values concise: use the exact values provided "
            "in the property hints without elaboration or paraphrasing "
            " beyond what is stated. "
            "Do not invent any new information. "
            "Do not generate dummy or placeholder values. "
            "If you do not have enough information for a field, "
            "leave it empty or null."
        )

        schema_str = json.dumps(filtered_schema, indent=2)

        hints_section = ""
        if property_hints:
            hints_section = (
                f"\nProperty values to use:\n"
                f"{json.dumps(property_hints, indent=2)}\n"
            )

        user_prompt = (
            f"Create a JSON document for the following entity.\n"
            f"Schema: {schema_path}\n"
            f"{hints_section}\n"
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
            model=llm, response_format=response_format, tools=[]
        )

        max_retries = 3
        for attempt in range(max_retries):
            print(
                f"  Creating entity [{entity_id}] "
                f"{schema_path} (attempt {attempt + 1})..."
            )

            try:
                result = agent.invoke({
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                })
            except Exception as e:
                print(f"  Error invoking agent for [{entity_id}]: {e}")
                return None

            if "structured_response" not in result:
                print(
                    f"  Error: No structured_response "
                    f"for [{entity_id}]: {result}"
                )
                return None

            result_dict = post_process_llm_json_response(
                result["structured_response"]
            )
            result_dict["uuid"] = str(entity_uuid)

            try:
                schema_cls(**result_dict)
                print(f"  Entity [{entity_id}] validated successfully")
                return result_dict
            except Exception as e:
                e_msg = str(e)
                print(
                    f"  Validation error for [{entity_id}]: {e_msg}"
                )
                user_prompt += (
                    f"\n\nThe previous response had an error: {e_msg}"
                )
                if attempt >= max_retries - 1:
                    print(
                        f"  Max retries reached for "
                        f"[{entity_id}], skipping"
                    )
                    return None

        return None

    # ------------------------------------------------------------------
    # Step 5: Deduplication
    # ------------------------------------------------------------------

    def _step5_deduplicate(
        self,
        seg_to_iri: Dict[str, str],
    ):
        """Step 5: Deduplicate against existing entities via RAG + LLM."""
        print("\n" + "=" * 60)
        print("Step 5: Deduplication")
        print("=" * 60)

        iri_remap: Dict[str, str] = {}

        for seg_id, iri in list(seg_to_iri.items()):
            entity = self.entities.get(iri)
            if entity is None:
                continue

            data_dict = json.loads(entity.json())
            data_dict.pop("uuid", None)
            data_dict = remove_empty(data_dict)
            data_dict = remove_nulls(data_dict)
            description = json.dumps(data_dict)

            lookup_result = lookup_excact_matching_entity(
                vector_store=self.vector_store,
                description=description,
                llm_judge=self.use_llm_judge,
            )

            if lookup_result is not None and lookup_result.osw_id:
                if lookup_result.needs_update:
                    self._merge_entity_data(
                        entity_id=lookup_result.osw_id,
                        existing_data=lookup_result.existing_data,
                        existing_type=lookup_result.existing_type,
                        data_instance=entity,
                    )
                    if iri != lookup_result.osw_id:
                        self.entities.pop(iri, None)
                        iri_remap[iri] = lookup_result.osw_id
                else:
                    print(
                        f"  Reusing existing entity: "
                        f"{lookup_result.osw_id} for [{seg_id}]"
                    )
                    self.entities.pop(iri, None)
                    self.entities[lookup_result.osw_id] = entity
                    iri_remap[iri] = lookup_result.osw_id

        if iri_remap:
            self._remap_range_references(iri_remap)

    # ------------------------------------------------------------------
    # Deduplication helpers (shared with oold_agent2)
    # ------------------------------------------------------------------

    def _merge_entity_data(
        self,
        entity_id: str,
        existing_data: Dict[str, Any],
        existing_type: str,
        data_instance: OswBaseModel,
    ) -> str:
        """Merge existing entity data with new data_instance."""
        print(f"\n>> Merging entity {entity_id}")
        print(f"   Existing type: {existing_type}")

        new_data = json.loads(data_instance.json(exclude_none=True))
        new_data.pop("uuid", None)
        print(f"Existing entity data: {existing_data}")
        print(f"New data instance: {new_data}")
        merged_data = {**existing_data, **new_data}
        print(f"   Merged fields: {list(merged_data.keys())}")

        try:
            schema_cls = None

            if existing_type:
                inventory = get_cached_inventory()
                schema_info = inventory.get_by_schema_id(existing_type)
                if schema_info and schema_info.cls_obj:
                    schema_cls = schema_info.cls_obj
                    print(f"   Using schema: {schema_info.full_path}")

            if schema_cls is None:
                schema_cls = data_instance.__class__
                print(
                    f"   Using fallback schema: {schema_cls.__name__}"
                )

            merged_entity = schema_cls(**merged_data)
            self.entities[entity_id] = merged_entity
            print(f"   Entity {entity_id} stored with merged data")

        except Exception as e:
            print(f"   Error creating merged entity: {e}")
            self.entities[entity_id] = data_instance
            print(
                f"   Fallback: stored new instance for {entity_id}"
            )

        return entity_id

    def _remap_range_references(self, iri_remap: Dict[str, str]):
        """Remap range property references after deduplication."""
        for iri, entity in list(self.entities.items()):
            data = json.loads(entity.json(exclude_none=True))
            changed = False
            for key, value in data.items():
                if isinstance(value, str) and value in iri_remap:
                    data[key] = iri_remap[value]
                    changed = True
            if changed:
                self.entities[iri] = entity.__class__(**data)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """Run the full 5-step entity construction pipeline.

        1. Schema detection - identify all entities and their schemas
        2. Fillable property definition - per entity via dynamic enum schema
        3A. Property extraction - single LLM call, range props as entity
            ID enums, literal props as strings
        4. Entity construction - per entity LLM call with OSW-IDs and
            filtered original schema
        5. Deduplication - RAG + LLM judging against vector store

        Returns dict mapping entity IRI -> OswBaseModel instance.
        """
        self.entities = {}

        # Step 1: Detect entities and schemas
        detected_entities = self._step1_detect_entities(prompt)
        if not detected_entities:
            print("Error: No entities detected")
            return self.entities

        # Load schemas for all detected entities
        print("\n>> Loading schemas...")
        entity_schemas: Dict[str, dict] = {}
        entity_classes: Dict[str, Any] = {}

        for entity in detected_entities:
            try:
                schema_cls, target_schema = self._load_schema(
                    entity.schema_path
                )
                entity_schemas[entity.id] = target_schema
                entity_classes[entity.id] = schema_cls
            except Exception as e:
                print(
                    f"  Error loading schema for "
                    f"{entity.schema_path}: {e}"
                )

        if not entity_schemas:
            print("Error: No schemas could be loaded")
            return self.entities

        # Step 2: Identify fillable properties
        fillable_map = self._step2_identify_fillable_properties(
            detected_entities, entity_schemas, prompt
        )

        # Step 3A: Extract property values
        extracted_values = self._step3a_extract_properties(
            detected_entities, entity_schemas, fillable_map, prompt
        )

        # Step 4: Construct entities (generates OSW-IDs + LLM calls)
        results, id_map = self._step4_construct_entities(
            detected_entities, entity_schemas, entity_classes,
            fillable_map, extracted_values,
        )

        # Build OswBaseModel instances
        print("\n>> Constructing final OswBaseModel instances...")
        seg_to_iri: Dict[str, str] = {}
        for entity_id, (result_dict, schema_cls, _) in results.items():
            try:
                instance = schema_cls(**result_dict)
                iri = instance.get_iri()
                self.entities[iri] = instance
                seg_to_iri[entity_id] = iri
                print(f"  [{entity_id}] -> {iri}")
            except Exception as e:
                print(f"  Error constructing [{entity_id}]: {e}")

        # Step 5: Deduplicate
        self._step5_deduplicate(seg_to_iri)

        print(
            f"\n>> Entity construction complete: "
            f"{len(self.entities)} entities"
        )
        return self.entities

    # ------------------------------------------------------------------
    # Interface methods (compatible with benchmark.py)
    # ------------------------------------------------------------------

    def get_entities(self) -> Dict[str, Any]:
        """Get all created entities."""
        return self.entities

    def store_entities(self):
        """Store all created entities in vector store."""
        for entity_id, entity in self.entities.items():
            jsondata = json.loads(entity.json(exclude_none=True))
            doc = Document(
                id=entity_id,
                page_content=json.dumps({"jsondata": jsondata}),
                metadata={
                    "name": jsondata.get("name", ""),
                    "type": (
                        jsondata.get("type", [""])[0]
                        if isinstance(jsondata.get("type"), list)
                        else jsondata.get("type", "")
                    ),
                },
            )
            self.vector_store.add_documents(documents=[doc])
            print(f"Stored entity {entity_id} in vector store")

    def clear(self):
        """Clear all stored entities."""
        self.entities.clear()


# ---------------------------------------------------------------------------
# Convenience / Test
# ---------------------------------------------------------------------------

def invoke_test(test_prompt: str):
    """Quick test of full entity construction."""
    agent = MultiStepAgent()
    entities = agent.invoke(test_prompt)
    print(f"\nConstructed {len(entities)} entities:")
    for iri, entity in entities.items():
        print(f"  {iri}: {entity.json(exclude_none=True)}")


if __name__ == "__main__":
    osl_client = get_osl_client()
    test_prompt = (
        "A tensile test experiment #1 conducted by Dr. Jane Doe "
        ", employed at Example Lab Corp.\n"
        "A tensile test experiment #2 conducted by Dr. John Doe, "
        "employed at Example Lab Corp.\n"
        "A tensile test experiment #3 conducted by Dr. John Doe "
        "(john.doe@example-lab.com)"
    )

    invoke_test(test_prompt)
