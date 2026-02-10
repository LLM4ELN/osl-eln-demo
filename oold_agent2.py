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
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from opensemantic.v1 import OswBaseModel
import opensemantic.core.v1  # noqa: F401 needed for eval
import opensemantic.base.v1  # noqa: F401 needed for eval
import opensemantic.lab.v1   # noqa: F401 needed for eval

from llm_init import get_llm, model_supports_structured_output
from oold_agent import _schema_build_cache
from osl_init import lookup_excact_matching_entity
from rag_init import get_vector_store
from schema_catalog import (
    get_cached_inventory,
    get_data_schema_inventory_markdown,
)
from util import (
    modify_schema,
    post_process_llm_json_response,
    deep_copy,
    remove_empty,
    remove_nulls,
)


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

class KeyValuePair(BaseModel):
    """A single key-value pair for structured property representation."""
    key: str = Field(
        ..., description="Property name from the schema"
    )
    value: str = Field(
        ..., description="Literal value or textual description of a linked entity"
    )


class FillableProperties(BaseModel):
    """List of property names that can be filled from a description."""
    properties: list[str] = Field(
        ...,
        description=(
            "List of property names that can be filled with actual "
            "information from the description"
        ),
    )


class EntityPropertyDescription(BaseModel):
    """Variant B: Schema + property key-value list."""
    schema_path: str = Field(
        ...,
        description=(
            "Full path of the selected data model, "
            "e.g. 'opensemantic.base.v1.Event'"
        ),
    )
    property_list: List[KeyValuePair] = Field(
        ...,
        description=(
            "List of key-value pairs where keys are property names from "
            "the schema and values are either literal values or textual "
            "descriptions of linked entities"
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
    literal_properties: List[KeyValuePair] = Field(
        default_factory=list,
        description=(
            "List of key-value pairs for properties with literal values "
            "(strings, numbers, dates)"
        ),
    )
    range_properties: List[KeyValuePair] = Field(
        default_factory=list,
        description=(
            "List of key-value pairs for range properties "
            "(links to other entities). "
            "Values are the 'id' fields of other entities in this plan."
        ),
    )

    def literal_dict(self) -> Dict[str, str]:
        """Convert literal_properties to a dict."""
        return {kv.key: kv.value for kv in self.literal_properties}

    def range_dict(self) -> Dict[str, str]:
        """Convert range_properties to a dict."""
        return {kv.key: kv.value for kv in self.range_properties}


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
1. Use the SPECIFIC AS NEEDED BUT GENERIC AS POSSIBLE schema available, e.g. \
'opensemantic.core.v1.Person' for a normal person, not 'opensemantic.core.v1.Entity' (too generic) \
and not 'opensemantic.lab.v1.Researcher' (too specific) if the description gives no indication that the person is a researcher). \
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
        schema = modify_schema(schema)  # Ensure compatibility with older models and xgrammar
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
    # Property identification helpers
    # ------------------------------------------------------------------

    def _identify_fillable_properties(
        self,
        entity_description: str,
        schema: dict,
        entity_context: str = "",
    ) -> list[str]:
        """Ask LLM to identify which properties can be filled from
        description.

        Args:
            entity_description: Description of the current entity.
            schema: JSON schema for this entity type.
            entity_context: Summary of all entities in the plan, so the
                LLM can identify range properties as fillable when a
                matching entity exists.

        Returns list of property names that can be filled without
        hallucinating.
        """
        print("\n>> Identifying fillable properties...")

        properties = schema.get("properties", {})
        property_descriptions = {
            prop: {
                "type": props.get("type", "unknown"),
                "description": props.get("description", ""),
                "range": props.get("range", None)
            }
            for prop, props in properties.items()
        }

        fillable_schema = FillableProperties.model_json_schema()

        if model_supports_structured_output(self._get_llm(), tools=[]):
            response_format = ProviderStrategy(
                schema=fillable_schema,
                strict=True
            )
        else:
            response_format = ToolStrategy(schema=fillable_schema)

        agent = create_agent(
            model=self._get_llm(),
            response_format=response_format,
            tools=[],
        )

        context_section = ""
        if entity_context:
            context_section = (
                f"\n\nThe following entities exist in the plan:\n"
                f"{entity_context}\n\n"
                f"For properties with a 'range' annotation, include them "
                f"as fillable if a matching entity of the expected type "
                f"exists in the plan above.\n"
            )

        prompt = (
            f"Given the following entity description:\n"
            f"\"{entity_description}\"\n\n"
            f"And the following schema properties:\n"
            f"{json.dumps(property_descriptions, indent=2)}\n"
            f"{context_section}\n"
            f"List ONLY the property names that can be filled with actual "
            f"information from the description or by linking to entities "
            f"in the plan. "
            f"Do not include properties where you would need to invent or "
            f"hallucinate data."
        )

        try:
            result = agent.invoke({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You identify which schema properties can be "
                            "filled from a given description without "
                            "hallucinating."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            })

            if "structured_response" in result:
                parsed = FillableProperties(**result["structured_response"])
                # Filter to only valid property names
                valid = [p for p in parsed.properties if p in properties]
                print(f"Fillable properties: {valid}")
                return valid
            else:
                print(
                    "Warning: No structured_response, using all properties"
                )
                return list(properties.keys())
        except Exception as e:
            print(f"Error identifying fillable properties: {e}, using all")
            return list(properties.keys())

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
    # Property value extraction (Step 3)
    # ------------------------------------------------------------------

    def _build_extraction_schema(
        self,
        entity_info_list: List[dict],
    ) -> dict:
        """Build a dynamic JSON schema for property value extraction.

        Each entity becomes a named object with its specific fillable
        properties as required string fields — forcing the LLM to fill
        every property (including range properties).
        """
        inventory = get_cached_inventory()
        entity_schemas = {}

        for info in entity_info_list:
            props = {}
            for prop_name in info["literal_props"]:
                props[prop_name] = {
                    "type": "string",
                    "description": "Literal value from the description",
                }
            for prop_name in info["range_props"]:
                range_schema_id = info["range_map"].get(prop_name, "")
                range_info = inventory.get_by_schema_id(range_schema_id)
                if range_info:
                    type_hint = range_info.full_path.split(".")[-1]
                else:
                    type_hint = "entity"
                props[prop_name] = {
                    "type": "string",
                    "description": (
                        f"Range property — use the entity id "
                        f"of the linked {type_hint}"
                    ),
                }

            all_prop_names = list(props.keys())
            entity_schemas[info["entity_id"]] = {
                "type": "object",
                "properties": props,
                "required": all_prop_names,
                "additionalProperties": False,
            }

        return {
            "type": "object",
            "properties": entity_schemas,
            "required": list(entity_schemas.keys()),
            "additionalProperties": False,
        }

    def _extract_property_values(
        self,
        entity_info_list: List[dict],
    ) -> Dict[str, Dict[str, str]]:
        """Extract property values for all entities in a single LLM call.

        Uses a dynamic per-entity JSON schema where all fillable
        properties (literal and range) are required.

        Args:
            entity_info_list: List of dicts with keys:
                entity_id, schema_path, description,
                literal_props, range_props, range_map

        Returns:
            Dict mapping entity_id -> {prop_name -> value}.
        """
        # Build entity reference table for the prompt
        ref_lines = []
        for info in entity_info_list:
            short_type = info["schema_path"].split(".")[-1]
            ref_lines.append(
                f"  {info['entity_id']}: {short_type} "
                f"— \"{info['description']}\""
            )
        entity_ref = "\n".join(ref_lines)

        # Build per-entity property listing
        entity_sections = []
        for info in entity_info_list:
            lines = [
                f"Entity {info['entity_id']} "
                f"({info['schema_path']}):",
                f"  Description: {info['description']}",
            ]
            if info["literal_props"]:
                lines.append(
                    f"  Literal properties: "
                    f"{', '.join(info['literal_props'])}"
                )
            if info["range_props"]:
                lines.append("  Range properties (use entity id as value):")
                for rp in info["range_props"]:
                    lines.append(f"    - {rp}")
            entity_sections.append("\n".join(lines))

        entity_listing = "\n\n".join(entity_sections)

        sys_prompt = (
            "You are an expert data extraction assistant. "
            "Extract property values from entity descriptions. "
            "For literal properties, provide the actual value "
            "(string, number, date). "
            "For range properties, provide the entity id of the "
            "linked entity from the plan below. "
            "You MUST fill ALL properties — they are all required. "
            "Do NOT invent information not in the descriptions."
        )

        user_prompt = (
            f"## Available Entities\n{entity_ref}\n\n"
            f"## Entities to Extract\n\n{entity_listing}\n\n"
            f"Fill ALL listed properties for each entity."
        )

        # Build and apply dynamic schema
        schema = self._build_extraction_schema(entity_info_list)
        schema = modify_schema(schema)

        llm = self._get_llm()
        if model_supports_structured_output(llm, tools=[]):
            response_format = ProviderStrategy(schema=schema, strict=True)
        else:
            response_format = ToolStrategy(schema=schema)

        agent = create_agent(
            model=llm, response_format=response_format, tools=[]
        )

        print("\n>> Extracting property values for all entities...")
        t0 = time.time()

        try:
            result = agent.invoke({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            })
        except Exception as e:
            print(f"  Error extracting property values: {e}")
            return {}

        elapsed = time.time() - t0

        if "structured_response" not in result:
            print(f"  Error: No structured_response: {result}")
            return {}

        extracted = result["structured_response"]

        print(
            f"  Extracted values for {len(extracted)} entities "
            f"in {elapsed:.2f}s"
        )
        for eid, props in extracted.items():
            print(f"  [{eid}] {props}")

        return extracted

    # ------------------------------------------------------------------
    # Single entity creation (LLM call with retries)
    # ------------------------------------------------------------------

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

        Args:
            entity_id: Local entity ID (e.g. "entity_1").
            schema_path: Full schema path.
            schema_cls: Schema class for validation.
            filtered_schema: JSON schema filtered to fillable properties.
            entity_uuid: UUID for the entity.
            property_hints: Property values from extraction step.
        """
        sys_prompt = (
            "You are an expert laboratory assistant. "
            "You always answer in valid JSON according to the provided schema. "
            "Fill ONLY the properties listed below with the given values. "
            "Do not invent any new information. "
            "Do not generate dummy or placeholder values. "
            "IMPORTANT: For properties with a 'range' annotation, "
            "provide a COMPLETE textual description of the linked entity "
            "including ALL available details in that single field. "
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
            model=llm,
            response_format=response_format,
            tools=[],
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
                print(f"  Validation error for [{entity_id}]: {e_msg}")
                user_prompt += (
                    f"\n\nThe previous response had an error: {e_msg}"
                )
                if attempt >= max_retries - 1:
                    print(
                        f"  Max retries reached for [{entity_id}], skipping"
                    )
                    return None

        return None

    # ------------------------------------------------------------------
    # Full entity construction pipeline
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """Run full entity construction pipeline.

        1. Segment prompt into entity graph (Variant A)
        2. Load schemas + identify fillable properties (parallel)
        3. Extract property values for all entities (single LLM call)
        4. Create entities in parallel via LLM (with property hints)
        5. Generate OSW-IDs + insert range property links
        6. Construct OswBaseModel instances
        7. Deduplicate against vector store

        Returns dict mapping entity IRI -> OswBaseModel instance.
        """
        self.entities = {}

        # Step 1: Segment with Variant A (simple, reliable)
        segmentation = self.segment(prompt, SegmentationVariant.A)
        if not segmentation:
            print("Error: Segmentation returned no entities")
            return self.entities

        # Assign local IDs (Variant A doesn't have them)
        entity_ids = [f"entity_{i+1}" for i in range(len(segmentation))]

        # Step 2: Load schemas + identify fillable properties (parallel)
        print("\n>> Loading schemas and identifying fillable properties...")
        entity_info_list: List[dict] = []

        # Build entity context string so each _identify_fillable_properties
        # call knows which other entities exist (needed for range properties)
        context_lines = []
        for i, ent in enumerate(segmentation):
            short_type = ent.schema_path.split(".")[-1]
            context_lines.append(
                f"  {entity_ids[i]}: {short_type} ({ent.schema_path}) "
                f"— \"{ent.description}\""
            )
        entity_context = "\n".join(context_lines)

        def _process_entity(idx, entity):
            eid = entity_ids[idx]
            try:
                schema_cls, target_schema = self._load_schema(
                    entity.schema_path
                )
            except Exception as e:
                print(f"  Error loading schema {entity.schema_path}: {e}")
                return None

            fillable = self._identify_fillable_properties(
                entity.description, target_schema, entity_context
            )
            print(f"  [{eid}] {entity.schema_path}: fillable={fillable}")

            range_props = self._extract_range_properties(target_schema)
            literal_props = [p for p in fillable if p not in range_props]
            range_fillable = [p for p in fillable if p in range_props]

            return {
                "idx": idx,
                "entity_id": eid,
                "schema_path": entity.schema_path,
                "description": entity.description,
                "schema_cls": schema_cls,
                "target_schema": target_schema,
                "fillable": fillable,
                "literal_props": literal_props,
                "range_props": range_fillable,
                "range_map": {p: range_props[p] for p in range_fillable},
            }

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_process_entity, i, ent): i
                for i, ent in enumerate(segmentation)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    entity_info_list.append(result)

        entity_info_list.sort(key=lambda x: x["idx"])

        if not entity_info_list:
            print("Error: No entities could be processed")
            return self.entities

        # Step 3: Extract property values (single LLM call, dynamic schema)
        extraction_map = self._extract_property_values(entity_info_list)

        # Step 4: Create entities in parallel (with property hints)
        print(f"\n>> Creating {len(entity_info_list)} entities...")
        t0 = time.time()

        entity_preparations: List[Tuple[dict, dict, uuid.UUID, dict]] = []
        for info in entity_info_list:
            filtered = self._filter_schema(
                info["target_schema"], info["fillable"]
            )
            try:
                filtered = modify_schema(filtered)
            except Exception as e:
                print(
                    f"  Error modifying schema for "
                    f"[{info['entity_id']}]: {e}"
                )
                continue

            hints = extraction_map.get(info["entity_id"], {})

            entity_uuid = uuid.uuid4()
            entity_preparations.append(
                (info, filtered, entity_uuid, hints)
            )

        results: Dict[str, Tuple[dict, Any, uuid.UUID]] = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._create_single_entity,
                    info["entity_id"],
                    info["schema_path"],
                    info["schema_cls"],
                    filtered,
                    uid,
                    hints,
                ): info
                for info, filtered, uid, hints in entity_preparations
            }
            for future in as_completed(futures):
                info = futures[future]
                try:
                    result_dict = future.result()
                except Exception as e:
                    print(f"  Exception for [{info['entity_id']}]: {e}")
                    continue
                if result_dict is not None:
                    uid = next(
                        u for i, _, u, _ in entity_preparations
                        if i["entity_id"] == info["entity_id"]
                    )
                    results[info["entity_id"]] = (
                        result_dict, info["schema_cls"], uid
                    )

        elapsed = time.time() - t0
        print(f"  {len(results)} entities created in {elapsed:.2f}s")

        # Step 5: Generate OSW-IDs + insert range property links
        id_map: Dict[str, Tuple[str, uuid.UUID]] = {}
        for entity_id, (result_dict, schema_cls, uid) in results.items():
            osw_id = "Item:OSW" + uid.hex
            id_map[entity_id] = (osw_id, uid)

        print(f"\n>> Generated IDs for {len(id_map)} entities:")
        for seg_id, (osw_id, _) in id_map.items():
            print(f"  {seg_id} -> {osw_id}")

        print("\n>> Inserting range property links...")
        for info in entity_info_list:
            entity_id = info["entity_id"]
            if entity_id not in results:
                continue
            result_dict, _, _ = results[entity_id]

            entity_values = extraction_map.get(entity_id, {})
            for prop_name in info["range_props"]:
                ref_id = entity_values.get(prop_name, "")
                if ref_id in id_map:
                    osw_id, _ = id_map[ref_id]
                    result_dict[prop_name] = osw_id
                    print(
                        f"  [{entity_id}].{prop_name} -> {osw_id}"
                    )
                elif ref_id:
                    print(
                        f"  Warning: [{entity_id}].{prop_name} "
                        f"references unknown '{ref_id}'"
                    )

        # Step 6: Construct OswBaseModel instances
        print("\n>> Constructing final entities...")
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

        # Step 7: Deduplicate against vector store
        print("\n>> Deduplicating against stored entities...")
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

        print(
            f"\n>> Entity construction complete: "
            f"{len(self.entities)} entities"
        )
        return self.entities

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _merge_entity_data(
        self,
        entity_id: str,
        existing_data: Dict[str, Any],
        existing_type: str,
        data_instance: OswBaseModel,
    ) -> str:
        """Merge existing entity data with new data_instance.

        When an entity in the vector store matches but the new description
        has additional data, this method merges them using the appropriate
        schema from the cache.

        Returns the entity_id (entity is updated in entities dict).
        """
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
                print(f"   Using fallback schema: {schema_cls.__name__}")

            merged_entity = schema_cls(**merged_data)
            self.entities[entity_id] = merged_entity
            print(f"   Entity {entity_id} stored with merged data")

        except Exception as e:
            print(f"   Error creating merged entity: {e}")
            self.entities[entity_id] = data_instance
            print(f"   Fallback: stored new instance for {entity_id}")

        return entity_id

    def _remap_range_references(self, iri_remap: Dict[str, str]):
        """Remap range property references after deduplication.

        After dedup, some entities' properties may reference old IRIs
        that have been remapped to existing entity IRIs.
        """
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
                print(f"    literals: {e.literal_dict()}")
                print(f"    ranges:   {e.range_dict()}")
            elif isinstance(e, EntityPropertyDescription):
                print(f"  {e.schema_path}: {e.property_list}")
            else:
                print(f"  {e.schema_path}: {e.description}")

def invoke_test(test_prompt: str):
    """Quick test of full entity construction."""
    agent = SegmentationAgent()
    entities = agent.invoke(test_prompt)
    print(f"\nConstructed {len(entities)} entities:")
    for iri, entity in entities.items():
        print(f"  {iri}: {entity.json(exclude_none=True)}")

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

    #segmentation_test(test_prompt)
    invoke_test(test_prompt)