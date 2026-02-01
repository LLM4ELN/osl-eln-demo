"""OoldAgent - Reusable agent for creating and managing OpenSemantic
entities."""

import json
import re
import uuid
from typing import Dict, Any, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from opensemantic.v1 import OswBaseModel
import opensemantic.core.v1  # noqa: F401 needed for eval
import opensemantic.base.v1  # noqa: F401 needed for eval
import opensemantic.lab.v1  # noqa: F401 needed for eval

from rag_init import get_vector_store
from util import (
    modify_schema,
    post_process_llm_json_response,
    deep_copy,
    remove_empty,
    remove_nulls,
)
from llm_init import get_llm, model_supports_structured_output
from schema_catalog import lookup_exact_schema, get_cached_inventory
from osl_init import lookup_excact_matching_entity


class CreateParam(BaseModel):
    """Parameters for creating or looking up an entity."""
    parent_id: str = "_root_"
    """OSW-ID, e.g. Item:OSWabcdef1234567890 of the parent entity
    which is completed. '_root_' if not applicable.
    """
    property_name: str = "_"
    """name of the property for which the entity is being looked up / created.
    '_' if not applicable.
    """
    schema_id: str = ""
    """ID of the schema to use for creation of the entity
    e.g. 'Category:OSWabcdef1234567890'. Empty to auto-detect.
    """
    schema_name: str = ""
    """name of the schema to use for creation of the entity.
    Empty to auto-detect."""
    entity_description: str = Field(..., min_length=1)
    """textual description of the entity to create
    containing all available information
    """


class ComparisonResult(BaseModel):
    """Result of comparing a new request against multiple previous requests."""
    matching_entity_id: str = Field(
        default="",
        description=(
            "Entity ID if match found (e.g. 'Item:OSW...'), "
            "empty string if no match"
        )
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the decision"
    )


class OoldAgent(BaseModel):
    """Agent for creating and managing OpenSemantic Lab entities.

    This agent handles:
    - Entity creation with structured output
    - Schema lookup and property filtering
    - Deduplication via previous request comparison
    - Deduplication via vector store lookup
    - Recursive creation of linked entities
    """

    vector_store: Optional[Any] = None
    """Vector store for entity lookup.
    If None, builds one on first use."""
    llm: Optional[Any] = None
    """Language model to use.
    If None, uses default from get_llm() on first use."""
    use_llm_judge: bool = True
    """Whether to use LLM judge for entity comparison."""
    entities: Dict[str, Any] = Field(default_factory=dict)
    """Created entities, keyed by IRI."""
    entity_requests: Dict[str, CreateParam] = Field(default_factory=dict)
    """Previous entity creation requests, keyed by entity ID."""

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __init__(self, **data):
        super().__init__(**data)
        # Lazy initialization of vector_store
        if self.vector_store is None:
            object.__setattr__(self, 'vector_store', get_vector_store())

    def _get_llm(self):
        """Get the LLM, creating one if needed."""
        if self.llm is None:
            object.__setattr__(self, 'llm', get_llm())
        return self.llm

    def _schemas_are_compatible(
        self, schema_id_1: str, schema_id_2: str
    ) -> bool:
        """Check if two schema IDs are compatible (same or subclass relation).

        Uses schema_id (e.g., 'Category:OSWxxx') not full paths.
        """
        if schema_id_1 == schema_id_2:
            return True

        inventory = get_cached_inventory()

        if inventory.is_subclass_of_by_schema_id(schema_id_1, schema_id_2):
            return True
        if inventory.is_subclass_of_by_schema_id(schema_id_2, schema_id_1):
            return True

        return False

    def _identify_fillable_properties(
        self,
        entity_description: str,
        schema: dict
    ) -> list[str]:
        """Ask LLM to identify which properties can be filled from
        description.

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

        prompt = f"""Given the following entity description:
"{entity_description}"

And the following schema properties:
{json.dumps(property_descriptions, indent=2)}

List ONLY the property names that can be filled with actual information
from the description.
Do not include properties where you would need to invent or
hallucinate data.
Return your answer as a JSON array of property names,
e.g.: ["property1", "property2"]
"""

        response = self._get_llm().invoke(prompt)
        try:
            content = (
                response.content if hasattr(response, 'content')
                else str(response)
            )
            json_match = re.search(
                r'\[.*?\]', content, re.DOTALL
            )
            if json_match:
                fillable = json.loads(json_match.group())
                print(f"Fillable properties: {fillable}")
                return fillable
            else:
                print(
                    "Warning: Could not parse fillable properties, "
                    "using all"
                )
                return list(properties.keys())
        except Exception as e:
            print(f"Error parsing fillable properties: {e}, using all")
            return list(properties.keys())

    def _filter_schema_properties(
        self,
        schema: dict,
        fillable_properties: list[str]
    ) -> dict:
        """Remove properties that cannot be filled from the schema.

        Always keeps required properties even if not in fillable list.
        """
        print("\n>> Filtering schema to only include fillable properties...")

        filtered_schema = deep_copy(schema)

        if "properties" in filtered_schema:
            required_props = filtered_schema.get("required", [])
            properties_to_keep = set(fillable_properties) | set(required_props)

            filtered_schema["properties"] = {
                prop: props
                for prop, props in filtered_schema["properties"].items()
                if prop in properties_to_keep
            }

        num_props = len(filtered_schema.get('properties', {}))
        print(f"Filtered schema has {num_props} properties")
        return filtered_schema

    def _extract_range_properties(self, schema: dict) -> dict:
        """Extract properties that have a 'range' annotation.

        Returns dict mapping property_name -> range_schema_id
        """
        range_props = {}
        properties = schema.get("properties", {})

        for prop_name, prop_schema in properties.items():
            if "range" in prop_schema:
                range_props[prop_name] = prop_schema["range"]

        return range_props

    def _compare_with_previous_requests(
        self,
        param: CreateParam
    ) -> str | None:
        """Compare request with previous requests using LLM judge.

        Returns entity ID if match found, None otherwise.
        """
        print("\n>> Comparing with previous requests using LLM judge...")

        if not self.entity_requests:
            print("No previous requests to compare")
            return None

        # Filter to compatible schemas (same or subclass relation)
        relevant_requests = {
            entity_id: req
            for entity_id, req in self.entity_requests.items()
            if self._schemas_are_compatible(req.schema_id, param.schema_id)
        }

        if not relevant_requests:
            print("No previous requests with compatible schema")
            return None

        comparison_schema = ComparisonResult.model_json_schema()

        if model_supports_structured_output(self._get_llm(), tools=[]):
            judge_response_format = ProviderStrategy(
                schema=comparison_schema,
                strict=True
            )
        else:
            judge_response_format = ToolStrategy(
                schema=comparison_schema
            )

        judge_agent = create_agent(
            model=self._get_llm(),
            response_format=judge_response_format,
            tools=[],
        )

        request_items = [
            f"- Entity ID: {entity_id}\n"
            f"  Schema: {req.schema_id}\n"
            f"  Description: {req.entity_description}"
            for entity_id, req in relevant_requests.items()
        ]
        previous_requests_text = "\n".join(request_items)

        prompt = f"""You need to determine if the following NEW request
describes the same entity as any of the PREVIOUS requests.

NEW REQUEST:
- Schema: {param.schema_id}
- Description: {param.entity_description}

PREVIOUS REQUESTS:
{previous_requests_text}

Determine if the NEW request describes the SAME entity as any of the
previous requests.
Consider entities the same if they refer to the same person, organization,
or object, even if worded differently.
If a match is found, return the matching_entity_id
(e.g., "Item:OSW...").
If no match is found, return an empty string for matching_entity_id.
"""

        try:
            result = judge_agent.invoke({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You compare entity descriptions to determine "
                            "if they refer to the same entity."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            if "structured_response" in result:
                comparison = ComparisonResult(**result["structured_response"])
                print(f"LLM judge result: {comparison.reasoning}")

                if (comparison.matching_entity_id and
                        comparison.matching_entity_id.strip()):
                    matching_id = comparison.matching_entity_id.strip()
                    if matching_id in relevant_requests:
                        print(
                            f"Found matching previous request: "
                            f"{matching_id}"
                        )
                        return matching_id
                    else:
                        print(
                            f"Warning: LLM returned ID not in list: "
                            f"{matching_id}"
                        )
            else:
                print("Warning: No structured_response in judge result")

        except Exception as e:
            print(f"Error in LLM judge comparison: {e}")

        print("No matching previous request found")
        return None

    def invoke(
        self,
        prompt: str,
        schema_id: str = "",
        schema_name: str = ""
    ) -> str | None:
        """Create an entity from a natural language description.

        This is the main public interface for creating entities.

        Args:
            prompt: Natural language description of the entity to
            create
            schema_id: Optional schema ID
            (e.g., 'Category:OSWxxx'). Auto-detected if empty.
            schema_name: Optional schema name. Auto-detected if
            empty.

        Returns:
            Entity ID (e.g., 'Item:OSWxxx') or None if creation failed
        """
        param = CreateParam(
            parent_id="_root_",
            property_name="_",
            schema_id=schema_id,
            schema_name=schema_name,
            entity_description=prompt
        )
        return self._create_linked_entity(param)

    def _create_linked_entity(self, param: CreateParam) -> str | None:
        """Create a linked entity with property filtering and deduplication.

        Internal method - use invoke() for the public interface.

        Steps:
        1. Compare with previous requests (early deduplication)
        2. Lookup schema
        3. Identify fillable properties
        4. Filter schema
        5. Create entity with structured output
        6. Post-process range properties (recursive creation)
        7. Compare with existing entities in database
        8. Store and return

        Args:
            param: CreateParam with entity details

        Returns:
            Entity ID (e.g., 'Item:OSWxxx') or None if creation failed
        """
        print((
            f"\n\n>> Lookup or create entity for "
            f"parent entity '{param.parent_id}', "
            f"property '{param.property_name}', "
            f"range '{param.schema_id}, {param.schema_name}' "
            f"based on description {param.entity_description}"
        ))

        entity_uuid = uuid.uuid4()
        entity_id = "Item:OSW" + entity_uuid.hex

        # Step 1: Early comparison with previous requests
        existing_from_log = self._compare_with_previous_requests(param)
        if existing_from_log is not None:
            return existing_from_log

        # Store this request
        self.entity_requests[entity_id] = param

        # Step 2: Lookup schema
        prompt = ""
        if param.schema_id != "":
            prompt = "The schema id is " + param.schema_id + ". "
        if param.schema_name != "":
            prompt += "The schema name is " + param.schema_name + ". "
        if param.entity_description != "":
            prompt += "The entity I want to describe: "
            prompt += param.entity_description + ". "

        schema_name = lookup_exact_schema(prompt)
        print(f"LLM returned class path: {schema_name}")

        # Get the class from the path
        schema_cls: OswBaseModel = eval(schema_name)

        if schema_cls is None:
            print(
                f"Schema name {param.schema_name} not found in "
                f"opensemantic modules"
            )
            return None
        else:
            print(f"Found schema class for {param.schema_name}: {schema_cls}")

        # Export the schema
        try:
            target_schema = schema_cls.export_schema()
        except Exception as e:
            print(f"Error exporting schema for {param.schema_name}: {e}")
            return None

        # Step 3: Identify fillable properties
        fillable_properties = self._identify_fillable_properties(
            param.entity_description,
            target_schema
        )

        # Step 4: Filter schema to only fillable properties
        filtered_schema = self._filter_schema_properties(
            target_schema, fillable_properties
        )

        # Modify schema after filtering
        try:
            filtered_schema = modify_schema(filtered_schema)
        except Exception as e:
            print(f"Error modifying filtered schema: {e}")
            return None

        # Extract range properties before creating entity
        range_properties = self._extract_range_properties(filtered_schema)
        print(f"Range properties to process: {range_properties}")

        # Step 5: Create entity with structured output
        sys_prompt = (
            "You are an expert laboratory assistant. "
            "You always answer in valid JSON according to the "
            "provided schema. "
            "Fill ONLY the properties that you have actual "
            "information for. "
            "Do not invent any new information that is not provided in "
            "the prompt. "
            "Do not generate any dummy or placeholder values. "
            "For properties with a 'range' annotation, provide a textual "
            "description of the linked entity (if you have information), "
            "not an ID. "
            "If you do not have enough information for a field, leave it "
            "empty or null. "
        )

        if model_supports_structured_output(self._get_llm(), tools=[]):
            effective_response_format = ProviderStrategy(
                schema=filtered_schema,
                strict=True
            )
        else:
            effective_response_format = ToolStrategy(
                schema=filtered_schema
            )

        llm = self._get_llm()
        if hasattr(llm, "max_retries"):
            llm.max_retries = 1

        agent = create_agent(
            model=self._get_llm(),
            response_format=effective_response_format,
            tools=[],
        )

        schema_str = json.dumps(filtered_schema, indent=2)
        user_prompt = (
            f"Create a JSON document based on the following "
            f"description:\n"
            f"{param.entity_description}\n\n"
            f"Use the following schema:\n{schema_str}"
        )

        max_retries = 3
        retry_count = 0
        result = None

        while retry_count < max_retries:
            attempt_num = retry_count + 1
            print(
                f"Invoking agent for entity creation "
                f"(attempt {attempt_num})..."
            )

            try:
                result = agent.invoke({
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                })
            except Exception as e:
                print(f"Error invoking agent: {e}")
                return None

            if "structured_response" not in result:
                print(
                    f"Error: Agent result does not contain "
                    f"structured_response: {result}"
                )
                return None

            result = post_process_llm_json_response(
                result["structured_response"]
            )
            result["uuid"] = str(entity_uuid)

            print(f"Structured Response for {param.schema_name}:")
            print(json.dumps(result, indent=2))

            try:
                break
            except Exception as e:
                print(f"Error in response format: {e}")
                user_prompt += f"\n\nThe previous response had an error: {e}"
                retry_count += 1
                if retry_count >= max_retries:
                    print("Max retries reached, aborting.")
                    return None

        if result is None:
            return None

        # Step 6: Post-process range properties
        print("\n>> Post-processing range properties...")
        for prop_name, range_schema_id in range_properties.items():
            if prop_name in result and result[prop_name]:
                description = result[prop_name]

                is_already_id = (
                    isinstance(description, str) and
                    description.startswith("Item:OSW")
                )
                if is_already_id:
                    continue

                print(
                    f"\n>> Processing range property '{prop_name}' "
                    f"with description: {description}"
                )

                schema_name = (
                    range_schema_id.split(":")[-1]
                    if ":" in range_schema_id
                    else range_schema_id
                )
                linked_param = CreateParam(
                    parent_id=entity_id,
                    property_name=prop_name,
                    schema_id=range_schema_id,
                    schema_name=schema_name,
                    entity_description=str(description)
                )

                # Recursive call
                linked_entity_id = self._create_linked_entity(linked_param)

                if linked_entity_id:
                    result[prop_name] = linked_entity_id
                    print(
                        f"Replaced '{prop_name}' description with "
                        f"ID: {linked_entity_id}"
                    )
                else:
                    result.pop(prop_name, None)
                    print(
                        f"Could not create linked entity for '{prop_name}', "
                        f"removing property"
                    )

        # Create the actual data instance
        try:
            data_instance: OswBaseModel = schema_cls(**result)
        except Exception as e:
            print(f"Error creating data instance after range processing: {e}")
            return None

        # Step 7: Compare with existing entities in database
        print("\n>> Comparing with existing entities in database...")
        data_instance_dict = json.loads(data_instance.json())
        data_instance_dict.pop("uuid", None)
        data_instance_dict = remove_empty(data_instance_dict)
        data_instance_dict = remove_nulls(data_instance_dict)
        data_instance_description = json.dumps(data_instance_dict)

        existing_entity = lookup_excact_matching_entity(
            vector_store=self.vector_store,
            description=data_instance_description,
            llm_judge=self.use_llm_judge
        )
        if existing_entity is not None:
            print(f"Found existing entity match: {existing_entity}")
            return existing_entity

        # Step 8: Store and return
        self.entities[data_instance.get_iri()] = data_instance
        print(f">> RETURN: {data_instance.get_iri()}")
        return data_instance.get_iri()

    def get_entities(self) -> Dict[str, OswBaseModel]:
        """Get all created entities."""
        return self.entities
    
    def store_entities(self):
        """Store all created entities in vector store."""
        for entity_id, entity in self.entities.items():
            # Serialize entity to JSON
            jsondata = json.loads(entity.json(exclude_none=True))

            # Create langchain document
            doc = Document(
                id=entity_id,
                page_content=json.dumps({"jsondata": jsondata}),
                metadata={
                    "name": (
                        jsondata.get("name", "")
                    ),
                    "type": jsondata.get("type", [""])[0] if isinstance(
                        jsondata.get("type"), list
                    ) else jsondata.get("type", "")
                }
            )

            # Add to vector store
            self.vector_store.add_documents(documents=[doc])
            print(f"Stored entity {entity_id} in vector store")

    def clear(self):
        """Clear all stored entities and requests."""
        self.entities.clear()
        self.entity_requests.clear()
