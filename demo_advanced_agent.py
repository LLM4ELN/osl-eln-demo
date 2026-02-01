import uuid
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

# note: pydantic v2 models currently cannot be initialized from JSON directly
# + osw-python requires v1 models for now
# => we use v1 models here
# see also https://www.jujens.eu/posts/en/2025/Apr/26/pydantic-enums/
from opensemantic.v1 import OswBaseModel
from opensemantic.core.v1 import Entity
from opensemantic.lab.v1 import LaboratoryProcess
import opensemantic.core.v1  # noqa: F401 needed for eval
import opensemantic.base.v1  # noqa: F401 needed for eval
import opensemantic.lab.v1  # noqa: F401 needed for eval
from pydantic import BaseModel, Field

from util import (
    modify_schema,
    post_process_llm_json_response,
    deep_copy,
)

import json

from llm_init import get_llm, model_supports_structured_output
from schema_catalog import lookup_exact_schema
from osw.core import OSW
from osl_init import (
    build_vector_store,
    get_osl_client,
    lookup_excact_matching_entity,
)

target_data_model = LaboratoryProcess


class CreateParam(BaseModel):
    parent_id: str = Field(..., min_length=1)
    """OSW-ID, e.g. Item:OSWabcdef1234567890 of the parent entity
    which is completed. Empty if not applicable.
    """
    property_name: str = Field(..., min_length=1)
    """name of the property for which the entity is being looked up / created.
    Empty if not applicable.
    """
    schema_id: str = Field(..., min_length=1)
    """ID of the schema to use for creation of the entity
    e.g. 'Category:OSWabcdef1234567890'"""
    schema_name: str
    """name of the schema to use for creation of the entity"""
    entity_description: str = Field(..., min_length=1)
    """textual description of the entity to create
    containing all available information
    """


class ComparisonResult(BaseModel):
    """Result of comparing two entity requests"""
    is_same_entity: bool = Field(
        ...,
        description=(
            "True if both requests describe the same entity, "
            "False otherwise"
        )
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why they are the same or different"
    )


entities = {}
entity_requests = {}
vector_store = build_vector_store()


def identify_fillable_properties(
    entity_description: str,
    schema: dict,
    llm
) -> list[str]:
    """Step 3: Ask LLM to identify which properties can actually be filled
    based on the provided description without inventing data.
    Returns list of property names that can be filled.
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

    response = llm.invoke(prompt)
    try:
        # Extract JSON array from response
        content = (
            response.content if hasattr(response, 'content')
            else str(response)
        )
        # Find JSON array in content
        import re
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            fillable = json.loads(json_match.group())
            print(f"Fillable properties: {fillable}")
            return fillable
        else:
            print("Warning: Could not parse fillable properties, using all")
            return list(properties.keys())
    except Exception as e:
        print(
            f"Error parsing fillable properties: {e}, "
            f"using all properties"
        )
        return list(properties.keys())


def filter_schema_properties(
    schema: dict, fillable_properties: list[str]
) -> dict:
    """Step 4: Remove properties that cannot be filled from the schema.
    Always keeps required properties even if not in fillable list."""
    print(
        "\n>> Filtering schema to only include fillable properties..."
    )

    filtered_schema = deep_copy(schema)

    if "properties" in filtered_schema:
        required_props = filtered_schema.get("required", [])

        # Keep fillable properties AND required properties
        properties_to_keep = set(fillable_properties) | set(required_props)

        filtered_schema["properties"] = {
            prop: props
            for prop, props in filtered_schema["properties"].items()
            if prop in properties_to_keep
        }

        # Keep all required fields unchanged
        # (they are required by the schema)

    num_props = len(filtered_schema.get('properties', {}))
    print(f"Filtered schema has {num_props} properties")
    return filtered_schema


def extract_range_properties(schema: dict) -> dict:
    """Extract properties that have a 'range' annotation.
    Returns dict mapping property_name -> range_schema_id
    """
    range_props = {}
    properties = schema.get("properties", {})

    for prop_name, prop_schema in properties.items():
        if "range" in prop_schema:
            range_props[prop_name] = prop_schema["range"]

    return range_props


def compare_with_previous_requests(param: CreateParam) -> str | None:
    """Step 2: Compare the request with previous requests stored in global log.
    Returns entity ID if match found, None otherwise.
    """
    print("\n>> Comparing with previous requests...")

    for entity_id, prev_request in entity_requests.items():
        # Simple heuristic: if schema and description are very similar, reuse
        if (prev_request.schema_id == param.schema_id and
                prev_request.entity_description.strip().lower() ==
                param.entity_description.strip().lower()):
            print(f"Found matching previous request: {entity_id}")
            return entity_id

    print("No matching previous request found")
    return None


def create_linked_entity(param: CreateParam) -> str | None:
    """Advanced approach: Create a linked entity with property filtering
    and post-processing of range properties.

    Steps:
    1. Lookup schema
    2. Compare with previous requests (early comparison)
    3. Identify fillable properties
    4. Filter schema
    5. Create entity with structured output (range props as strings)
    6. Post-process range properties
    7. Compare with existing entities
    8. Store and return
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

    # Step 2: Early comparison with previous requests
    existing_from_log = compare_with_previous_requests(param)
    if existing_from_log is not None:
        return existing_from_log

    # Store this request
    entity_requests[entity_id] = param

    # Step 1: Lookup schema
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

    # Export the schema (without modification yet)
    try:
        target_schema = schema_cls.export_schema()
    except Exception as e:
        print(f"Error exporting schema for {param.schema_name}: {e}")
        return None

    model = get_llm()

    # Step 3: Identify fillable properties
    fillable_properties = identify_fillable_properties(
        param.entity_description,
        target_schema,
        model
    )

    # Step 4: Filter schema to only fillable properties
    filtered_schema = filter_schema_properties(
        target_schema, fillable_properties
    )

    # Modify schema after filtering
    try:
        filtered_schema = modify_schema(filtered_schema)
    except Exception as e:
        print(f"Error modifying filtered schema: {e}")
        return None

    # Extract range properties before creating entity
    range_properties = extract_range_properties(filtered_schema)
    print(f"Range properties to process: {range_properties}")

    # Step 5: Create entity with structured output
    # Range properties will be filled as strings with descriptions
    sys_prompt = (
        "You are an expert laboratory assistant. "
        "You always answer in valid JSON according to the provided schema. "
        "Fill ONLY the properties that you have actual information for. "
        "Do not invent any new information that is not provided in "
        "the prompt. "
        "For properties with a 'range' annotation, provide a textual "
        "description of the linked entity (if you have information), "
        "not an ID. "
        "If you do not have enough information for a field, leave it "
        "empty or null. "
    )

    if model_supports_structured_output(model, tools=[]):
        effective_response_format = ProviderStrategy(
            schema=filtered_schema,
            strict=True
        )
    else:
        effective_response_format = ToolStrategy(
            schema=filtered_schema
        )

    if hasattr(model, "max_retries"):
        model.max_retries = 1

    agent = create_agent(
        model=model,
        response_format=effective_response_format,
        tools=[],  # No tools for this step
    )

    user_prompt = (
        f"Create a JSON document based on the following description:\n"
        f"{param.entity_description}\n\n"
        f"Use the following schema:\n{json.dumps(filtered_schema, indent=2)}"
    )

    max_retries = 3
    retry_count = 0
    result = None

    while retry_count < max_retries:
        attempt_num = retry_count + 1
        print(f"Invoking agent for entity creation (attempt {attempt_num})...")

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

        result = post_process_llm_json_response(result["structured_response"])
        result["uuid"] = str(entity_uuid)

        print(f"Structured Response for {param.schema_name}:")
        print(json.dumps(result, indent=2))

        # Try to create instance to validate
        try:
            # Don't create final instance yet - we need to process
            # range properties
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
    # For each range property, recursively lookup/create the linked entity
    print("\n>> Post-processing range properties...")
    for prop_name, range_schema_id in range_properties.items():
        if prop_name in result and result[prop_name]:
            description = result[prop_name]

            # Skip if already an ID
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

            # Recursively create/lookup the linked entity
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

            linked_entity_id = create_linked_entity(linked_param)

            if linked_entity_id:
                result[prop_name] = linked_entity_id
                print(
                    f"Replaced '{prop_name}' description with "
                    f"ID: {linked_entity_id}"
                )
            else:
                # Could not create linked entity, remove the property
                result.pop(prop_name, None)
                print(
                    f"Could not create linked entity for '{prop_name}', "
                    f"removing property"
                )

    # Now create the actual data instance
    try:
        data_instance: OswBaseModel = schema_cls(**result)
    except Exception as e:
        print(f"Error creating data instance after range processing: {e}")
        return None

    # Step 7: Compare with existing entities
    print("\n>> Comparing with existing entities in database...")
    existing_entity = lookup_excact_matching_entity(
        vector_store=vector_store,
        description=data_instance.json(),
        llm_judge=True
    )
    if existing_entity is not None:
        print(f"Found existing entity match: {existing_entity}")
        return existing_entity

    # Step 8: Store and return
    entities[data_instance.get_iri()] = data_instance
    print(f">> RETURN: {data_instance.get_iri()}")
    return data_instance.get_iri()


# Main execution
result = create_linked_entity(
    CreateParam(
        parent_id="_root_",
        property_name="_",
        schema_id="LaboratoryProcess",
        schema_name="LaboratoryProcess",
        entity_description=(
            "A laboratory process to document an experiment "
            "created by Dr. Jane Doe, Example Lab Corp., "
            "starting at 05.03.2025 and ending at 06.03.2025, "
            "status in finished."
        )
    )
)

print("\n\n=== Created / Looked up entities ===")
for i, e in entities.items():
    e: OswBaseModel
    print(f"#### {i} ({e.name}) ####")
    print(e.json(indent=2, exclude_none=True))


# Generate a short random id prefix
id_prefix = uuid.uuid4().hex[:6]

# Prefix all entity names with the id_prefix to avoid name collisions
for i, e in entities.items():
    e: Entity
    if e.name is not None:
        e.name = f"{id_prefix}_{e.name}"
    if e.label is not None:
        for lb in e.label:
            lb.text = f"{id_prefix} {lb.text}"

STORE = False
# Uncomment to store entities in OSL
# STORE = True
if STORE:
    print("\n\n=== Storing entities in OSL ===")
    osl_client = get_osl_client()
    osl_client.store_entity(OSW.StoreEntityParam(
        entities=list(entities.values()),
        overwrite=True,
        change_id="demo_advanced_agent-0002",
    ))
