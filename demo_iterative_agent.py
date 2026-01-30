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
)

import json

from llm_init import get_llm, model_supports_structured_output
from schema_catalog import lookup_exact_schema
from osl_init import (
    build_vector_store,
    lookup_excact_matching_entity,
)

target_data_model = LaboratoryProcess

# prompt = (
#     "Create a laboratory process entry."
#     "Author is Dr. John Doe, working at the Example Lab Corp."
#     "starting at 01.01.2025 and ending at 02.01.2025."
#     "Status is in progress."
# )

sys_prompt = (
    "You are an expert laboratory assistant. "
    "You always answer in valid JSON according to the provided schema. "
    "First ask yourself 'Which properties can I actually fill "
    " with the given information without inventing data?'. "
    "If you do not have enough information to fill a field, "
    "leave it empty or null. "
    "Do not invent any new information that is not provided in the prompt. "
    "Do not add any additional properties that are not defined in the schema. "
    "If you encounter a property that is annotated in the schema with "
    "a 'range', "
    "e.g. \"range\": \"Category:OSW44deaa5b806d41a2a88594f562b110e9\" "
    "and you have actual information about the entity that could be "
    "linked there, "
    "use the tool 'create_linked_entity' to "
    "create the referenced entity "
    "if (and only if) you have enough information. "
    "Provide the name of the property that has the range, "
    "the schema ID and name (if known)stored in the range, "
    "the ID of the parent entity you are completing, "
    "and all available information about the entity."
    "Store the returned ID (or an empty or null value) in the "
    "corresponding property."
    "Store empty or null value directly without calling the tool "
    "if you do not have enough information to create the entity. "
    "Do not invent IDs on your own. "
    "Do not create any placeholders or dummy values. "
    "Do not call the tool twice with the same parameters."
    "Do not use the tool for the root entity you are creating, only "
    "for linked entities."
)


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


entitites = {}
entitity_requests = {}
root = True
vector_store = build_vector_store()


def create_linked_entity(param: CreateParam) -> str | None:
    """Create a linked entity stored in a property with 'range' annotation
    based on the provided description
    containing all available information, or create it if not found.
    Returns the entity's OSW-ID to store in the corresponding property
    of the parent object or None if the entity could not be created.
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
    entitity_requests[entity_id] = param

    prompt = ""
    if param.schema_id != "":
        prompt = "The schema id is " + param.schema_id + ". "
    if param.schema_name != "":
        prompt += "The schema name is " + param.schema_name + ". "
    if param.entity_description != "":
        prompt += "The entity I want to describe: "
        prompt += param.entity_description + ". "

    LOOKUP_FIRST = False
    if LOOKUP_FIRST:
        existing_entity = lookup_excact_matching_entity(
            vector_store=vector_store,
            description=prompt,
            llm_judge=True
        )
        if existing_entity is not None:
            print(f"Found existing entity match: {existing_entity}")
            return existing_entity

    schema_name = lookup_exact_schema(prompt)
    print(f"LLM returned class path: {schema_name}")

    # get the class from the path
    schema_cls: OswBaseModel = eval(schema_name)

    if schema_cls is None:
        # raise ValueError(
        # f"Schema name {param.schema_name} not found in opensemantic modules"
        # )
        print(
          f"Schema name {param.schema_name} not found in opensemantic modules"
        )
        return None
    else:
        print(f"Found schema class for {param.schema_name}: {schema_cls}")

    # create a structured output agent with a provider strategy
    # based on the target data model's schema
    # preprocess the schema to comply
    # with https://platform.openai.com/docs/guides/structured-outputs#supported-schemas  # noqa: E501

    try:
        # Fixme: schema contains "definitions" instead of "$defs"
        # but $refs are pointing to ""
        # target_schema = schema_cls
        target_schema = schema_cls.export_schema()
        target_schema = modify_schema(target_schema)
    except Exception as e:
        print(f"Error exporting schema for {param.schema_name}: {e}")
        return None

    model = get_llm()

    if model_supports_structured_output(model, tools=[create_linked_entity]):
        # Model supports provider strategy - use it
        effective_response_format = ProviderStrategy(
            schema=target_schema,
            strict=True
        )
    else:
        # Model doesn't support provider strategy - use ToolStrategy
        effective_response_format = ToolStrategy(
            schema=target_schema
        )

    if hasattr(model, "max_retries"):
        model.max_retries = 1
    # model.temperature = 0.0 # not allowed for reasoning models

    agent = create_agent(
        model=model,
        response_format=effective_response_format,
        tools=[
            create_linked_entity
        ],
    )

    # prompt while providing previous requests (entitity_requests)
    # to avoid duplicates
    user_prompt = (
        "For the parent entity with OSW-ID 'Item:OSW" + entity_uuid.hex + "'"
        " create a JSON Document "
        "based on the following description:"
        + ". \n------Description-------\n"
        + param.entity_description
        + ". \n-------------\n"
        "In case you find a similar request below, you are obliged to reuse"
        " its OSW-ID, e.g. Item:OSWabcdef1234567890 directly without calling "
        "'create_linked_entity. Store this OSW-ID directly in the "
        "corresponding property.'\n"
        "------Previous requests-------\n"
        + "\n".join(
            f"OSW-ID: {osw_id} - Request for property '{r.property_name}', "
            f"schema '{r.schema_id}, {r.schema_name}': {r.entity_description}"
            for osw_id, r in entitity_requests.items()
        )
        + "\n-------------\n\n"
        "In case no similar request is found, create a new entity by "
        "returning a JSON object according to the following schema:\n"
        + json.dumps(target_schema, indent=2)
    )

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:

        print(
            f"Invoking agent for entity creation with "
            f"prompt:\n{user_prompt}"
        )

        try:
            result = agent.invoke({
                "messages": [
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            })
        except Exception as e:
            print(f"Error invoking agent: {e}")
            print(f"  Prompt was:\n{user_prompt}")
            print(f"  Schema was:\n{target_schema}")
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

        # create an instance of the target data model from the result
        try:
            data_instance: OswBaseModel = schema_cls(**result)
            break
        except Exception as e:
            print(f"Error creating data instance: {e}")
            user_prompt += (
                f"\n\nThe previous response could not be parsed correctly: {e}"
            )
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... ({retry_count}/{max_retries})")
            else:
                print("Max retries reached, aborting.")
                return None

    if not LOOKUP_FIRST:
        existing_entity = lookup_excact_matching_entity(
            vector_store=vector_store,
            description=data_instance.json(),
            llm_judge=True
        )
        if existing_entity is not None:
            print(f"Found existing entity match: {existing_entity}")
            return existing_entity

    entitites[data_instance.get_iri()] = data_instance
    print(f">> RETURN: {data_instance.get_iri()}")
    return data_instance.get_iri()


result = create_linked_entity(
    CreateParam(
        parent_id="_root_",
        property_name="_",
        schema_id="LaboratoryProcess",
        schema_name="LaboratoryProcess",
        entity_description=(
            "A laboratory process to document an experiment "
            "created by Dr. Jane Doe, Example Lab, "
            "starting at 05.02.2025 and ending at 06.02.2025, "
            "status in finished."
        )
    )
)

print("Created / Looked up entities:")
for i, e in entitites.items():
    e: OswBaseModel
    print(f"#### {i} ({e.name}) ####")
    print(e.json(indent=2, exclude_none=True))


# generate a short random id prefix
id_prefix = uuid.uuid4().hex[:6]

# prefix all entity names with the id_prefix to avoid name collisions
for i, e in entitites.items():
    e: Entity
    if e.name is not None:
        e.name = f"{id_prefix}_{e.name}"
    if e.label is not None:
        for lb in e.label:
            lb.text = f"{id_prefix} {lb.text}"

osl_client = get_osl_client()
osl_client.store_entity(OSW.StoreEntityParam(
    entities=list(entitites.values()),
    overwrite=True,
    change_id="demo_iterative_agent-0001",
))
