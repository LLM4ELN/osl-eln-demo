import uuid
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

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
from pydantic import BaseModel

from util import (
    modify_schema,
    post_process_llm_json_response,
)

import json

from llm_init import get_llm
from schema_catalog import lookup_exact_schema
from osw.core import OSW
from osl_init import (
    build_vector_store,
    get_osl_client,
    lookup_excact_matching_entity,
)

target_data_model = LaboratoryProcess

prompt = (
    "Create a laboratory process entry."
    "Author is Dr. John Doe, working at the Example Lab."
    "starting at 01.01.2025 and ending at 02.01.2025."
    "Status is in progress."
)

sys_prompt = (
    "You are an expert laboratory assistant. "
    "You always answer in valid JSON according to the provided schema. "
    "If you do not have enough information to fill a field, "
    "leave it empty or null. "
    "Do not add any additional fields that are not defined in the schema. "
    "If you encounter a property that is annotated with a 'range', "
    "use the tool 'lookup_or_create_entity' to "
    "lookup or create the referenced entity."
    "Store the returned ID (if not None) in the corresponding fields."
)


class LookupOrCreateParam(BaseModel):
    parent_id: str
    """OSW-ID, e.g. Item:OSWabcdef1234567890 of the parent entity
    which is completed. Empty if not applicable.
    """
    property_name: str
    """name of the property for which the entity is being looked up / created.
    Empty if not applicable.
    """
    schema_id: str
    """ID of the schema to use for lookup / creation of the entity
    e.g. 'Category:OSWabcdef1234567890'"""
    schema_name: str
    """name of the schema to use for lookup / creation of the entity"""
    entity_description: str
    """textual description of the entity to lookup / create
    containing all available information
    """


entitites = {}
entitity_requests = {}
root = True
vector_store = build_vector_store()


def lookup_or_create_entity(param: LookupOrCreateParam) -> str | None:
    """lookup an entity by schema ID and description
    containing all available information, or create it if not found.
    return the entity's title / ID.
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
        target_schema = modify_schema(schema_cls.export_schema())
    except Exception as e:
        print(f"Error exporting schema for {param.schema_name}: {e}")
        return None

    provider_strategy = ProviderStrategy(
        schema=target_schema,
        strict=True
    )

    model = get_llm()
    model.max_retries = 1

    agent = create_agent(
        model=get_llm(),
        response_format=provider_strategy,
        tools=[
            lookup_or_create_entity
        ],
    )

    # prompt while providing previous requests (entitity_requests)
    # to avoid duplicates
    user_prompt = (
        "For the parent entity with OSW-ID 'Item:OSW" + entity_uuid.hex + "'"
        " create or lookup an linked entity "
        "based on the following description:"
        + ". \n------Description-------\n"
        + param.entity_description
        + ". \n-------------\n"
        "In case you find a similar request below, you are obliged to reuse"
        " its OSW-ID, e.g. Item:OSWabcdef1234567890 directly without calling "
        "'lookup_or_create_entity. Store this OSW-ID directly in the "
        "corresponding property.'\n"
        "------Previous requests-------\n"
        + "\n".join(
            f"OSW-ID: {osw_id} - Request for property '{r.property_name}', "
            f"schema '{r.schema_id}, {r.schema_name}': {r.entity_description}"
            for osw_id, r in entitity_requests.items()
        )
    )

    print(f"Invoking agent for entity creation with prompt:\n{user_prompt}")

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
        return None

    result = post_process_llm_json_response(result["structured_response"])
    result["uuid"] = str(entity_uuid)

    print(f"Structured Response for {param.schema_name}:")
    print(json.dumps(result, indent=2))

    # create an instance of the target data model from the result
    try:
        data_instance: OswBaseModel = schema_cls(**result)
    except Exception as e:
        print(f"Error creating data instance: {e}")
        return None

    entitites[data_instance.get_iri()] = data_instance
    return data_instance.get_iri()


result = lookup_or_create_entity(
    LookupOrCreateParam(
        parent_id="",
        property_name="",
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
