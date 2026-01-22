from typing import Any, Union
import uuid
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

# note: pydantic v2 models currently cannot be initialized from JSON directly
# + osw-python requires v1 models for now
# => we use v1 models here
# see also https://www.jujens.eu/posts/en/2025/Apr/26/pydantic-enums/
from opensemantic.v1 import OswBaseModel
from opensemantic.lab.v1 import LaboratoryProcess
import opensemantic.core.v1
import opensemantic.base.v1
import opensemantic.lab.v1
from pydantic import BaseModel

from util import (
    modify_schema, post_process_llm_json_response,
    post_process_llm_json_response
)

import json

from llm_init import get_llm, llm
from schema_catalog import lookup_exact_schema
#from osl_init import osl_client

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
    "use the tool 'lookup_or_create_entity' to lookup or create the referenced entity."
    "Store the returned ID (if not None) in the corresponding fields."
)



# create a structured output agent with a provider strategy
# based on the target data model's schema
# preprocess the schema to comply
# with https://platform.openai.com/docs/guides/structured-outputs#supported-schemas  # noqa: E501

provider_strategy = ProviderStrategy(
    schema=modify_schema(target_data_model.export_schema()),
    strict=True
)

# create a tool that lookup a connected entity, return its IDs

from langchain.tools import tool

class KeyValuePair(BaseModel):
    key: str
    value: Union[str, int, float, bool, "KeyValuePair", list[Union[str, int, float, bool, "KeyValuePair"]]]

class LookupOrCreateParamStructured(BaseModel):
    # model_config = {
    #     "json_schema_extra": {
    #     "required": ["schema_id", "data"],
    #     }
    # }
    schema_id: str
    """ID of the schema to use for lookup / creation of the entity"""
    data: list[KeyValuePair]
    """serialized JSON data of the entity to lookup / create
    e.g. [{"key": "title", "value": "Example Entity"}]
    """

arg_schema = {
  "$defs": {
    "KeyValuePair": {
      "properties": {
        "key": {
          "title": "Key",
          "type": "string"
        },
        "value": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "integer"
            },
            {
              "type": "number"
            },
            {
              "type": "boolean"
            },
            {
              "$ref": "#/$defs/KeyValuePair",
              "type": "object",
              "additionalProperties": False
            },
            {
              "items": {
                "anyOf": [
                  {
                    "type": "string"
                  },
                  {
                    "type": "integer"
                  },
                  {
                    "type": "number"
                  },
                  {
                    "type": "boolean"
                  },
                  {
                    "$ref": "#/$defs/KeyValuePair",
                    "type": "object",
                    "additionalProperties": False
                  }
                ]
              },
              "type": "array"
            }
          ],
          "title": "Value"
        }
      },
      "required": [
        "key",
        "value"
      ],
      "title": "KeyValuePair",
      "type": "object"
    }
  },
  "properties": {
    "schema_id": {
      "title": "Schema Id",
      "type": "string"
    },
    "data": {
      "items": {
        "$ref": "#/$defs/KeyValuePair"
      },
      "title": "Data",
      "type": "array"
    }
  },
  "required": [
    "schema_id"
    "data"
  ],
  "title": "LookupOrCreateParam",
  "type": "object"
}

@tool(args_schema=arg_schema)
def lookup_or_create_entity_structured(schema_id: str, data: list[KeyValuePair]) -> str:
    """lookup an entity by schema ID, description containing all available information
    and structured data based on the description, or create it if not found.
    return the entity's title / ID.
    """
    print("Lookup or create entity:", schema_id, data)
    
    return "Item:ExampleEntityID12345"

class LookupOrCreateParam(BaseModel):
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

def lookup_or_create_entity(param: LookupOrCreateParam) -> str | None:
    """lookup an entity by schema ID and description containing all available information,
    or create it if not found.
    return the entity's title / ID.
    """
    print(f"Lookup or create entity for property '{param.property_name}', range '{param.schema_id}, {param.schema_name}' based on description {param.entity_description}")
    
    entity_uuid = uuid.uuid4()
    entity_id = "Item:OSW" + entity_uuid.hex
    entitity_requests[entity_id] = param
    
    # lookup the schema_name in the modules
    # opensemantic.core, opensemantic.base, opensemantic.lab, ...
    # schema_cls: OswBaseModel | None = None
    # if hasattr(opensemantic.lab.v1, param.schema_name):
    #     schema_cls = getattr(opensemantic.lab.v1, param.schema_name)
    # elif hasattr(opensemantic.base.v1, param.schema_name):
    #     schema_cls = getattr(opensemantic.base.v1, param.schema_name)
    # elif hasattr(opensemantic.core.v1, param.schema_name):
    #     schema_cls = getattr(opensemantic.core.v1, param.schema_name)
    
    prompt = ""
    if param.schema_id != "":
        prompt = "The schema id is " + param.schema_id + ". "
    if param.schema_name != "":
        prompt += "The schema name is " + param.schema_name + ". "
    if param.entity_description != "":
        prompt += "The entity it want to describe: " + param.entity_description + ". "
    schema_name = lookup_exact_schema(prompt)
    print(f"LLM returned class path: {schema_name}")
    # get the class from the path
    schema_cls: OswBaseModel = eval(schema_name)
    
    if schema_cls is None:
        #raise ValueError(f"Schema name {param.schema_name} not found in opensemantic modules")
        print(f"Schema name {param.schema_name} not found in opensemantic modules")
        return None
    else:
        print(f"Found schema class for {param.schema_name}: {schema_cls}")
    
    try:
        # Fixme: schema contains "definitions" instead of "$defs"
        # but $refs are pointing to ""
        target_schema = modify_schema(schema_cls.export_schema())
    except Exception as e:
        print(f"Error exporting schema for {param.schema_name}: {e}")
        return None
    
    provider_strategy = ProviderStrategy(
        schema=target_schema,
        #schema=modify_schema(json.loads(schema_cls.schema_json())),
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

    try:
        result = agent.invoke({
            "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": param.entity_description
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
        property_name="",
        schema_id="LaboratoryProcess",
        schema_name="LaboratoryProcess",
        entity_description=(
            "A laboratory process to document an experiment "
            "created by Dr. John Doe, Example Lab, "
            "starting at 01.01.2025 and ending at 02.01.2025, "
            "status in progress."
        )
    )
)

print("Created / Looked up entities:")
for i, e in entitites.items():
    e: OswBaseModel
    print(f"#### {i} ({e.name}) ####")
    print(e.json(indent=2, exclude_none=True))
