from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

# note: pydantic v2 models currently cannot be initialized from JSON directly
# + osw-python requires v1 models for now
# => we use v1 models here
# see also https://www.jujens.eu/posts/en/2025/Apr/26/pydantic-enums/
from opensemantic.lab.v1 import LaboratoryProcess

from util import (
    modify_schema, post_process_llm_json_response,
    post_process_llm_json_response
)

import json

from llm_init import llm
from osl_init import osl_client

target_data_model = LaboratoryProcess

prompt = (
    "Describe the process of PCR in a laboratory setting"
    "starting at 01.01.2025 and ending at 02.01.2025."
    "Status is in progress."
)

sys_prompt = (
    "You are an expert laboratory assistant. "
    "You always answer in valid JSON according to the provided schema. "
    "If a field is not specified, leave it empty or null. "
    "Do not add any additional fields that are not defined in the schema. "
    "If you encounter a property that is annotated with a 'range', "
    "leave it empty / null."
)

# create a structured output agent with a provider strategy
# based on the target data model's schema
# preprocess the schema to comply
# with https://platform.openai.com/docs/guides/structured-outputs#supported-schemas  # noqa: E501

provider_strategy = ProviderStrategy(
    schema=modify_schema(target_data_model.export_schema()),
    strict=True
)

agent = create_agent(
    model=llm,
    response_format=provider_strategy
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": sys_prompt + "\n\n" + prompt
    }]
})

result = post_process_llm_json_response(result["structured_response"])

print("Structured Response:")
print(json.dumps(result, indent=2))

# create an instance of the target data model from the result
data_instance = target_data_model(**result)

osl_client.store_entity(data_instance)
