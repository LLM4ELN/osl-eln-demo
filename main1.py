import json
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from opensemantic.lab.v1 import LaboratoryProcess
from util import (modify_schema, post_process_llm_json_response, post_process_llm_json_response)
from llm_init import llm
from osl_init import osl_client
from tools import pathIterator

sys_prompt = (
    "You are an expert laboratory assistant for tensile mechanical tests. "
    "You always answer in valid JSON according to the provided schema. "
    "If a field is not specified, leave it empty or null. "
    "Do not add any additional fields that are not defined in the schema. "
    "If you encounter a property that is annotated with a 'range', "
    "leave it empty / null."
)

agentCreateJson = create_agent(model=llm)
rawDataSet = pathIterator('../tensileData')
dataModel = LaboratoryProcess
dataSchema = dataModel.export_schema()
providerStrategy = ProviderStrategy(schema=modify_schema(dataSchema), strict=True) #modify schema -> openAI compliant
agentUseSchema = create_agent(model=llm, response_format=providerStrategy)
for file, contentText in rawDataSet:
    if not contentText:
        continue
    if file.name != "Protokoll-Zugversuch RT Zx.xlsx":
        continue
    print(file, contentText[:200])
    result = agentCreateJson.invoke({"messages": [{"role": "user", "content": rawDataSet.prompt+'\n\n'+contentText}]})
    contentJson = result['messages'][1].content

    result = agentUseSchema.invoke({"messages": [{"role": "user", "content": sys_prompt + "\n\n" + contentJson}]})
    result = post_process_llm_json_response(result["structured_response"])
    print("\n\nStructured Response:", json.dumps(result, indent=2))

    break
