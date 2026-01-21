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
)
createJsonPrompt = (
    "Extract a JSON schema that captures the structure of this json document and can be used to load the data "
    "of multiple similar ones. The JSON schema must have a top-level 'title' and 'description' field."
    "Only answer in a valid JSON only with no other text."
)

agentCreateJson = create_agent(model=llm)
rawDataSet = pathIterator('../tensileData')
for file, contentText in rawDataSet:
    if not contentText:
        continue
    if file.name != "Protokoll-Zugversuch RT Zx.xlsx":
        continue
    print(file, contentText[:200])
    result = agentCreateJson.invoke({"messages": [{"role": "user", "content": rawDataSet.prompt+'\n\n'+contentText}]})
    contentJson = result['messages'][1].content

    result = agentCreateJson.invoke({"messages": [{"role": "user", "content": createJsonPrompt+'\n\n'+contentJson}]})
    content2 = result['messages'][1].content.strip()
    if content2.endswith('```'):
        content2 = content2[:-3]
    if content2.startswith('```json'):
        content2 = content2[7:]
    schema = json.loads(content2.strip())
    print(schema)
    providerStrategy = ProviderStrategy(schema=modify_schema(schema), strict=True) #modify schema -> openAI compliant

    agentUseSchema = create_agent(model=llm, response_format=providerStrategy)
    result = agentUseSchema.invoke({"messages": [{"role": "user", "content": sys_prompt + "\n\n" + contentJson}]})
    result = post_process_llm_json_response(result["structured_response"])
    print("\n\nStructured Response:", json.dumps(result, indent=2))

    break
