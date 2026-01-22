from itertools import zip_longest
import json, time
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from opensemantic.lab.v1 import LaboratoryProcess
from util import modify_schema, post_process_llm_json_response, post_process_llm_json_response, remove_nulls
from llm_init import llm
from osl_init import osl_client
from tools import pathIterator

waitTime = 60
sys_prompt = (
    "You are an expert laboratory assistant for tensile mechanical tests. "
    "You always answer in valid JSON according to the provided schema. "
)
createJsonPrompt = (
    "Extract a JSON schema that captures the structure of this document and can be used to load data "
    "of multiple similar ones. The JSON schema must have a top-level 'title' and 'description' field. The 'title' field must be ^[a-zA-Z0-9_-]+$. "
    "All keys must be in english language and use only ascii characters. "
    'Generate only valid JSON output. Do not include any natural language, explanations, markdown, or additional text. '
    'The output must be a single, well-formed JSON object or array. Do not wrap the JSON in code blocks or quotes. \n'
    'Example: {"result": "success", "data": [1, 2, 3]}'
)

# OSL data model = data schema
dataModelOSL = LaboratoryProcess
schemaOSL = dataModelOSL.export_schema()
providerStrategyOSL = ProviderStrategy(schema=modify_schema(schemaOSL), strict=True) #modify schema -> openAI compliant
agentOSL = create_agent(model=llm, response_format=providerStrategyOSL)

# Agent for creating general json
agentCreateJson = create_agent(model=llm)

rawDataSet = pathIterator('../tensileData')
for file, contentText in rawDataSet:
    if not contentText:
        continue
    if file.name not in ("Protokoll-Zugversuch RT Zx.xlsx", "results_sonic_resonance_tests.xlsx"):
        continue
    print(file,'\n',contentText[:200])

    # create a general json from it: no LLM usage
    time.sleep(waitTime)
    print('Start creating original json...')
    startTime = time.time()
    result = agentCreateJson.invoke({"messages": [{"role": "user", "content": rawDataSet.prompt+'\n\n'+contentText}]})
    print(f'  ... end: {time.time()-startTime:.1f} s')
    jsonOriginal = result['messages'][1].content

    # use that json and create a OSL-schema based json from it
    time.sleep(waitTime)
    print('Start creating OSL-schema-based json...')
    startTime = time.time()
    result = agentOSL.invoke({"messages": [{"role": "user", "content": sys_prompt + "\n\n" + jsonOriginal}]})
    print(f'  ... end: {time.time()-startTime:.1f} s')
    jsonOSL = post_process_llm_json_response(result["structured_response"])

    # create a JSON schema, that fits to this test
    time.sleep(waitTime)
    print('Start creating new schema json...')
    startTime = time.time()
    result = agentCreateJson.invoke({"messages": [{"role": "user", "content": createJsonPrompt+'\n\n'+jsonOriginal}]})
    print(f'  ... end: {time.time()-startTime:.1f} s')
    content = result['messages'][1].content.strip()
    content = content.replace('\u00b5','mu').replace('\u03c1','rho')  #TODO separate function
    schemaNew = json.loads(content.strip())
    providerStrategyNew = ProviderStrategy(schema=modify_schema(schemaNew), strict=True) #modify schema -> openAI compliant

    # use that json and create a OSL approved json from it
    time.sleep(waitTime)
    print('Start creating new-schema-based json...')
    startTime = time.time()
    agentNew = create_agent(model=llm, response_format=providerStrategyNew)
    print(f'  ... end: {time.time()-startTime:.1f} s')
    result = agentNew.invoke({"messages": [{"role": "user", "content": sys_prompt + "\n\n" + jsonOriginal}]})
    jsonNew = post_process_llm_json_response(result["structured_response"])

    print('------------------------------------------')
    # print side by side
    left = json.dumps(jsonOSL, indent=2).splitlines()
    right = json.dumps(jsonNew, indent=2).splitlines()
    width = min(max(len(line) for line in left), 80)
    for l, r in zip_longest(left, right, fillvalue=""):
        print(f"{l[:width].ljust(width)}    {r[:width]}")
    print('\n\n---------- Which is better?  -------------')

    print('\n\nUpload result to OSL...')
    data_instance = dataModelOSL(**jsonOSL)
    osl_client.store_entity(data_instance)
