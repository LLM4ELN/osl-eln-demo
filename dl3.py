import json
from docling.backend.pdf_backend import PdfPageBackend
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from docling.datamodel.base_models import ConversionStatus, InputFormat, PipelineOptions
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PipelineOptions,
    EasyOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from util import post_process_llm_json_response

from os import environ, pipe
from docling.document_converter import DocumentConverter


# SOURCE = "data/02_secondary_data/results_protocols/Protokoll-Zugversuch RT Zx.xlsx"
# SOURCE = "data/sonic_resonance_test/01_primary_data/5.22_430_ERX1_ASTM_E1875.xlsx"
SOURCE = "data/03_metadata/Testing_measuring_instruments_tensile_test.pdf"

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.ocr_options = TesseractOcrOptions()

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
)
doc = converter.convert(SOURCE).document
markdown_doc = doc.export_to_markdown()

print(markdown_doc)

llm = ChatOpenAI(
    model=environ.get("API_MODEL"),
    api_key=environ.get("API_KEY"),
    base_url=environ.get("API_ENDPOINT"),
)

### GET SCHEMA

prompt = (
    "Please extract a JSON schema that captures the structure of "
    "this document and can be used to load the data of multiple similar ones."
    "The JSON schema must have a top-level 'title' and 'description' field."
)

sys_prompt = "You are an expert laboratory assistant. You always answer in valid JSON. "

agent = create_agent(model=llm)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": sys_prompt + "\n\n" + prompt,
            },
            {
                "role": "user",
                "content": markdown_doc,
            },
        ]
    }
)

schema = None
for message in response["messages"]:
    if isinstance(message, AIMessage):
        print(">> got schema")
        schema = json.loads(message.content)
        print(json.dumps(schema, indent=2))


### GET DATA

prompt = "Please extract data of this document."

sys_prompt = (
    "You are an expert laboratory assistant. "
    "You always answer in valid JSON according to the provided schema. "
    "If a field is not specified, leave it empty or null. "
    "Do not add any additional fields that are not defined in the schema. "
    "If you encounter a property that is annotated with a 'range', "
    "leave it empty / null."
)

schema = {
    "type": "object",
    "title": "foo",
    "description": "bar",
    "properties": {
        "data": schema,
    },
    "required": ["data"],
}

provider_strategy = ProviderStrategy(schema=schema, strict=True)

agent = create_agent(
    model=llm,
    response_format=provider_strategy,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": sys_prompt + "\n\n" + prompt,
            },
            {
                "role": "user",
                "content": markdown_doc,
            },
        ]
    }
)

result = post_process_llm_json_response(result["structured_response"])

print("Structured Response:")
print(json.dumps(result, indent=2))
