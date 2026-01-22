import json
from os import environ

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.structured_output import ProviderStrategy

from docling.document_converter import DocumentConverter



# ---------------------------------------------------------------------
# Load document
# ---------------------------------------------------------------------

SOURCE = "data/02_secondary_data/results_protocols/Protokoll-Zugversuch RT Zx.xlsx"

converter = DocumentConverter()
doc = converter.convert(SOURCE).document
markdown_doc = doc.export_to_markdown()


# ---------------------------------------------------------------------
# LLM setup (NO tools, NO RDF)
# ---------------------------------------------------------------------

llm = ChatOpenAI(
    model=environ.get("API_MODEL"),
    api_key=environ.get("API_KEY"),
    base_url=environ.get("API_ENDPOINT"),
    temperature=0,
)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert laboratory assistant.\n"
                "Extract information from the document.\n\n"
                "Rules:\n"
                "- Output MUST be valid JSON\n"
                "- Output MUST conform EXACTLY to the provided JSON Schema\n"
                "- Do NOT add fields\n"
                "- Do NOT invent data\n"
                "- Do NOT invent IRIs or ontology terms\n"
                "- Use plain strings and numbers only\n"
                "- If a value is missing, use an empty string\n"
            ),
        ),
        ("human", "{document}"),
    ]
)


# ---------------------------------------------------------------------
# Structured output parser
# ---------------------------------------------------------------------

extraction_schema = {
    "title": "Hardness Test – Raw Extraction Schema",
    "type": "object",
    "properties": {
        "test_id": {
            "type": "string",
            "description": "Identifier for the test (can be synthetic)"
        },
        "test_label": {
            "type": "string",
            "description": "Human-readable test name"
        },
        "test_comment": {
            "type": "string",
            "description": "General comments from the protocol"
        },
        "order_number": {
            "type": "string",
            "description": "Order or project number"
        },
        "sample": {
            "type": "object",
            "properties": {
                "sample_id": {"type": "string"},
                "sample_label": {"type": "string"},
                "sample_comment": {"type": "string"},
                "sample_number": {"type": "string"}
            },
            "required": [
                "sample_id",
                "sample_label",
                "sample_comment",
                "sample_number"
            ]
        },
        "displacement_rate": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "comment": {"type": "string"},
                "value": {"type": "number"},
                "unit": {
                    "type": "string",
                    "description": "Unit as written in the document, e.g. 1/s"
                }
            },
            "required": [
                "label",
                "comment",
                "value",
                "unit"
            ]
        }
    },
    "required": [
        "test_id",
        "test_label",
        "test_comment",
        "order_number",
        "sample",
        "displacement_rate"
    ]
}


# parser = ProviderStrategy(
#     schema=extraction_schema,
#     strict=True,
# )

structured_llm = llm.with_structured_output(
    extraction_schema,
    strict=True,
)



# ---------------------------------------------------------------------
# Extraction chain
# ---------------------------------------------------------------------


pipeline = prompt | structured_llm


# ---------------------------------------------------------------------
# Run Step 1
# ---------------------------------------------------------------------

raw_extraction = pipeline.invoke(
    {"document": markdown_doc}
)

print(json.dumps(raw_extraction, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

print("STEP 1 — RAW EXTRACTION")
print(json.dumps(raw_extraction, indent=2, ensure_ascii=False))
# ---------------------------------------------------------------------
# STEP 2: RDF MAPPING
# ---------------------------------------------------------------------


from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from urllib.parse import quote

# Step 1 output (replace this with your real Step 1 JSON)
raw = raw_extraction  # JSON from Step 1

# Define a safe test namespace
EX = Namespace("http://test.example.org/")
QUDT = Namespace("http://qudt.org/vocab/unit/")

# Load your QUDT units TTL file
units_graph = Graph()
units_graph.parse("unit.ttl", format="ttl")

def lookup_unit(unit_str: str) -> URIRef:
    """
    Map a unit string (e.g., '1/s') to a QUDT IRI from your TTL file.
    Returns a URIRef. If not found, creates a fallback safe URI.
    """
    query = f"""
    PREFIX qudt: <http://qudt.org/schema/qudt/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?unit
    WHERE {{
        {{ ?unit rdfs:label "{unit_str}" . }}
        UNION
        {{ ?unit qudt:symbol "{unit_str}" . }}
    }}
    LIMIT 1
    """
    res = units_graph.query(query)
    for row in res:
        return row.unit
    # fallback safe URI
    
    if not res:
        print("unit iri not found for:", unit_str)
        print("editing unit sting and trying again...")
        if '1/' in unit_str:
            unit_str = unit_str.replace('1/', '/')
            print("new unit string:", unit_str)
            return lookup_unit(unit_str)
        else:
            safe_unit = unit_str.replace("/", "_per_").replace(" ", "_")
            fallback_iri = URIRef(f"http://qudt.org/vocab/unit/{safe_unit}")
            print("using fallback IRI:", fallback_iri)
            return fallback_iri
            


# -----------------------------
# Create RDF graph
# -----------------------------
g = Graph()
g.bind("ex", EX)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)
g.bind("qudt", QUDT)

# -----------------------------
# Create top-level test node
# -----------------------------
test_id_safe = quote(raw["test_id"])
test_node = URIRef(f"{EX}test/{test_id_safe}")
g.add((test_node, RDF.type, EX.HardnessTest))
g.add((test_node, RDFS.label, Literal(raw["test_label"])))
g.add((test_node, RDFS.comment, Literal(raw["test_comment"])))
g.add((test_node, EX.orderNumber, Literal(raw["order_number"])))

# -----------------------------
# Add sample node
# -----------------------------
sample_data = raw["sample"]
sample_id_safe = quote(sample_data["sample_id"])
sample_node = URIRef(f"{EX}sample/{sample_id_safe}")
g.add((sample_node, RDF.type, EX.Sample))
g.add((sample_node, RDFS.label, Literal(sample_data["sample_label"])))
g.add((sample_node, RDFS.comment, Literal(sample_data["sample_comment"])))
g.add((sample_node, EX.sampleNumber, Literal(sample_data["sample_number"])))

# Link sample to test
g.add((test_node, EX.input, sample_node))

# -----------------------------
# Add displacement rate node
# -----------------------------
disp = raw["displacement_rate"]
disp_node = URIRef(f"{EX}test/{test_id_safe}/displacementRate")
g.add((disp_node, RDF.type, EX.DisplacementRate))
g.add((disp_node, RDFS.label, Literal(disp["label"])))
g.add((disp_node, RDFS.comment, Literal(disp["comment"])))
g.add((disp_node, EX.value, Literal(disp["value"], datatype=XSD.decimal)))

# Resolve and add unit
print("looking up unit IRI for:", disp["unit"])
unit_iri = lookup_unit(disp["unit"])
print("mapped to unit IRI:", unit_iri)
g.add((disp_node, EX.unit, unit_iri))

# Link displacementRate to test
g.add((test_node, EX.displacementRate, disp_node))

# -----------------------------
# Serialize RDF graph (Turtle)
# -----------------------------
turtle_output = g.serialize(format="turtle")

print("STEP 2 — RDF MAPPING:")
print(turtle_output)