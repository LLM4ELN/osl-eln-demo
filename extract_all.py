import json
from os import environ
from urllib.parse import quote
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, DC

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from docling.document_converter import DocumentConverter



# ---------------------------------------------------------------------
# Load document
# ---------------------------------------------------------------------

SOURCE = "data/02_secondary_data/results_protocols/Protokoll-Zugversuch RT Zx.xlsx"

converter = DocumentConverter()
doc = converter.convert(SOURCE).document
markdown_doc = doc.export_to_markdown()

print("Document loaded and converted to markdown.")
print(markdown_doc)

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

def schema_type(iri):
    return {
        "type": "array",
        "items": {
            "type": "string",
            "const": iri,
        },
        "minItems": 1,
        "maxItems": 1,
    }


def schema_string():
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"@value": {"type": "string"}},
            "required": ["@value"],
        },
    }


def schema_decimal():
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "@value": {"type": "string"},
                "@type": {"const": "http://www.w3.org/2001/XMLSchema#decimal"},
            },
            "required": [
                "@value",
                "@type",
            ],
        },
        "minItems": 1,
        "maxItems": 1,
    }


extraction_schema = {
    "title": "Hardness Test Schema",
    "description": "A very minimal JSON schema for the hardness test processes.",
    "type": "object",
    "properties": {
        "@id": {"type": "string"},
        "@type": schema_type("http://example.org#HardnessTest"),
        "http://www.w3.org/2000/01/rdf-schema#label": schema_string(),
        "http://www.w3.org/2000/01/rdf-schema#comment": schema_string(),
        "http://example.org#orderNumber": schema_string(),
        "http://example.org#input": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "@id": {"type": "string"},
                    "@type": schema_type("http://example.org#Sample"),
                    "http://www.w3.org/2000/01/rdf-schema#label": schema_string(),
                    "http://www.w3.org/2000/01/rdf-schema#comment": schema_string(),
                    "http://example.org#sampleNumber": schema_string(),
                },
                "required": [
                    "@id",
                    "@type",
                    "http://www.w3.org/2000/01/rdf-schema#label",
                    "http://www.w3.org/2000/01/rdf-schema#comment",
                    "http://example.org#sampleNumber",
                ],
            },
        },
        "http://example.org#displacementRate": {
            "type": "object",
            "properties": {
                "@type": schema_type("http://example.org#DisplacementRate"),
                "http://www.w3.org/2000/01/rdf-schema#label": schema_string(),
                "http://www.w3.org/2000/01/rdf-schema#comment": schema_string(),
                "http://example.org#value": schema_decimal(),
                "http://example.org#unit": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"@value": {"type": "string"}},
                        "required": ["@value"],
                    },
                    "minItems": 1,
                    "maxItems": 1,
                },
            },
            "required": [
                "@type",
                "http://www.w3.org/2000/01/rdf-schema#label",
                "http://www.w3.org/2000/01/rdf-schema#comment",
                "http://example.org#value",
                "http://example.org#unit",
            ],
        },
    },
    "required": [
        "@id",
        "@type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://example.org#orderNumber",
        "http://example.org#input",
        "http://example.org#displacementRate",
    ],
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
# RDF Graph
# ---------------------------------------------------------------------

EX = Namespace("http://test.example.org/")
g = Graph()
g.bind("ex", EX)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)
g.bind("dc", DC)


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
        else: # fallback will probably not be ok... find another solution later
            safe_unit = unit_str.replace("/", "_per_").replace(" ", "_")
            fallback_iri = URIRef(f"http://qudt.org/vocab/unit/{safe_unit}")
            print("using fallback IRI:", fallback_iri)
            return fallback_iri

# Helper to add JSON-LD arrays
def get_value(obj):
    if isinstance(obj, list) and len(obj) > 0 and "@value" in obj[0]:
        return obj[0]["@value"]
    return ""

# Multiple test support
tests = raw_extraction if isinstance(raw_extraction, list) else [raw_extraction]

for test in tests:
    test_node = URIRef(test["@id"])
    g.add((test_node, RDF.type, EX.HardnessTest))
    g.add((test_node, RDFS.label, Literal(get_value(test["http://www.w3.org/2000/01/rdf-schema#label"]))))
    g.add((test_node, RDFS.comment, Literal(get_value(test["http://www.w3.org/2000/01/rdf-schema#comment"]))))
    g.add((test_node, EX.orderNumber, Literal(get_value(test["http://example.org#orderNumber"]))))
    g.add((test_node, DC.creator, Literal("LLM extraction")))
    g.add((test_node, DC.date, Literal("2026-01-22", datatype=XSD.date)))

    # Samples
    for sample in test["http://example.org#input"]:
        sample_node = URIRef(sample["@id"])
        g.add((sample_node, RDF.type, EX.Sample))
        g.add((sample_node, RDFS.label, Literal(get_value(sample["http://www.w3.org/2000/01/rdf-schema#label"]))))
        g.add((sample_node, RDFS.comment, Literal(get_value(sample["http://www.w3.org/2000/01/rdf-schema#comment"]))))
        g.add((sample_node, EX.sampleNumber, Literal(get_value(sample["http://example.org#sampleNumber"]))))
        g.add((test_node, EX.input, sample_node))

    # Displacement Rate
    dr = test["http://example.org#displacementRate"]
    dr_node = URIRef(f"{test_node}/displacementRate")
    g.add((dr_node, RDF.type, EX.DisplacementRate))
    g.add((dr_node, RDFS.label, Literal(get_value(dr["http://www.w3.org/2000/01/rdf-schema#label"]))))
    g.add((dr_node, RDFS.comment, Literal(get_value(dr["http://www.w3.org/2000/01/rdf-schema#comment"]))))
    g.add((dr_node, EX.value, Literal(get_value(dr["http://example.org#value"]), datatype=XSD.decimal)))
        
    unit_str = get_value(dr["http://example.org#unit"])
    unit_iri = lookup_unit(unit_str)
    g.add((dr_node, EX.unit, unit_iri))

    g.add((test_node, EX.displacementRate, dr_node))

# ---------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------

turtle_output = g.serialize(format="turtle")
jsonld_output = g.serialize(format="json-ld")

print("\n--- Turtle ---\n")
print(turtle_output)

print("\n--- JSON-LD ---\n")
print(jsonld_output)
