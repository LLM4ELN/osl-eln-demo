import json
from langchain.agents import create_agent
from os import environ
from langchain.agents.structured_output import ProviderStrategy
# from langchain_core.tools import tool
from langchain.tools import tool
from openai.types.responses import tool_choice_allowed
from rdflib import Graph, URIRef
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


from util import (
    modify_schema,
    post_process_llm_json_response,
)
from docling.document_converter import DocumentConverter

SOURCE = "data/02_secondary_data/results_protocols/Protokoll-Zugversuch RT Zx.xlsx"
# SOURCE = "data/sonic_resonance_test/01_primary_data/5.22_430_ERX1_ASTM_E1875.xlsx"

converter = DocumentConverter()
doc = converter.convert(SOURCE).document
markdown_doc = doc.export_to_markdown()


llm = ChatOpenAI(
    model=environ.get("API_MODEL"),
    api_key=environ.get("API_KEY"),
    base_url=environ.get("API_ENDPOINT"),
)

### GET SCHEMA

schema_generated = {
    "title": "Zugversuch Pr\u00fcfprotokoll Schema",
    "description": "JSON Schema zur Strukturierung von Zugversuch-Pr\u00fcfprotokollen, basierend auf mehreren \u00e4hnlichen Dokumenten. Enth\u00e4lt alle relevanten Pr\u00fcfparameter, Werkstoffdaten, Umgebungsbedingungen, verwendete Pr\u00fcfmittel und Bemerkungen.",
    "type": "object",
    "properties": {
        "protokoll": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "protokolltyp": {
                        "type": "string",
                        "description": "Typ des Protokolls, z.B. 'Pr\u00fcfprotokoll Zugversuch'",
                    },
                    "vorhaben_auftrags_nr": {
                        "type": "string",
                        "description": "Vorhaben- oder Auftragsnummer",
                    },
                    "ag_ontologie": {
                        "type": "string",
                        "description": "AG Ontologie, z.B. 'AG Ontologie Zugversuch'",
                    },
                    "proben_nr": {
                        "type": "string",
                        "description": "Nummer der Probe(n), z.B. 'Zx1' oder 'Zx2 / Zx3 / Zx4'",
                    },
                    "werkstoff": {
                        "type": "string",
                        "description": "Werkstoffbezeichnung, z.B. 'S355'",
                    },
                    "probenform": {
                        "type": "string",
                        "description": "Form der Probe mit Z.-Nr., z.B. 'E6x20x60'",
                    },
                    "waermebehandlung": {
                        "type": "string",
                        "description": "Art der W\u00e4rmebehandlung, z.B. '-' oder 'None'",
                    },
                    "prozedur_vers_file": {
                        "type": "string",
                        "description": "Referenz auf Prozedur- oder Versuchsdatei",
                    },
                    "norm": {
                        "type": "string",
                        "description": "Angegebene Norm, z.B. 'DIN EN ISO 6892-1'",
                    },
                    "pruefdatum": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Datum und Uhrzeit der Pr\u00fcfung",
                    },
                    "pruefer": {
                        "type": "string",
                        "description": "Name des Pr\u00fcfers, z.B. 'J.'",
                    },
                    "pruefleitung": {
                        "type": "string",
                        "description": "Name der Pr\u00fcfleitung, z.B. 'None'",
                    },
                    "datenfile": {
                        "type": "string",
                        "description": "Name der Datendatei, z.B. 'None'",
                    },
                    "lis_datei": {
                        "type": "string",
                        "description": "Name der LIS-Datei, z.B. 'None'",
                    },
                    "bemerkungen": {
                        "type": "string",
                        "description": "Allgemeine Bemerkungen zur Probe, z.B. 'In Walzrichtung aus Stahlblech entnommen'",
                    },
                    "umgebungsbedingungen": {
                        "type": "object",
                        "properties": {
                            "raumtemp_feuchte_ueberwachung": {
                                "type": "boolean",
                                "description": "Ob Raumtemperatur und -feuchte \u00fcberwacht wurden",
                            }
                        },
                    },
                    "pruefparameter": {
                        "type": "object",
                        "properties": {
                            "probenerwaermung": {
                                "type": "string",
                                "description": "Probenerw\u00e4rmung, z.B. '-'",
                            },
                            "dehngeschwindigkeit": {
                                "type": "number",
                                "description": "Dehngeschwindigkeit in 1/s",
                            },
                            "umschaltpunkt": {
                                "type": "string",
                                "description": "Umschaltpunkt in %",
                            },
                            "weggeschwindigkeit": {
                                "type": "string",
                                "description": "Weggeschwindigkeit in mm/s",
                            },
                            "prueftemperatur": {
                                "type": "string",
                                "description": "Pr\u00fcftemperatur, z.B. 'RT' oder '\u00b0C'",
                            },
                        },
                    },
                    "verwendete_pruefmittel": {
                        "type": "object",
                        "properties": {
                            "pruefmaschine": {
                                "type": "string",
                                "description": "Pr\u00fcfmaschine, z.B. 'Instron 4505'",
                            },
                            "kraftbereich": {
                                "type": "number",
                                "description": "Kraftbereich in kN",
                            },
                            "extensometer": {
                                "type": "string",
                                "description": "Extensometer mit Seriennummer, z.B. 'HBM; Serien-Nr.: 28023/28024'",
                            },
                            "dehnungsbereich": {
                                "type": "number",
                                "description": "Dehnungsbereich in %",
                            },
                            "nennger\u00e4temessl\u00e4nge": {
                                "type": "number",
                                "description": "Nennger\u00e4temessl\u00e4nge in mm",
                            },
                            "mikrometerschraube": {
                                "type": "string",
                                "description": "Mikrometerschraube, z.B. 'Orion Satz 1 (Werkstatt)'",
                            },
                            "messschieber": {
                                "type": "string",
                                "description": "Messschieber, z.B. 'Orion Satz 1, BA606983'",
                            },
                            "thermoelemente": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "nummer": {
                                            "type": "string",
                                            "description": "Nummer des Thermoelements",
                                        },
                                        "status": {
                                            "type": "string",
                                            "description": "Status, z.B. 'None' oder 'vor dem Versuchstart'",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "vorversuch": {
                        "type": "object",
                        "properties": {
                            "kraft_max": {
                                "type": "string",
                                "description": "Maximale Kraft im Vorversuch in kN",
                            },
                            "e_modul": {
                                "type": "string",
                                "description": "E-Modul im Vorversuch in GPa",
                            },
                            "aufheizzeit_bis_pt": {
                                "type": "string",
                                "description": "Aufheizzeit bis Pr\u00fcftemperatur in min",
                            },
                            "dehnungsanzeige_rt": {
                                "type": "string",
                                "description": "Dehnungsanzeige bei Raumtemperatur in %",
                            },
                            "dehnung_max": {
                                "type": "string",
                                "description": "Maximale Dehnung",
                            },
                            "referenzwert_e_modul": {
                                "type": "string",
                                "description": "Referenzwert E-Modul in GPa",
                            },
                            "durchwaermzeit_bei_pt": {
                                "type": "string",
                                "description": "Durchw\u00e4rmzeit bei Pr\u00fcftemperatur in min",
                            },
                            "dehnungsanzeige_pt": {
                                "type": "string",
                                "description": "Dehnungsanzeige bei Pr\u00fcftemperatur in %",
                            },
                        },
                    },
                    "bemerkungen_zusatz": {
                        "type": "string",
                        "description": "Zus\u00e4tzliche Bemerkungen, z.B. 'Abweichend von den anderen Proben. Weggeschwindigkeit wurde erh\u00f6ht.'",
                    },
                    "signatur_pruefer": {
                        "type": "string",
                        "description": "Signatur des Pr\u00fcfers, z.B. 'J.'",
                    },
                    "signatur_pruefleitung": {
                        "type": "string",
                        "description": "Signatur der Pr\u00fcfleitung, z.B. 'R.'",
                    },
                },
                "required": [
                    "protokolltyp",
                    "vorhaben_auftrags_nr",
                    "ag_ontologie",
                    "proben_nr",
                    "werkstoff",
                    "probenform",
                    "waermebehandlung",
                    "prozedur_vers_file",
                    "norm",
                    "pruefdatum",
                    "pruefer",
                    "pruefleitung",
                    "datenfile",
                    "lis_datei",
                    "bemerkungen",
                    "umgebungsbedingungen",
                    "pruefparameter",
                    "verwendete_pruefmittel",
                    "vorversuch",
                    "bemerkungen_zusatz",
                    "signatur_pruefer",
                    "signatur_pruefleitung",
                ],
            },
        }
    },
    "required": ["protokoll"],
}


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


schema = {
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
                    "type": "object",
                    "properties": {
                        "@id": {
                            "type": "string",
                            "pattern": "^http://qudt.org/vocab/unit/*",
                        },
                    },
                    "required": ["@id"],
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


### GET DATA

#prompt = "Please extract data of this document."
prompt = "Please extract data from this document and use the 'search_unit' tool for any units (e.g., '1/s') you encounter."

sys_prompt = (
    "You are an expert laboratory assistant."
    " Always use the 'search_unit' tool to look up QUDT unit IRIs for any units you encounter in your responses."
)

units = Graph().parse("unit.ttl")

@tool
def search_unit(query: str) -> str:
    """Get QUDT unit IRI for a given query."""

    query = f"""
    PREFIX qudt: <http://qudt.org/schema/qudt/>

    SELECT ?unit
    WHERE {{
        {{
            ?unit rdfs:label "{query}" .
        }}
        UNION
        {{
            ?unit qudt:symbol "{query}" .
        }}
    }}
    """

    result = units.query(query)
    print(result)
    breakpoint()

    if len(result.bindings) > 0:
        return URIRef(result.bindings[0]["unit"])

from opensemantic.lab.v1 import LaboratoryProcess

target_data_model = LaboratoryProcess


provider_strategy = ProviderStrategy(
    schema=modify_schema(target_data_model.export_schema()),
    #schema=schema,
    strict=True,
)




agent = create_agent(
    model=llm,
    tools=[search_unit],
    response_format=provider_strategy,
)

result = agent.invoke({
    "messages": [
        {
                "role": "user",
                "content": sys_prompt + "\n\n" + prompt,
            },
    ]
})

print("Agent Result:", result)