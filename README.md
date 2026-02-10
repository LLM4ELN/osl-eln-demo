# OSL ELN Demo

> **[Iterative Agent Demo Visualization](docs/README.md)** - See how the iterative agent builds a knowledge graph from natural language experiment descriptions.

![Iterative Agent Knowledge Graph Animation](docs/knowledge-graph-d3.gif)

## Prerequisites

Follow https://github.com/LLM4ELN/llm4eln-assessment to set up the project, virtual environment and llm provider (step 1-3).

Make sure to install the additional packages `opensemantic.lab` and `"git+https://github.com/OpenSemanticLab/osw-python"`, overall
```bash
 uv add langchain langchain-openai langchain-ollama langchain-anthropic langchain-google-genai langchain-ollama python-dotenv opensemantic.lab "git+https://github.com/OpenSemanticLab/osw-python"
```
or use
```bash
uv sync
```

### OSL Login

Go to https://llm4eln.semos.dev and login via ORCID

Optional: Go to Preferences and set your Real Name (otherwise only your ORCID ID will be shown)

Got to https://llm4eln.semos.dev/wiki/Special:BotPasswords and create a bot password. 
Grant "edit existing pages", "create, edit and move pages", "upload new files", "upload, replace and move files" and "delete pages, revisions and log entries" permissions.

### .env File
Create a `.env` file in the project root based on the `.env.example` file and fill in your credentials.

```env
# Azure OpenAI
API_PROVIDER=azure
API_KEY=<your-api-key>
API_ENDPOINT=https://your-deplyoment.openai.azure.com/ # example
API_VERSION=2024-10-21 # example
API_MODEL=gpt-5-nano-2025-08-07 # example

OSW_USER=<user-id>@<bot-name>
OSW_PASSWORD=<bot-password>
```

## Run the Demo

### Basic Example

Construct a LaboratoryProcess entry based on a natural language description.

```bash
python demo_basic.py
```

You should not see any error in the console.
A link should be logged that points to the generated entry in the demo instance - open it in your browser to validate the results.

### Data Model Suggestion Example

Lookup or suggest a data model based on a natural language description.

```bash
python schema_catalog.py
```

### Iterative Agent Example
Run an agent that iteratively creates a LaboratoryProcess entry and linked entities (people, organizations, etc.) based on a natural language description.

```bash
python demo_iterative_agent.py
```


## Concept

### Simple iterative approach

define the tool `create_linked_entity` with the following steps:
1. Lookup a schema that matches the user request
2. Prompt the LLM to fill the schema properties while
   * ask for special treatment of properties with `range` annotation by re-entering (via recursive tool call) at step 1  for each such property
   * ask to skip calling `create_linked_entity` if the request is already stored in the global log in order to prevent endless loops (e.g. if Person -hasOrganization-> Organization -hasMember-> Person)
4. Compare the created entity with existing entities by using RAG + LLM judging. If a match is found, return the existing entity ID
5. Else return the created entity ID to be stored in the parent object

#### Challenges / Limitations
Despite prompt engineering efforts, LLMs still struggle with certain aspects of the task:
* LLMs often try complete all properties of a schema even if no information is provided, e.g. `deepseek-v3.2` creates keywords like "Keyword for Example Lab" when no keywords are provided in the description.
* Since properties with `range` annotation are of type `string` in the JSON-SCHEMA representation of OSL schemas, LLMs sometimes try to fill them with textual descriptions instead of looking up / creating the linked entity. 
* Placing step 4. directly after step 1 using the textual description instead of the already created JSON representation does not work well with some LLMs due to missing data normalization. E.g. `deepseek-v3.2`  struggle to compare dates in different formats (e.g. "01.01.2025" vs "2025-01-01") - using the ISO format (JSON-SCHEMA format `date`) on both sides helps here.
* Some LLMs have limitations regarding tool use (e.g. only tool use OR structured output) or JSON schema features (e.g. no nested objects).

#### Evaluation

The following models were evaluated for the simple iterative approach:

| Provider | Model | Result |
|----------|-------|--------|
| azure | gpt-5-nano-2025-08-07 | ✅ Basic works, interactive works |
| azure | gpt-5-mini-2025-08-07 | ✅ Basic works, interactive works |
| azure-foundry-anthropic | claude-sonnet-4-5 | ✅ Basic works, interactive works |
| azure-foundry-anthropic | claude-haiku-4-5 | ✅ Basic works, interactive works |
| azure-foundry | deepseek-v3.2-speciale | ❌ Error: too many requests |
| azure-foundry | deepseek-v3.2 | ⚠️ Basic works, iterative not: cannot compare ISO dates correctly (works with structured data compare) |
| azure-foundry | cohere-command-a | ⚠️ Basic works, iterative not: cannot compare ISO dates correctly |
| azure-foundry | deepseek-r1-0528 | ⚠️ Basic works, iterative partially, struggles with tool calling |
| azure-foundry | mistral-large-3 | ❌ Error: invalid input |
| azure-foundry | llama-3.3-70b-instruct | ⚠️ Basic works, interactive not: model doesn't support more than one tool call |
| azure-foundry | llama-4-maverick-17b-128e-instruct-fp8 | ⚠️ Basic works, interactive not: states entities are equal when they are not |
| azure-foundry | gpt-oss-120b | ❌ Error: tool_choice 'required' not supported |
| azure-foundry | kimi-k2-thinking | ✅ Basic works, interactive works |
| azure-foundry | phi-4-reasoning | ❌ Error: JSON schema contains unsupported xgrammar features |
| vllm | microsoft/Phi-4-reasoning | ⚠️ Basic works, interactive not: model refuses tool calling |
| vllm | microsoft/Phi-4-mini-instruct | ⚠️ Basic works, interactive not: model refuses tool calling |
| vllm | microsoft/Phi-4-mini-reasoning | ⚠️ Basic works, interactive not: model refuses tool calling |

### Advanced approach

1. lookup the schema that matches the request
2. Compare the request with previous requests stored in a global log via LLM judging. If a match is found, return the corresponding entity ID
3. Else prompt a LLM to provide a list of properties that can actually be filled based on the user request
4. remove all other properties from the schema
5. use the modified schema to prompt a LLM to create the entity via structured output, handling properties with `range` annotation as strings containing the descriptions of the linked entities
6. for each property with `range` annotation re-enter at step 1 with the description provided in step 4 to lookup / create the linked entity and the annotations of the corresponding property. Replace the textual description with the found / created entity ID.
7. Compare the created entity with existing entities by using RAG + LLM judging. If a match is found, return the existing entity ID
8. Store the created entity and return its ID

-----
### Segmentation approach

We need to build a generic agent that can operate with a large schema repo so we cannot give specific hint. Only our test scenario is specific about a LaboratoryProcess

Skip property selection was tested, but lead to poor relevance decision on which properties should be populated (often additional properties got populated with assumption and redudant information)

In a realistic usecase the agent will operate on a huge knowledge base so the initial RAG vector store lookup based just on the initial prompt may have a limited recall. As the availability of a vector store is not guranteed, we could also rely on a convectional query interface which cannot operate with the prompt but with the already structured json.

lookup_excact_matching_entity gets the json of the previously generated entity - so for a Person linked to a LaboratoryProcess it is only about the Person

------

create a oold_agent2.py which utilized a different strategy (reusing methods / subclassing OoldAgent of needed):

1. Segment the initial input by providing all available data models to the llms and ask for a set of data models that (a) can capture all provided information (b) are linked to each other (by following the 'range' annotation of properties. Optimization criteria: (i) use more specific properties (e.g. store an email in a 'email' property, not in a 'description' property) (ii) use a minimal number of schemas. Result is a structured list of entities

Variant A
class EntityDescription(BaseModel)
  schema: str
  "name of the selected data model"
  description: str
  "part of the initial prompt where this entity is the subject"

Variant B
class EntityPropertyDescription(BaseModel)
  schema: str
  "name of the selected data model"
  properties: dict[str, str]
  "a key-value map for the properties of the selected schema with literal values of descriptions of the linked entity"

Variant C
class EntityPropertyMap(BaseModel)
  id: str
  "within the current context unique id"
  schema: str
  "name of the selected data model"
  literal_properties: dict[str, str]
  "a key-value map for the properties of the selected schema with literal values"
  range_properties:
  "a key-value map for the range properties of the selected schema. values are ids of other entities in this context"

for now implement just this step and write a simple test based on the benchmark input prompts

--

Again property completion motivation is poor.
we need to go with Variant A and continue then as we did in OoldAgent but for all identified schemas: do another prompt to select fillable properties and their values. Then construct a ad-hoc schema for all identified entities

_extract_property_values needs to run for all entities in a single prompt (compare Variant C) so the LLM can link to other entities by their local id. But different to Variant C, a per entity schema is constructed with identified range properties mandatory.

---

create oold_agent3.py which utilizes the following strategy (use 3A, not 3B).
Take inspiration from oold_agent2.py and oold_agent.py for overlapping functionality
Dont use any specifics of the example data and schemas in the prompts - only the general task of creating linked entities based on natural language descriptions

Example input prompt:
"A tensile test experiment #1 conducted by Dr. Jane Doe, employed at Example Lab Corp.
A tensile test experiment #2 conducted by Dr. John Doe, employed at Example Lab Corp.
A tensile test experiment #3 conducted by Dr. John Doe (john.doe@example-lab.com)"


#1 Schema detection with original prompt provided as context
schema: (static JSON-SCHEMA)
  properties:
    entities:
      type: array
      items:
        type: object
        properties:
          id:
            type: string
          schema:
            type: string
          description:
            type: string
Example output:
entity_11: opensemantic.lab.v1.LaboratoryProcess "A tensile test experiment #1 conducted by Dr. Jane Doe"
entity_12: opensemantic.core.v1.Person "Dr. Jane Doe, employed at Example Lab Corp."
entity_13: opensemantic.base.v1.Organization "Example Lab Corp."
entity_21: opensemantic.lab.v1.LaboratoryProcess "A tensile test experiment #2 conducted by Dr. John Doe"
entity_22: opensemantic.core.v1.Person  "Dr. John Doe (john.doe@example-lab.com), employed at Example Lab Corp."
entity_31: opensemantic.lab.v1.LaboratoryProcess "A tensile test experiment #3 conducted by Dr. John Doe"

#2 Fillable Property definition: original prompt + result of step 1 provided as context
schema: (auto-generated JSON-SCHEMA)
  properties:
    fillable_properties:
      type: object
      properties:
         entity_11:
           type: array
           items:
             type: string
             enum: [list of all possible properties of the schema]
         entity_12:
           type: array
           ...
Example output:
entity_11: 'name', 'creator'
entity_12: 'first_name', 'surname', 'organization'

#3.B Property extraction, one-shot (all fillable properties + required properties in schema mandatory via dynamic JSON-SCHEMA generation!)
schema: (auto-generated JSON-SCHEMA)
  properties:
    entities:
      type: object
      properties:
         entity_11:
           type: object
           <opensemantic.lab.v1.LaboratoryProcess schema with non-fillable properties and non-required properties removed, e.g.:>
           properties:
             name: <definition from opensemantic.lab.v1.LaboratoryProcess.name>
             label: <definition from opensemantic.lab.v1.LaboratoryProcess.label>
             creator: <definition from opensemantic.lab.v1.LaboratoryProcess.creator>
         entity_12:
           type: object
             <opensemantic.core.v1.Person schema with non-fillable properties and non-required properties removed, ...>
Problem: range properties are still of type string here!

#3.A Property extraction like Variant C: all entities in one prompt, but with all fillable properties + required properties in schema mandatory via dynamic JSON-SCHEMA generation. Literal properties are of type string, range properties are of type string enum with the ids of the other entities in this context. Original prompt + result of step 1 provided as context.

schema: (auto-generated JSON-SCHEMA)
  properties:
    entities:
      type: object
      properties:
         entity_11:
           type: object
           properties:
             name:
               type: string
             label:
               type: string
             creator:
               type: string
               enum: [entity_12, entity_22, entity_32]
         entity_12:
           properties:
             first_name:
               type: string
             surname:
               type: string
             organization:
               type: string
               enum: [entity_13]
         ...

Example output:
entity_11: 
  name: "Exp 1"
  creator: entity_12
entity_12
  first_name: ...

#4 Entity construction: Replace readable ids with OSW-IDs. Per entity ask LLM to construct the final JSON based on the original schema (with all non-mandatory properties removed) and the filled properties from step 3.

#5 Deduplication with existing entities via RAG + LLM judging

# Validation / Benchmarking

entity_11: opensemantic.lab.v1.LaboratoryProcess ["name", "creator"]
entity_12: opensemantic.core.v1.Person ["first_name", "surname", "organization"]
entity_13: opensemantic.base.v1.Organization ["name"]
entity_21: opensemantic.lab.v1.LaboratoryProcess ["name", "creator"]
entity_22: opensemantic.core.v1.Person ["first_name", "surname", "organization", "email"]
entity_23: opensemantic.lab.v1.LaboratoryProcess ["name", "creator"]