# OSL ELN Demo

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

