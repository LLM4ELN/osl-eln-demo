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


