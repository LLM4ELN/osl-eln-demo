from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from llm_init import get_llm, model_supports_structured_output
import opensemantic.core.v1._model
import opensemantic.base.v1._model
import opensemantic.lab.v1._model

# load the source code of the opensemantic.core.v1 module
import inspect
source_code = "##### opensemantic.core.v1 #####\n\n"
source_code += inspect.getsource(opensemantic.core.v1._model)
source_code += "\n\n##### opensemantic.base.v1 #####\n\n"
source_code += inspect.getsource(opensemantic.base.v1._model)
source_code += "\n\n##### opensemantic.lab.v1 #####\n\n"
source_code += inspect.getsource(opensemantic.lab.v1._model)



def get_data_schema_inventory_markdown(include_properties=True, include_property_def=True) -> str:
    """returns a markdown list of available data models in opensemantic"""
    
    root_class = opensemantic.core.v1._model.Entity

    inventory = "# Available Data Models\n\n"
    for module in [
        opensemantic.core.v1._model,
        opensemantic.base.v1._model,
        opensemantic.lab.v1._model
    ]:
        for name, obj in inspect.getmembers(module):
            obj: type[opensemantic.core.v1._model.Entity]
            if inspect.isclass(obj) and issubclass(obj, root_class):
                doc = obj.__doc__ or ""
                inventory += f"## {module.__name__.replace('._model', '')}.{name}\n\n{doc}\n\n"
                for bc in obj.__bases__:
                    inventory += f"- Inherits from: `{bc.__module__}.{bc.__name__}`\n".replace("._model", "")
                if not include_properties:
                    inventory += "\n"
                    continue
                fields = getattr(obj, "__fields__", {})
                if fields:
                    inventory += "### Fields:\n\n"
                    for field_name, field_info in fields.items():
                        field_title = field_info.field_info.title or ""
                        field_desc = field_info.field_info.description or ""
                        data_type = str(field_info.annotation).replace("typing.", "").replace("._model", "")
                        # assembly <name>: <type> | <title> - <description>
                        inventory += f"- **{field_name}**: *{data_type}*"
                        if include_property_def:
                            if field_title:
                                inventory += f" - {field_title}"
                            if field_desc:
                                inventory += f" - {field_desc}"
                        inventory += "\n"
                    inventory += "\n"
    return inventory

source_markdown = get_data_schema_inventory_markdown()


def suggest_existing_or_new_schema(prompt: str) -> str:
    """asks the llm what is the most suitable data model for the given task"""

    messages = [
        (
            "system",
            "You are a helpful assistant called 'LLM4ELN'."
            "Your purpose is to help users of electronic lab notebook "
            "to find the most suitable data model for their documentation task. "
            "If the existing data models are not sufficient, "
            "extend them by adding a new class."
            "Attached is the source code of data models you can choose from:"
            "\n\n" + source_code
        ),
        (
            "human", prompt
        ),
    ]
    llm = get_llm()
    ai_reply = llm.invoke(messages)
    if isinstance(ai_reply, str):
        ai_msg = ai_reply
    else:
        ai_msg = ai_reply.content
    return ai_msg


def lookup_exact_schema(prompt: str) -> str:
    """ask the llm what is the most suitable data model for the given task

    Returns the full module path (e.g., 'opensemantic.core.v1.Entity')
    """

    class SchemaResponse(BaseModel):
        # property with regex
        module_path: str = Field(..., pattern=r"^opensemantic\.[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
        """Full import path to the schema class
        (e.g., 'opensemantic.core.v1.Entity')"""
        explanation: str
        """Brief explanation of why this schema was chosen"""

    system_prompt = (
        "You are a helpful assistant called 'LLM4ELN'. "
        "Your purpose is to help users of electronic lab notebooks "
        "to find the most suitable data model for their documentation task. "
        "Analyze the user's request and select the most appropriate data "
        "model from the available schemas. "
        "Reply only with a JSON object containing the fields `module_path` and `explanation`. "
        "Return the exact full import path `module_path` to the Python class, "
        "e.g., 'opensemantic.<submodule>.v1.<classname>'. "
        "Also provide a brief `explanation` of why this schema was chosen. "
        "Available data models:\n\n" + get_data_schema_inventory_markdown(False, False)
    )
    
    llm = get_llm()
    if hasattr(llm, "reasoning_effort"):
        llm.reasoning_effort = "high"
    if model_supports_structured_output(llm):
        # Model supports provider strategy - use it
        effective_response_format = ProviderStrategy(
            schema=SchemaResponse.model_json_schema(),
            strict=True
        )
    else:
        # Model doesn't support provider strategy - use ToolStrategy
        effective_response_format = ToolStrategy(
            schema=SchemaResponse
        )

    agent = create_agent(
        model=llm,
        response_format=effective_response_format,
    )

    response = agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    })

    result = response["structured_response"]
    result = SchemaResponse.model_validate(result)
    print(f"Schema lookup: {result.module_path} - {result.explanation}")
    return result.module_path


if __name__ == "__main__":
    
    print("Data Schema Inventory:\n")
    inventory_md = get_data_schema_inventory_markdown()
    print(inventory_md)
    
    prompt = (
        "I want to document a chemical synthesis of aspirin including the "
        "reagents, quantities, reaction conditions, and safety precautions."
    )
    schema_suggestion = suggest_existing_or_new_schema(prompt)
    print("Schema suggestion:\n", schema_suggestion)

    exact_schema = lookup_exact_schema(prompt)
    print("Exact schema lookup:\n", exact_schema)
