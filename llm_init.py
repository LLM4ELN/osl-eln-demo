from langchain.chat_models.base import BaseChatModel
from langchain.agents.factory import _supports_provider_strategy
from dotenv import load_dotenv
from os import environ
from oold.static import GenericLinkedBaseModel
from pydantic import BaseModel
from util import modify_schema

load_dotenv()


def get_llm():
    """initialize and return a new language model object
    based on environment variables"""

    if environ.get("API_PROVIDER") == "azure":
        # https://docs.langchain.com/oss/python/integrations/providers/microsoft
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_deployment=environ.get("API_MODEL"),  # or your deployment
            api_version=environ.get("API_VERSION"),  # or your api version
            api_key=environ.get("API_KEY"),  # or your api key
            azure_endpoint=environ.get("API_ENDPOINT")
        )

    if environ.get("API_PROVIDER") == "azure-foundry-anthropic":
        # Azure AI Foundry with Claude Sonnet
        # https://learn.microsoft.com/en-us/azure/ai-studio/
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=environ.get("API_MODEL"),  # e.g., claude-3-5-sonnet-20241022
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT"),
            default_headers={
                "x-ms-api-version": environ.get("API_VERSION", "2024-02-15")
            }
        )
        
    # azure-foundry-mistral
    if environ.get("API_PROVIDER") == "azure-foundry-mistral":
        # Azure AI Foundry with Mistral
        from langchain_mistralai.chat_models import ChatMistralAI
        llm = ChatMistralAI(
            model=environ.get("API_MODEL"),  # e.g., mistral-large-3
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT"),
        )    

    if environ.get("API_PROVIDER") == "azure-foundry":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=environ.get("API_MODEL"),  # e.g., deepseek-1.0
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT"),
            default_headers={
                "x-ms-api-version": environ.get("API_VERSION", "2024-02-15")
            }
        )
        # AzureAIChatCompletionsModel is alpha/beta
        # from langchain_azure_ai.chat_models import (
        #     AzureAIChatCompletionsModel
        # )
        # llm = AzureAIChatCompletionsModel(
        #     model=environ.get("API_MODEL"),
        #     credential=environ.get("API_KEY"),
        #     endpoint=environ.get("API_ENDPOINT"),
        #     temperature=0.0,
        #     api_version=environ.get("API_VERSION", "2024-02-15")
        # )

    if environ.get("API_PROVIDER") == "ollama":
        # https://docs.langchain.com/oss/python/integrations/chat/ollama
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=environ.get("API_MODEL"))

    if environ.get("API_PROVIDER") == "llama-cpp":
        # https://docs.langchain.com/oss/python/integrations/chat/llama_cpp
        from langchain_community.chat_models import ChatLlamaCpp
        llm = ChatLlamaCpp(
            model_path=environ.get("API_MODEL"),
        )

    if environ.get("API_PROVIDER") == "openai-compatible-api":
        # Generic OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=environ.get("API_MODEL"),
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT")
        )

    if environ.get("API_PROVIDER") == "blablador":
        # https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=environ.get("API_MODEL"),
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT")
        )

    if environ.get("API_PROVIDER") == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key=environ.get("API_KEY"))

    if environ.get("API_PROVIDER") == "chatai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=environ.get("API_KEY"),
            openai_api_base=environ.get("API_ENDPOINT"),
            model=environ.get("API_MODEL")
        )

    if environ.get("API_PROVIDER") == "vllm":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=environ.get("API_KEY"),
            base_url=environ.get("API_ENDPOINT"),
            model=environ.get("API_MODEL")
        )

    if environ.get("API_PROVIDER") == "gemini":
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
        llm = ChatGoogleGenerativeAI(
            api_key=environ.get("API_KEY"),
            model=environ.get("API_MODEL")
        )

    model_name = environ.get("API_MODEL")

    provider = environ.get("API_PROVIDER", "").lower()

    supports_structured_output = False
    if provider == "vllm":
        supports_structured_output = True

    if model_name is None or model_name == "":
        model_name = (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or getattr(llm, "model_id", "")
        )

    if llm.profile is None:
        llm.profile = {}
    if "structured_output" not in llm.profile:
        llm.profile[
            "structured_output"
        ] = supports_structured_output

    return llm


def model_supports_structured_output(llm: BaseChatModel, tools=None):
    """Check if the LLM model supports structured output"""
    if (
        hasattr(llm, "model_name")
        and llm.model_name in ["gpt-oss-120b", "mistral-large-3"]
    ):
        return False
    return _supports_provider_strategy(llm, tools)


def get_response_format(
    llm: BaseChatModel,
    target_data_model: GenericLinkedBaseModel | BaseModel | dict,
    tools=None
):
    """Get the appropriate response format (ProviderStrategy or ToolStrategy)
    based on whether the model supports structured output.

    Args:
        llm: The language model
        target_data_model: Either a BaseModel class, GenericLinkedBaseModel,
            or a JSON schema dict directly
        tools: Optional tools list
    """
    from langchain.agents.structured_output import (
        ProviderStrategy,
        ToolStrategy
    )

    # Handle dict (already a JSON schema)
    if isinstance(target_data_model, dict):
        target_schema = target_data_model
    elif issubclass(target_data_model, GenericLinkedBaseModel):
        target_schema = target_data_model.export_schema()
        target_schema = modify_schema(target_schema)
    else:
        target_schema = target_data_model.model_json_schema()

    if model_supports_structured_output(llm, tools):
        # Model supports provider strategy - use it
        return ProviderStrategy(
            schema=target_schema,
            strict=True
        )
    else:
        # Model doesn't support provider strategy - use ToolStrategy
        return ToolStrategy(
            schema=target_schema
        )

if __name__ == "__main__":
    llm = get_llm()
    print("LLM initialized:", llm)
    result = llm.invoke("Hello, world!")
    print("LLM invocation result:", result)
