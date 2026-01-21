""" Initialize the language model and expose an llm-instance, if API-PROVIDER executed"""
from dotenv import load_dotenv
from os import environ
load_dotenv()


print('Using API provider:', environ.get("API_PROVIDER"))
if environ.get("API_PROVIDER") == "azure":
    # https://docs.langchain.com/oss/python/integrations/providers/microsoft
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_deployment=environ.get("API_MODEL"),  # or your deployment
        api_version=environ.get("API_VERSION"),  # or your api version
        api_key=environ.get("API_KEY"),  # or your api key
        azure_endpoint=environ.get("API_ENDPOINT")
    )

elif environ.get("API_PROVIDER") == "ollama":
    # https://docs.langchain.com/oss/python/integrations/chat/ollama
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=environ.get("API_MODEL"))

elif environ.get("API_PROVIDER") == "blablador":
    # https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=environ.get("API_MODEL"),
        api_key=environ.get("API_KEY"),
        base_url=environ.get("API_ENDPOINT")
    )

elif environ.get("API_PROVIDER") == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(api_key=environ.get("API_KEY"),
                    model=environ.get("API_MODEL"))

elif environ.get("API_PROVIDER") == "chatai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        api_key=environ.get("API_KEY"),
        openai_api_base=environ.get("API_ENDPOINT"),
        model=environ.get("API_MODEL")
    )

elif environ.get("API_PROVIDER") == "gemini":
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
    llm = ChatGoogleGenerativeAI(
        api_key=environ.get("API_KEY"),
        model=environ.get("API_MODEL")
    )

else:
    print("No API provider specified in .env file !!")