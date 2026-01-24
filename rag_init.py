import json
from dotenv import load_dotenv
from os import environ
from langchain_core.documents import Document

load_dotenv()


def get_embedding():
    """initialize and return a new language model object
    based on environment variables"""

    if environ.get("API_PROVIDER") == "azure":
        # https://docs.langchain.com/oss/python/integrations/providers/microsoft
        from langchain_openai import AzureOpenAIEmbeddings
        embedding = AzureOpenAIEmbeddings(
            # or your deployment
            azure_deployment=environ.get("API_EMBEDDING_MODEL"),
            api_version=environ.get("API_VERSION"),  # or your api version
            api_key=environ.get("API_KEY"),  # or your api key
            azure_endpoint=environ.get("API_ENDPOINT")
        )
    return embedding


def get_vector_store():
    """initialize and return a vector store instance"""
    embedding = get_embedding()

    from langchain_core.vectorstores import InMemoryVectorStore
    vector_store = InMemoryVectorStore(embedding=embedding)
    return vector_store


if __name__ == "__main__":

    vector_store = get_vector_store()

    json_docs = [
        {
            "uuid": "asdf-1234",
            "type": "LaboratoryProcess",
            "label": "PCR",
            "description": "This is a sample document about PCR."
        },
        {
            "uuid": "qwer-5678",
            "type": "Article",
            "label": "Gel Electrophoresis",
            "description": (
                "This document discusses gel electrophoresis techniques."
            )
        }
    ]

    # generate langchain documents and add to vector store
    for doc in json_docs:
        print(f"Adding document {doc['uuid']} with label {doc['label']}")
        ldoc = Document(
            id=doc["uuid"],
            page_content=json.dumps(doc),
            metadata={
                "type": doc["type"]
            }
        )
        vector_store.add_documents(documents=[ldoc])

    # perform a similarity search
    def query(query):
        results = vector_store.similarity_search_with_score(
            query=query,
            k=2
        )
        print(f"Results for query '{query}':")
        # print data including the similarity score
        for (res, score) in results:
            print(
                f"- Document ID: {res.id}, Score: {score}, "
                f"Content: {res.page_content}, Metadata: {res.metadata}"
            )

    query("PCR techniques")
    query("Electrophoresis methods")
    query("Elephant")
