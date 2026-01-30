import json
from dotenv import load_dotenv
from os import environ
from osw.express import OswExpress, CredentialManager, OSW
import osw.model.entity as model
from pydantic import BaseModel
from langchain.agents import create_agent

from llm_init import get_response_format

load_dotenv()


def get_osl_client():
    cred_mngr = CredentialManager()
    cred_mngr.add_credential(CredentialManager.UserPwdCredential(
        iri="llm4eln.semos.dev",
        username=environ.get("OSW_USER"),
        password=environ.get("OSW_PASSWORD"),
    ))

    osl_client = OswExpress(
        domain="llm4eln.semos.dev",
        cred_mngr=cred_mngr,
    )
    return osl_client


# default osl_client instance
osl_client = get_osl_client()


def search_by_label(osl_client: OswExpress, query: str):
    result = osl_client.site.semantic_search(
        "[[HasLabel::~*" + query + "*]]"
    )
    # this triggers a schema fetch & build
    # entities = osl_client.load_entity(result)
    entities = osl_client.load_entity(OSW.LoadEntityParam(
        titles=result,
        autofetch_schema=False,
        model_to_use=model.Entity
    ))
    # results = osl_client.site._site.search(search=query)
    # for res in results:
    #     print(res)
    # return results

    # const encodedQuery = encodeURIComponent(query);
    # let url = mw.config.get("wgScriptPath");
    # url += `/index.php?title=Special:Search&limit=100&offset=0
    # &profile=default&search=${encodedQuery}`;
    # const response = await fetch(url);

    # url = osl_client.site._site.scheme + "://"
    # + osl_client.site._site.host + osl_client.site._site.path
    # + "/index.php"
    # url += f"?title=Special:Search&limit=100&offset=0
    # &profile=default&search={query}"
    # print("Search URL:", url)
    # result = osl_client.site._site.connection.get(url)
    # print("Search Response Status Code:", result.status_code)
    # print("Search Response Text:", result.text)

    # print(entities)
    return entities


def search_by_category(osl_client: OswExpress, category: str):
    result = osl_client.site.semantic_search(
        f"[[Category:{category}]]"
    )
    entities = osl_client.load_entity(OSW.LoadEntityParam(
        titles=result,
        autofetch_schema=False,
        model_to_use=model.Entity
    ))
    return entities


def get_all_pages(osl_client: OswExpress):
    result = []
    for namespace in [
        # ":Category",
        "Item",
        # "Property",
        # "File",
    ]:
        print(f"Fetching all pages in namespace {namespace}...")
        _result = osl_client.site.semantic_search(
            f"[[{namespace}:+]][[HasOswId::!~*#*]]|limit=10000"
        )
        print(f"   ...found {len(_result)} pages so far.")
        result.extend(_result)
    # result = osl_client.site.semantic_search(
    #     "[[:Category:+||Item:+||Property:+||File:+]]|limit=10000"
    #     #"OR [[Item:+]] "
    #     #"OR [[Property:+]]"
    # )
    # filter titles containing "#" (subobjects)
    result = [title for title in result if "#" not in title]
    return result


def build_vector_store():
    osl_client = get_osl_client()
    # search_by_label(osl_client, "PCR")

    all_titles = get_all_pages(osl_client)
    print(f"Total pages found: {len(all_titles)}")
    # print all titles
    # for title in all_titles:
    #    print(title)

    # load all entities
    # entities = osl_client.load_entity(OSW.LoadEntityParam(
    #     titles=all_titles,
    #     autofetch_schema=False,
    #     model_to_use=model.Entity
    # ))

    # load all pages
    from osw.wtsite import WtSite
    pages = osl_client.site.get_page(
        WtSite.GetPageParam(
            titles=all_titles
        )
    ).pages

    # create Documents
    from langchain_core.documents import Document
    documents = []
    for page in pages:
        doc = Document(
            id=page.title,
            page_content=json.dumps(page._slots),
            metadata={
                "name": (
                    page.get_slot_content("jsondata").get("name", "Unknown")
                ),
                "type": (
                    page.get_slot_content("jsondata").get("type", "Unknown")
                ),
                "url": page.get_url(),
            }
        )
        documents.append(doc)

    from rag_init import get_vector_store

    vector_store = get_vector_store()

    # add documents to vector store
    print(f"Adding {len(documents)} documents to vector store...")
    vector_store.add_documents(documents=documents)

    return vector_store


def lookup_excact_matching_entity(
    vector_store, description, llm_judge=False, debug=False
) -> str | None:
    """lookup an entity by its description using the vector store
    and return the entity's title / ID if a good match is found.
    """
    # perform a similarity search
    results = vector_store.similarity_search_with_score(
        description, k=5
    )
    print(f"\n\nLookup description: {description}")
    if debug:
        for res, score in results:
            print(
                f"- Document ID: {res.id}, Score: {score} \n"
                f"  Metadata: {res.metadata}\n"
                f"  Data: {res.page_content}\n"
            )

    # if llm_judge is True, use LLM to judge the best match
    if llm_judge:
        from llm_init import get_llm

        class ResultSchema(BaseModel):
            osw_id: str
            """the OSW-ID of the best matching entity,
            or empty '' if no good match is found"""
            explanation: str
            """explanation of the decision"""

        system_prompt = (
            "Check if one of the candidate entities matches the given "
            "description exactly by comparing all fields. "
            "For free text fields, consider minor variations in wording "
            "as matches. "
            "Structured fields like ids, dates, enums, have to match "
            "exactly. "
            "Go over each candidate entity and compare its data to the "
            "description. "
            "Side by side compare each field and decide if it matches "
            "the description. "
            "If all match, return the matching entity's OSW-ID "
            "(e.g. Item:OSW123..). "
            "Return an empty OSW-ID if no good match was found."
            "Respond in valid JSON according to the schema: "
            "{'osw_id': str, 'explanation': str}"
        )

        candidates_str = "\n".join([
            f"- OSW-ID: {res.id}, Data: {res.page_content}"
            for res, score in results
        ])

        user_prompt = (
            f"Description of the entity:\n"
            f"{description}\n\n"
            f"Candidate entities with their metadata:\n"
            f"{candidates_str}"
        )

        llm = get_llm()
        # if hasattr(llm, "reasoning_effort"):
        #     llm.reasoning_effort = "high"

        response_format = get_response_format(
            llm, target_data_model=ResultSchema
        )
        agent = create_agent(
            model=llm,
            response_format=response_format,
        )

        response = agent.invoke({"messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]})["structured_response"]
        print(f"LLM Judge Response: {response}")
        response = ResultSchema.model_validate(response)
        if not response.osw_id.startswith("Item:OSW"):
            return None
        else:
            return response.osw_id
    else:
        # return the best match if score is above a threshold
        best_res, best_score = results[0]
        if best_score > 0.4:  # arbitrary threshold
            return best_res.id
        else:
            return None


if __name__ == "__main__":
    vector_store = build_vector_store()

    # # query vector store
    # def query(q):
    #     results = vector_store.similarity_search_with_score(
    #         q, k=5
    #     )
    #     print(f"\n\nQuery: {q}")
    #     for res, score in results:
    #          print(f"- Document ID: {res.id}, Score: {score}, "
    #                f"Metadata: {res.metadata}")

    # query("Dr. John Doe")
    # query("Example Lab")
    # query("Person Dr. John Doe")
    # query("Organization Example Lab")
    # query("Dr. Jane Smith")

    res1 = lookup_excact_matching_entity(
        vector_store=vector_store,
        description=(
            "A laboratory process to document an experiment "
            "created by Dr. John Doe, Example Lab, "
            "starting at 01.01.2025 and ending at 02.01.2025, "
            "status in progress."
        ),
        llm_judge=False
    )
    print(f"Lookup result without LLM judge: {res1}")

    res2 = lookup_excact_matching_entity(
        vector_store=vector_store,
        description=(
            "A laboratory process to document an experiment "
            "created by Dr. John Doe, Example Lab, "
            # "starting at 01.01.2025 and ending at 02.01.2025, "
            "starting 2025-01-01 and ending 2025-01-02, "
            "status in progress."
        ),
        llm_judge=True
    )
    print(f"Lookup result with LLM judge: {res2}")

    res3 = lookup_excact_matching_entity(
        vector_store=vector_store,
        description=(
            "A laboratory process to document an experiment "
            "created by Dr. John Doe, Example Lab, "
            # "starting at 01.03.2025 and ending at 02.03.2025, "
            "starting 2025-03-01 and ending 2025-03-02, "
            "status in progress."
        ),
        llm_judge=True
    )
    print(f"Lookup result with LLM judge: {res3}")
