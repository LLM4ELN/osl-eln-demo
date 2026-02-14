import json
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
from os import environ
from osw.express import OswExpress, CredentialManager, OSW
import osw.model.entity as model
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from llm_init import get_response_format

load_dotenv()


class LookupResult(BaseModel):
    """Result from entity lookup in vector store."""
    osw_id: str = ""
    """The OSW-ID of the matching entity, or empty string if no match."""
    needs_update: bool = False
    """True if the new description has additional info to merge."""
    existing_data: Dict[str, Any] = Field(default_factory=dict)
    """Existing entity data from vector store (for merge in OoldAgent)."""
    existing_type: str = ""
    """The type/schema of the existing entity."""
    explanation: str = ""
    """Explanation of the decision."""


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
                    page.get_slot_content("jsondata").get("name", "")
                ),
                "type": (
                    page.get_slot_content("jsondata").get("type", "")
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
) -> LookupResult | None:
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

    # return if no results
    if len(results) == 0:
        print("No results to process.")
        return None
    # if llm_judge is True, use LLM to judge the best match
    if llm_judge:
        from llm_init import get_llm

        # ========== STEP 1: Match Decision ==========
        class MatchDecision(BaseModel):
            """Result of entity matching."""
            decision: Literal["no_match", "match", "match_with_update"]
            osw_id: str = ""  # Empty for no_match
            explanation: str = ""

        step1_prompt = (
            "Check if one of the candidate entities matches the given "
            "description. Return:\n"
            "- 'no_match': No candidate matches the description\n"
            "- 'match': A candidate matches exactly (no new info)\n"
            "- 'match_with_update': A candidate matches but the description "
            "has additional info (new fields, relationships, attributes)\n\n"
            "Include the osw_id (e.g., 'Item:OSWxxx') for match/match_with_update. "
            "Provide a brief explanation for your decision in the 'explanation' field. "
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

        # Serialize schema into prompt so grammar-driven engines
        # (vLLM, llama.cpp) see description/maxLength annotations
        schema = MatchDecision.model_json_schema()
        if environ.get("LIMIT_STR_FIELDS", "false") == "true":
            props = schema.get("properties", {})
            if "explanation" in props:
                props["explanation"]["maxLength"] = 500
        schema_str = json.dumps(schema, indent=2)
        user_prompt += f"\n\n## Output Schema\n\n{schema_str}"

        llm = get_llm()
        response_format = get_response_format(llm, target_data_model=MatchDecision)
        agent = create_agent(model=llm, response_format=response_format)

        step1_response = agent.invoke({"messages": [
            {"role": "system", "content": step1_prompt},
            {"role": "user", "content": user_prompt}
        ]})["structured_response"]

        decision = MatchDecision.model_validate(step1_response)
        print(f"Step 1 - Match Decision: {decision}")

        if decision.decision == "no_match":
            return None

        if decision.decision == "match":
            return LookupResult(
                osw_id=decision.osw_id,
                needs_update=False,
                existing_data={},
                existing_type="",
                explanation=decision.explanation
            )

        # ========== match_with_update: Return existing entity data ==========
        # Merge will be handled by OoldAgent using schema cache
        print(f"Match with update for {decision.osw_id} - returning existing data")

        # Get existing entity data
        existing_data = None
        existing_type = ""
        for res, score in results:
            if res.id == decision.osw_id:
                content = json.loads(res.page_content)
                existing_data = content.get("jsondata", content)
                # Get type from the entity data
                type_list = existing_data.get("type", [])
                if type_list:
                    existing_type = type_list[0] if isinstance(type_list, list) else type_list
                break

        if existing_data is None:
            print(f"Warning: Could not find existing data for {decision.osw_id}")
            return LookupResult(
                osw_id=decision.osw_id,
                needs_update=False,
                existing_data={},
                existing_type="",
                explanation="Could not retrieve existing data"
            )

        print(f"   Existing type: {existing_type}")

        return LookupResult(
            osw_id=decision.osw_id,
            needs_update=True,
            existing_data=existing_data,
            existing_type=existing_type,
            explanation=decision.explanation
        )
    else:
        # return the best match if score is above a threshold
        best_res, best_score = results[0]
        if best_score > 0.4:  # arbitrary threshold
            return LookupResult(
                osw_id=best_res.id,
                needs_update=False,
                existing_data={},
                existing_type="",
                explanation="Score-based match"
            )
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
