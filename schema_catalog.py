from llm_init import get_llm
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


def suggest_existing_or_new_schema(prompt: str) -> str:
    """asks the llm what is the most suitable data model for the given task"""

    messages = [
        (
            "system",
            "You are a helpful assistant called 'LLM4ELN'."
            "Your purpose is to help users of electronic lab notebook."
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
    """ask the llm what is the most suitable data model for the given task"""

    messages = [
        (
            "system",
            "You are a helpful assistant called 'LLM4ELN'."
            "Your purpose is to help users of electronic lab notebook."
            "Answer only with the exact full import path."
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


if __name__ == "__main__":
    prompt = (
        "I want to document a chemical synthesis of aspirin including the "
        "reagents, quantities, reaction conditions, and safety precautions."
    )
    schema_suggestion = suggest_existing_or_new_schema(prompt)
    print("Schema suggestion:\n", schema_suggestion)
    
    exact_schema = lookup_exact_schema(prompt)
    print("Exact schema lookup:\n", exact_schema)