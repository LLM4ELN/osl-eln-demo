from typing import Any, Dict, List
from pydantic import BaseModel
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


class SchemaClass(BaseModel):
    """Represents a single schema class with its metadata."""
    full_path: str
    """Full module path (e.g., 'opensemantic.core.v1.Entity')"""
    markdown: str
    """Markdown representation of the class"""
    subclass_of: List[str]
    """Complete list of all super classes, including super-super-... classes"""
    cls_obj: Any = None
    """Reference to the class object"""
    schema_uuid: str = ""
    """UUID from Config.schema_extra['uuid']"""
    schema_id: str = ""
    """First element of the type field default (e.g., 'Category:OSWxxx')"""

    class Config:
        arbitrary_types_allowed = True


class SchemaClassInventory(BaseModel):
    """Inventory of all schema classes."""
    items: Dict[str, SchemaClass]
    """Dictionary with full_path as key"""

    def get_all_full_paths(self) -> List[str]:
        """Returns a list of all full paths in the inventory."""
        return list(self.items.keys())

    def is_subclass_of(self, child_path: str, parent_path: str) -> bool:
        """Check if child_path is a subclass of parent_path."""
        if child_path not in self.items:
            return False
        return parent_path in self.items[child_path].subclass_of

    def get_class_hierarchy(self, full_path: str) -> List[str]:
        """Returns the complete class hierarchy for a given path."""
        if full_path not in self.items:
            return []
        return self.items[full_path].subclass_of

    def get_by_schema_id(self, schema_id: str) -> "SchemaClass | None":
        """Lookup a schema by its schema_id (e.g., 'Category:OSWxxx')."""
        for item in self.items.values():
            if item.schema_id == schema_id:
                return item
        return None

    def is_subclass_of_by_schema_id(
        self, child_id: str, parent_id: str
    ) -> bool:
        """Check subclass relation using schema_ids."""
        child = self.get_by_schema_id(child_id)
        parent = self.get_by_schema_id(parent_id)
        if not child or not parent:
            return False
        return parent.full_path in child.subclass_of


def _get_all_superclasses(cls: type) -> List[str]:
    """Recursively get all superclasses for a given class."""
    superclasses = []
    for base in cls.__mro__[1:]:  # Skip the class itself
        if base is object:
            continue
        base_path = f"{base.__module__}.{base.__name__}"
        base_path = base_path.replace("._model", "")
        superclasses.append(base_path)
    return superclasses


def _get_schema_metadata(cls: type) -> tuple[str, str]:
    """Extract schema_uuid and schema_id from a class.

    Returns (schema_uuid, schema_id)
    - schema_uuid: from Config.schema_extra['uuid']
    - schema_id: first element of the type field default (e.g., 'Category:OSWxxx')
    """
    schema_uuid = ""
    schema_id = ""

    # Get uuid from Config.schema_extra (v1) or model_config (v2)
    if hasattr(cls, 'Config') and hasattr(cls.Config, 'schema_extra'):
        schema_uuid = cls.Config.schema_extra.get('uuid', '')
    elif hasattr(cls, 'model_config'):
        extra = cls.model_config.get('json_schema_extra', {})
        if extra:
            schema_uuid = extra.get('uuid', '')

    # Get schema_id from type field default (v1: __fields__, v2: model_fields)
    fields = getattr(cls, '__fields__', None) or {}
    if not fields and hasattr(cls, 'model_fields'):
        fields = cls.model_fields
    if 'type' in fields:
        type_field = fields['type']
        default = getattr(type_field, 'default', None)
        if isinstance(default, list) and len(default) > 0:
            schema_id = default[0]
        elif isinstance(default, str):
            schema_id = default

    return schema_uuid, schema_id


def _is_schema_class(cls: type) -> bool:
    """Check if cls is an opensemantic Entity (v1 or v2)."""
    if not inspect.isclass(cls):
        return False
    v1_entity = opensemantic.core.v1._model.Entity
    if issubclass(cls, v1_entity):
        return True
    try:
        import opensemantic.core._model as _v2_core
        if hasattr(_v2_core, "Entity") and issubclass(cls, _v2_core.Entity):
            return True
    except ImportError:
        pass
    return False


def get_data_schema_markdown(
    cls: type,
    include_properties: bool = True,
    include_property_def: bool = True
) -> str:
    """Returns a markdown representation of a single data model class."""
    if not _is_schema_class(cls):
        return ""

    module_name = cls.__module__.replace('._model', '')
    doc = cls.__doc__ or ""
    markdown = f"## {module_name}.{cls.__name__}\n\n{doc}\n\n"

    for bc in cls.__bases__:
        bc_path = f"{bc.__module__}.{bc.__name__}"
        bc_path = bc_path.replace("._model", "")
        markdown += f"- Inherits from: `{bc_path}`\n"

    if not include_properties:
        markdown += "\n"
        return markdown

    fields = getattr(cls, "__fields__", {})
    if fields:
        markdown += "### Fields:\n\n"
        for field_name, field_info in fields.items():
            field_title = field_info.field_info.title or ""
            field_desc = field_info.field_info.description or ""
            data_type = str(field_info.annotation)
            data_type = data_type.replace("typing.", "")
            data_type = data_type.replace("._model", "")
            markdown += f"- **{field_name}**: *{data_type}*"
            if include_property_def:
                if field_title:
                    markdown += f" - {field_title}"
                if field_desc:
                    markdown += f" - {field_desc}"
            markdown += "\n"
        markdown += "\n"

    return markdown


def build_inventory(
    include_properties: bool = True,
    include_property_def: bool = True,
    extra_modules: list = None,
) -> SchemaClassInventory:
    """Build an inventory of all schema classes.

    Parameters
    ----------
    extra_modules
        Additional Python modules whose classes should be included in the
        inventory (e.g. ``[process_models]``).
    """
    items: Dict[str, SchemaClass] = {}

    modules = [
        opensemantic.core.v1._model,
        opensemantic.base.v1._model,
        opensemantic.lab.v1._model,
    ]
    if extra_modules:
        modules.extend(extra_modules)

    for module in modules:
        for name, obj in inspect.getmembers(module):
            if _is_schema_class(obj):
                module_name = module.__name__.replace('._model', '')
                full_path = f"{module_name}.{name}"
                markdown = get_data_schema_markdown(
                    obj, include_properties, include_property_def
                )
                subclass_of = _get_all_superclasses(obj)
                schema_uuid, schema_id = _get_schema_metadata(obj)

                items[full_path] = SchemaClass(
                    full_path=full_path,
                    markdown=markdown,
                    subclass_of=subclass_of,
                    cls_obj=obj,
                    schema_uuid=schema_uuid,
                    schema_id=schema_id
                )

    return SchemaClassInventory(items=items)


# Build a cached inventory for the enum
_cached_inventory: SchemaClassInventory | None = None


def get_cached_inventory() -> SchemaClassInventory:
    """Get the cached inventory, building it if necessary."""
    global _cached_inventory
    if _cached_inventory is None:
        _cached_inventory = build_inventory(
            include_properties=False, include_property_def=False
        )
    return _cached_inventory


def get_data_schema_inventory_markdown(
    include_properties: bool = True, include_property_def: bool = True
) -> str:
    """Returns a markdown list of all available data models in opensemantic."""
    inventory_obj = build_inventory(include_properties, include_property_def)
    result = "# Available Data Models\n\n"
    for schema_class in inventory_obj.items.values():
        result += schema_class.markdown
    return result


source_markdown = get_data_schema_inventory_markdown()


def suggest_existing_or_new_schema(prompt: str) -> str:
    """asks the llm what is the most suitable data model for the given task"""

    messages = [
        (
            "system",
            "You are a helpful assistant called 'LLM4ELN'."
            "Your purpose is to help users of electronic lab notebook "
            "to find the most suitable data model for their "
            "documentation task. "
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
    """Ask the LLM to select the most suitable data model for the given task.

    Uses an enum of valid schema paths to constrain LLM selection.
    Returns the full module path (e.g., 'opensemantic.core.v1.Entity')
    """
    inventory = get_cached_inventory()
    valid_paths = inventory.get_all_full_paths()

    # Build schema with enum constraint for valid paths
    schema_response_schema = {
        "title": "SchemaResponse",
        "description": (
            "Response schema for selecting the most suitable data model "
            "for a given documentation task."
        ),
        "type": "object",
        "properties": {
            "module_path": {
                "type": "string",
                "enum": valid_paths,
                "description": (
                    "Full import path to the schema class "
                    "(e.g., 'opensemantic.core.v1.Entity')"
                )
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of why this schema was chosen"
            }
        },
        "required": ["module_path", "explanation"],
        "additionalProperties": False
    }

    system_prompt = (
        "You are a helpful assistant called 'LLM4ELN'. "
        "Your purpose is to help users of electronic lab notebooks "
        "to find the most suitable data model for their "
        "documentation task. "
        "Analyze the user's request and select the most "
        "appropriate data model from the available schemas. "
        "You MUST select a module_path from the provided enum list. "
        "Also provide a brief explanation of why this schema was chosen.\n\n"
        "Available data models:\n\n"
        + get_data_schema_inventory_markdown(False, False)
    )

    llm = get_llm()
    if model_supports_structured_output(llm):
        effective_response_format = ProviderStrategy(
            schema=schema_response_schema,
            strict=True
        )
    else:
        # For models without native structured output, use a Pydantic model
        # with Literal type for the enum constraint
        from typing import Literal
        ValidPaths = Literal[tuple(valid_paths)]  # type: ignore

        class SchemaResponse(BaseModel):
            module_path: ValidPaths  # type: ignore
            explanation: str

        effective_response_format = ToolStrategy(
            schema=SchemaResponse.model_json_schema()
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
    module_path = result["module_path"]
    explanation = result["explanation"]
    print(f"Schema lookup: {module_path} - {explanation}")
    return module_path


if __name__ == "__main__":

    print("Data Schema Inventory:\n")
    inventory_md = get_data_schema_inventory_markdown()
    print(inventory_md)

    prompt = (
        "I want to document a chemical synthesis of aspirin including the "
        "reagents, quantities, reaction conditions, and safety precautions."
    )
    
    # requires > 100k context size
    # schema_suggestion = suggest_existing_or_new_schema(prompt)
    # print("Schema suggestion:\n", schema_suggestion)

    exact_schema = lookup_exact_schema(prompt)
    print("Exact schema lookup:\n", exact_schema)
