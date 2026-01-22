def is_object(item):
    return item is not None and isinstance(item, dict)


def is_array(item):
    return isinstance(item, list)


def is_string(item):
    return isinstance(item, str)


def is_integer(item):
    return isinstance(item, int)


def is_number(item):
    try:
        float(item)
        return True
    except ValueError:
        return False


def deep_copy(obj):
    import copy  # noqa: E402
    return copy.deepcopy(obj)


def deep_equal(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(
            deep_equal(x[key], y[key]) for key in x
        )
    return x == y


def unique_array(array):
    result = []
    for item in array:
        add = True
        for added_item in result:
            if deep_equal(added_item, item):
                add = False
                break
        if add:
            result.append(item)
    return result


def merge_deep(target, source):
    if not target:
        return source
    if not source:
        return target
    output = {}
    output.update(target)
    if is_object(target) and is_object(source):
        for key in source:
            if is_array(source[key]) and is_array(target.get(key)):
                if key not in target:
                    output[key] = source[key]
                else:
                    output[key] = unique_array(target[key] + source[key])
            elif is_object(source[key]):
                if key not in target:
                    output[key] = source[key]
                else:
                    output[key] = merge_deep(target[key], source[key])
            else:
                output[key] = source[key]
    return output


def merge_all_of(schema):
    """The most specific schema is on the root level"""
    if isinstance(schema, dict):
        merged_schema = {}

        for key, value in schema.items():
            if key == "allOf":
                pass  # handled later

            # if key == "$ref":
            #    pass # not implemented

            elif isinstance(value, dict):
                merged_schema[key] = merge_all_of(value)
            elif isinstance(value, list):
                merged_schema[key] = [
                    merge_all_of(item)
                    if isinstance(item, dict) else item for item in value
                ]
            else:
                merged_schema[key] = value

            if key == "oneOf":
                merged_schema["anyOf"] = merged_schema.pop("oneOf")

        if "allOf" in schema:
            for super_schema in schema["allOf"]:
                # process it first
                super_schema = merge_all_of(super_schema)
                # print("merge", sub_schema, " with ", merged_schema)
                # then merge our schema over the super_schema
                merged_schema = merge_deep(super_schema, merged_schema)

            return merged_schema
    return schema


def modify_schema(schema, seen_refs=None, root=None):
    """makes jsonschema OpenAI conform,
    see https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
    Interate over a JsonSchema recursively.
    add the key additionalProperties: false to every type object schema.
    for each object properties which is not required add the union type with null,
    e.g. type: [string, null]. finally, make all properties required.
    """  # noqa: E501

    # fix: rename "defintions" to "$defs"
    if isinstance(schema, dict) and "definitions" in schema:
        schema["$defs"] = schema.pop("definitions")

    root_level = False
    if root is None:
        root = schema
        root_level = True

    if seen_refs is None:
        seen_refs = {}

    schema = merge_all_of(schema)
    if isinstance(schema, dict) and schema.get("type") is not None:

        rm_keys = [
            '@context',
            'default*',
            # 'defaultProperties',
            'description*',
            'eval_template',
            'headerTemplate',
            # 'options',
            'propertyOrder',
            # 'range',
            # 'template',
            'titel*',
            'title*',
            'uuid',
            'watch',
            # 'x-enum-descriptions',
            # 'x-enum-varnames',
            # 'x-oold-required-iri',
            'uniqueItems',
        ]
        for rk in rm_keys:
            if rk in schema:
                schema.pop(rk)

        if "object" in schema.get("type"):
            schema["additionalProperties"] = False
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for prop, prop_schema in properties.items():
                if prop not in required:
                    if "type" in prop_schema:
                        if isinstance(prop_schema["type"], list):
                            if "null" not in prop_schema["type"]:
                                prop_schema["type"].append("null")
                        else:
                            prop_schema["type"] = [prop_schema["type"], "null"]
                    else:
                        prop_schema["type"] = ["null"]
                # Recursively modify nested objects
                schema["properties"][prop] = modify_schema(
                    prop_schema, root=root, seen_refs=seen_refs
                )
            schema["required"] = list(properties.keys())
        elif "array" in schema.get("type"):
            items = schema.get("items")
            if items:
                schema["items"] = modify_schema(
                    items, root=root, seen_refs=seen_refs
                )
    elif isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = modify_schema(item, root=root, seen_refs=seen_refs)

    # #handle $defs
    # if isinstance(schema, dict) and "$defs" in schema:
    #     for def_key, def_schema in schema["$defs"].items():
    #         schema["$defs"][def_key] = modify_schema(
    #             def_schema, seen_refs=seen_refs
    #         )

    # replace $refs from root $defs
    ref_whitelist = ["Label", "LangCode", "Description"]
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/$defs/"):
            def_key = ref[len("#/$defs/"):]
            if root and "$defs" in root and def_key in root["$defs"]:
                if ref in seen_refs and def_key not in ref_whitelist:
                    # print(f"Warning: Circular $ref detected for {ref}")
                    schema.pop("$ref", None)
                    schema["type"] = "null"
                else:
                    seen_refs[ref] = True
                    schema.update(modify_schema(
                        root["$defs"][def_key], root=root, seen_refs=seen_refs
                    ))
                    schema.pop("$ref", None)
            else:

                # if ref name matches root title, replace with root schema
                if root and "title" in root and def_key == root["title"]:
                    schema["$ref"] = "#"
                else:
                    print(f"Warning: $ref {ref} not found in root $defs")

    if isinstance(schema, dict):
        if "anyOf" in schema:
            for i, subschema in enumerate(schema["anyOf"]):
                schema["anyOf"][i] = modify_schema(
                    subschema, root=root, seen_refs=seen_refs
                )
        if "format" in schema:
            if schema["format"] in [
                "uri", "uri-reference", "iri", "iri-reference"
            ]:
                schema.pop("format")

    if root_level:
        # finally, remove $defs from root
        if "$defs" in schema:
            schema.pop("$defs")
            # for def_key in list(schema["$defs"].keys()):
            #     if def_key in ref_whitelist:
            #         continue
            #     schema["$defs"].pop(def_key)
            # for def_key, def_schema in schema["$defs"].items():
            #     schema["$defs"][def_key] = modify_schema(
            #         def_schema, seen_refs=seen_refs
            #     )
    return schema


def remove_nulls(d):
    """Remove null values from nested dicts"""
    if isinstance(d, dict):
        return {k: remove_nulls(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_nulls(i) for i in d if i is not None]
    else:
        return d


def remove_empty_strings(d):
    """Remove empty string values from nested dicts"""
    if isinstance(d, dict):
        return {k: remove_empty_strings(v) for k, v in d.items() if v != ""}
    elif isinstance(d, list):
        return [remove_empty_strings(i) for i in d if i != ""]
    else:
        return d


def remove_auto_defined_fields(d):
    """Remove auto-defined fields like 'uuid' and 'type' from nested dicts"""
    auto_defined_fields = ["uuid", "type"]
    if isinstance(d, dict):
        return {
            k: remove_auto_defined_fields(v)
            for k, v in d.items() if k not in auto_defined_fields
        }
    elif isinstance(d, list):
        return [remove_auto_defined_fields(i) for i in d]
    else:
        return d


def post_process_llm_json_response(response_json):
    """Post-process LLM JSON response by
    removing null values, empty strings, and auto-defined fields"""
    cleaned_response = remove_nulls(response_json)
    cleaned_response = remove_empty_strings(cleaned_response)
    cleaned_response = remove_auto_defined_fields(cleaned_response)
    return cleaned_response
