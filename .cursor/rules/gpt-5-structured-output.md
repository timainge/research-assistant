# GPT-5.1 Structured Output Requirements

When using GPT-5.1's JSON schema mode with `strict: true`:

## All Objects Need `additionalProperties: false`

Every object in the schema must explicitly set this:

```python
schema = {
    "type": "object",
    "properties": {...},
    "required": [...],
    "additionalProperties": False  # REQUIRED for strict mode
}
```

## All Properties Must Be Required

With strict mode, every property defined must be in the `required` array:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age", "email"],  # ALL properties listed
    "additionalProperties": False
}
```

## Nested Objects Too

Apply recursively to all nested objects:

```python
schema = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": False  # Also here
        }
    },
    "required": ["user"],
    "additionalProperties": False
}
```

## Optional Fields

If a field is optional, make it nullable instead:

```python
"optional_field": {
    "type": ["string", "null"]  # Nullable, but still required
}
```

## Helper Function

Consider adding a recursive helper to ensure compliance:

```python
def add_additional_properties_false(schema: dict) -> dict:
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            for prop in schema["properties"].values():
                add_additional_properties_false(prop)
    if "items" in schema:
        add_additional_properties_false(schema["items"])
    return schema
```

