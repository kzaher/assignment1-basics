#!/usr/bin/env python3
"""
Generate JSON schema from dataclasses for better IDE support.
"""

import json
import dataclasses
from typing import get_type_hints, get_origin, get_args
from cs336_basics.experiments.configuration import (
    LlmPretrainingConfiguration,
    OptimizerConfiguration,
    TransformerLlmConfiguration, 
    AnnealingConfiguration
)

def dataclass_to_json_schema(cls) -> dict:
    """Convert a dataclass to a JSON schema for IDE autocomplete."""
    
    if not dataclasses.is_dataclass(cls):
        # Handle primitive types
        type_mapping = {
            int: {"type": "integer"},
            float: {"type": "number"}, 
            str: {"type": "string"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"}
        }
        return type_mapping.get(cls, {"type": "string"})
    
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Get type hints for the dataclass
    type_hints = get_type_hints(cls)
    
    for field in dataclasses.fields(cls):
        field_type = type_hints[field.name]
        field_schema = convert_type_to_schema(field_type)
        
        schema["properties"][field.name] = field_schema
        
        # Add to required if no default value
        if field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:
            schema["required"].append(field.name)
    
    return schema

def convert_type_to_schema(field_type) -> dict:
    """Convert a Python type to JSON schema."""
    
    # Handle basic types
    if field_type == int:
        return {"type": "integer"}
    elif field_type == float:
        return {"type": "number"}
    elif field_type == str:
        return {"type": "string"} 
    elif field_type == bool:
        return {"type": "boolean"}
    
    # Handle generic types (List, Dict, etc.)
    origin = get_origin(field_type)
    args = get_args(field_type)
    
    if origin is list:
        if args:
            item_schema = convert_type_to_schema(args[0])
            return {
                "type": "array",
                "items": item_schema
            }
        return {"type": "array"}
    
    elif origin is dict:
        if len(args) >= 2:
            value_schema = convert_type_to_schema(args[1])
            return {
                "type": "object",
                "additionalProperties": value_schema
            }
        return {"type": "object"}
    
    # Handle dataclass types
    elif dataclasses.is_dataclass(field_type):
        return dataclass_to_json_schema(field_type)
    
    # Fallback
    return {"type": "string"}

def generate_config_schema():
    """Generate complete JSON schema for the configuration."""
    return dataclass_to_json_schema(LlmPretrainingConfiguration)

def save_schema_file(schema_path: str = "/workspace/config_schema.json"):
    """Save the JSON schema to a file for IDE integration."""
    schema = generate_config_schema()
    
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"Schema saved to {schema_path}")
    return schema_path

if __name__ == "__main__":
    # Generate and save schema
    schema_path = save_schema_file()
    
    # Print schema for inspection
    schema = generate_config_schema()
    print("Generated schema:")
    print(json.dumps(schema, indent=2))
