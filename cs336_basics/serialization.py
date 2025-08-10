# %%
import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from typing import List, Dict, get_origin, get_args
import json

def from_str(cls, data: str):
  return from_dict(cls, json.loads(data))

def from_dict(cls, data: dict):
    if not is_dataclass(cls):
        return data  # Primitive case

    fieldtypes = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for field_name, field_type in fieldtypes.items():
        value = data.get(field_name)
        if value is None:
            kwargs[field_name] = None
            continue

        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list and is_dataclass(args[0]):
            # Handle List[Dataclass]
            kwargs[field_name] = [from_dict(args[0], item) for item in value]

        elif origin is dict:
            key_type, val_type = args
            # Deserialize dictionary keys and values
            new_dict = {}
            for k, v in value.items():
                # Convert key to the appropriate type
                deserialized_key = key_type(k)
                
                # Convert value (dataclass or primitive)
                if is_dataclass(val_type):
                    deserialized_value = from_dict(val_type, v)
                else:
                    deserialized_value = val_type(v)
                
                new_dict[deserialized_key] = deserialized_value
            kwargs[field_name] = new_dict

        elif is_dataclass(field_type):
            # Nested dataclass
            kwargs[field_name] = from_dict(field_type, value)

        else:
            # Primitive field
            kwargs[field_name] = value

    return cls(**kwargs)

def as_dict(o: object) -> dict:
  return dataclasses.asdict(o)

def as_str(o: object) -> str:
  return json.dumps(as_dict(o))

if test := False:
  @dataclass
  class Address:
      street: str
      city: str
      zip_code: str

  @dataclass
  class Person:
      name: str
      age: int
      addresses: List[Address]
      address_book: Dict[int, Address]  # Dict with int keys now!

  json_data = '''
  {
      "name": "Alice",
      "age": 30,
      "addresses": [
          {
              "street": "123 Maple St",
              "city": "Springfield",
              "zip_code": "12345"
          }
      ],
      "address_book": {
          "1": {
              "street": "123 Maple St",
              "city": "Springfield",
              "zip_code": "12345"
          },
          "2": {
              "street": "456 Oak St",
              "city": "Greenville",
              "zip_code": "67890"
          }
      }
  }
  '''

  person = from_str(Person, json_data)

  print(person)
  print(from_str(Person, as_str(person)) == person)
  # %%
