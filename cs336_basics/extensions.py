import logging
import sys
from typing import Callable, TypeVar, Any
import dataclasses


def setup_default_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


_T = TypeVar("_T")
_R = TypeVar("_R")


def _replace_at_index(o, replacement_index, value):
    if isinstance(o, list):
        return [
            value if replacement_index == index else original_value
            for index, original_value in enumerate(o)
        ]
    elif isinstance(o, dict):
        return {
            key: value if replacement_index == key else original_value
            for key, original_value in o.items()
        }
    else:
        raise Exception("Replace at index")


class Recorder:
    def __init__(self, current_object: Any):
        self._replace_value: Callable[[_R], _T] = lambda x: x
        self._current_object: Any = current_object

    def __getattribute__(self, name: str) -> Any:
        if name in ["_replace_value", "_current_object"]:
            return super().__getattribute__(name)

        previous_replace_value = self._replace_value
        current_object = self._current_object
        self._replace_value = lambda x: (
            previous_replace_value(dataclasses.replace(current_object, **{name: x}))
        )
        self._current_object = getattr(current_object, name)
        return self

    def __getitem__(self, item: str) -> Any:
        previous_replace_value = self._replace_value
        current_object = self._current_object
        self._replace_value = lambda x: (
            previous_replace_value(_replace_at_index(current_object, item, x))
        )
        self._current_object = self._current_object[item]
        return self


def replace_recursively(
    object: _T, select_element: Callable[[_T], _R], final_value: _R | None = None, transform: Callable[[_R], _R] | None = None
) -> _T:
    recorder = select_element(Recorder(object))
    if transform is not None:
        final_value = transform(recorder._current_object)
    return recorder._replace_value(final_value)

def _flatten_as_removed(node, path):
    """Helper: flatten a dict/list/primitive into removals."""
    changes = []
    if isinstance(node, dict):
        for k, v in node.items():
            new_path = f"{path}.{k}" if path else k
            changes.extend(_flatten_as_removed(v, new_path))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_path = f"{path}[{i}]"
            changes.extend(_flatten_as_removed(v, new_path))
    else:
        changes.append(f"{path}=<removed>")
    return changes


def _flatten_as_added(node, path):
    """Helper: flatten a dict/list/primitive into additions."""
    changes = []
    if isinstance(node, dict):
        for k, v in node.items():
            new_path = f"{path}.{k}" if path else k
            changes.extend(_flatten_as_added(v, new_path))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            new_path = f"{path}[{i}]"
            changes.extend(_flatten_as_added(v, new_path))
    else:
        changes.append(f"{path}={node}")
    return changes


def diff_json(json1, json2, path=""):
    """
    Recursively find differences between two JSON trees and record new values.
    Returns:
        List of "path=new_value" strings (<removed> for deletions)
    """
    changes = []

    # Handle None explicitly
    if json1 is None and json2 is not None:
        changes.extend(_flatten_as_added(json2, path))
        return changes
    if json2 is None and json1 is not None:
        changes.extend(_flatten_as_removed(json1, path))
        return changes

    # Type changed
    if type(json1) != type(json2):
        changes.extend(_flatten_as_removed(json1, path))
        changes.extend(_flatten_as_added(json2, path))
        return changes

    # Both dicts
    if isinstance(json1, dict):
        keys = set(json1.keys()).union(json2.keys())
        for key in keys:
            new_path = f"{path}.{key}" if path else key
            if key not in json1:
                changes.extend(_flatten_as_added(json2[key], new_path))
            elif key not in json2:
                changes.extend(_flatten_as_removed(json1[key], new_path))
            else:
                changes.extend(diff_json(json1[key], json2[key], new_path))

    # Both lists
    elif isinstance(json1, list):
        max_len = max(len(json1), len(json2))
        for i in range(max_len):
            new_path = f"{path}[{i}]"
            if i >= len(json1):
                changes.extend(_flatten_as_added(json2[i], new_path))
            elif i >= len(json2):
                changes.extend(_flatten_as_removed(json1[i], new_path))
            else:
                changes.extend(diff_json(json1[i], json2[i], new_path))

    # Both primitives
    else:
        if json1 != json2:
            changes.append(f"{path}={json2}")

    return changes


def json_diff_as_csv(json1, json2):
    return ",".join(diff_json(json1, json2))