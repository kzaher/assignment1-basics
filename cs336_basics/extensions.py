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
    object: _T, select_element: Callable[[_T], _R], final_value: _R
) -> _T:
    return select_element(Recorder(object))._replace_value(final_value)
