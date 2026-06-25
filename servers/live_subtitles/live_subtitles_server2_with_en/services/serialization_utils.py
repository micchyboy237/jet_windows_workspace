import base64
import collections.abc
import datetime
import json
import math  # ← Add this import
import re
import types
from dataclasses import asdict, is_dataclass
from decimal import Decimal  # Added for Decimal serialization
from enum import Enum
from typing import Any, Dict

import numpy as np
from pydantic.main import BaseModel


# Function to check if an object is an instance of a user-defined class
def _is_class_instance(obj):
    return isinstance(obj, object) and not isinstance(obj, dict)


def _get_non_empty_attributes(obj: Any) -> Dict[str, Any]:
    """Returns non-callable attributes that are not empty or private (_attr) for objects or dictionaries."""
    attributes = {}

    if isinstance(obj, dict):
        # Handle dictionary case
        for key, value in obj.items():
            if not str(key).startswith("_"):  # Exclude private keys
                if (
                    value is not None
                    and value not in ["", [], {}, ()]
                    and not callable(value)
                ):
                    attributes[str(key)] = value
    else:
        # Handle object case
        for attr in dir(obj):
            if not attr.startswith("_"):  # Exclude private and dunder attributes
                try:
                    value = getattr(obj, attr, None)  # Safely get attribute
                    if (
                        value is not None
                        and value not in ["", [], {}, ()]
                        and not callable(value)
                    ):
                        attributes[attr] = value
                except AttributeError as e:
                    print(f"Skipping attribute {attr} due to error: {e}")
                except Exception as e:
                    print(f"Unexpected error for attribute {attr}: {e}")

    return attributes


def _to_snake_case(s: str) -> str:
    """Convert a string to lowercase and underscore-separated, replacing spaces and normalizing underscores."""
    # Replace all whitespace with a single underscore
    s = re.sub(r"\s+", "_", s.strip())
    # Insert underscore before uppercase letters (except at start), then lowercase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Convert to lowercase and normalize multiple underscores to one
    s = re.sub(r"_+", "_", s.lower())
    return s


def _convert_dict_keys_to_snake_case(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert dictionary keys to snake_case."""
    if not isinstance(d, dict):
        return d
    return {
        _to_snake_case(k): [_convert_dict_keys_to_snake_case(item) for item in v]
        if isinstance(v, list)
        else _convert_dict_keys_to_snake_case(v)
        if isinstance(v, dict)
        else v
        for k, v in d.items()
    }


def serialize(obj, seen=None):
    """
    Recursively converts an object's attributes to be serializable.
    Args:
        obj: The input object to process.
        seen: Set of object IDs to track processed objects and prevent circular references.
    Returns:
        A serializable representation of the object.
    """
    if seen is None:
        seen = set()

    def _serialize_inner(inner_obj, seen):
        # Handle Protobuf objects from google._upb._message
        if hasattr(
            inner_obj, "__class__"
        ) and inner_obj.__class__.__module__.startswith("google._upb._message"):
            # Check if the object is iterable (but not a string or bytes)
            try:
                is_iterable = isinstance(
                    inner_obj, collections.abc.Iterable
                ) and not isinstance(inner_obj, (str, bytes))
            except TypeError:
                is_iterable = False
            if is_iterable:
                try:
                    # Serialize each item and apply snake_case to dictionaries
                    serialized_items = []
                    for item in inner_obj:
                        serialized_item = _serialize_inner(item, seen.copy())
                        if isinstance(serialized_item, dict):
                            serialized_item = _convert_dict_keys_to_snake_case(
                                serialized_item
                            )
                        serialized_items.append(serialized_item)
                    return serialized_items
                except Exception:
                    return (
                        inner_obj.__class__.__name__
                    )  # Fallback to class name if iteration fails
            return inner_obj.__class__.__name__  # Non-iterable Protobuf objects

        if isinstance(inner_obj, Enum):
            return inner_obj.value

        if isinstance(inner_obj, Decimal):
            if inner_obj == inner_obj.to_integral_value(rounding="ROUND_DOWN"):
                return int(inner_obj)
            else:
                return float(inner_obj)

        elif isinstance(inner_obj, bytes):
            try:
                decoded_str = inner_obj.decode("utf-8")
            except UnicodeDecodeError:
                decoded_str = base64.b64encode(inner_obj).decode("utf-8")
            return _serialize_inner(decoded_str, seen.copy())

        elif isinstance(inner_obj, complex):
            return {
                "real": _serialize_inner(inner_obj.real, seen.copy()),
                "imag": _serialize_inner(inner_obj.imag, seen.copy()),
            }

        # ──── Quick recursion depth guard ─────────────────────────────────────
        recursion_depth = len(seen)
        if recursion_depth > 45:
            type_name = type(inner_obj).__name__
            short_repr = repr(inner_obj)[:120].replace("\n", " ")
            return f"[RECURSION_LIMIT {recursion_depth}] {type_name} {short_repr}"
        # ──────────────────────────────────────────────────────────────────────

        elif isinstance(inner_obj, (int, float, bool, type(None))):
            if isinstance(inner_obj, float):
                if math.isinf(inner_obj):
                    return "Infinity" if inner_obj > 0 else "-Infinity"
                if math.isnan(inner_obj):
                    return "NaN"
            return inner_obj
        elif isinstance(inner_obj, str):
            try:
                parsed_obj = json.loads(inner_obj)
                if isinstance(parsed_obj, (dict, list)):
                    return _serialize_inner(parsed_obj, seen.copy())
                return inner_obj
            except json.JSONDecodeError:
                return inner_obj
        elif isinstance(
            inner_obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)
        ):
            return str(type(inner_obj))
        elif isinstance(inner_obj, (np.integer, np.floating)):
            value = inner_obj.item()
            if isinstance(value, float):
                if math.isinf(value):
                    return "Infinity" if value > 0 else "-Infinity"
                if math.isnan(value):
                    return "NaN"
            return value
        elif isinstance(inner_obj, np.ndarray):
            return inner_obj.tolist()

        # ──── datetime family ───────────────────────────────────────────────
        elif isinstance(inner_obj, datetime.datetime):
            return inner_obj.isoformat()
        elif isinstance(inner_obj, (datetime.date, datetime.time)):
            return inner_obj.isoformat()
        # ────────────────────────────────────────────────────────────────────

        elif isinstance(inner_obj, BaseModel):
            try:
                return _serialize_inner(inner_obj.model_dump(), seen.copy())
            except (AttributeError, TypeError):
                return _serialize_inner(vars(inner_obj), seen.copy())
        elif is_dataclass(inner_obj):
            try:
                return _serialize_inner(asdict(inner_obj), seen.copy())
            except Exception:
                return _serialize_inner(vars(inner_obj), seen.copy())
        if id(inner_obj) in seen:
            return f"<{inner_obj.__class__.__name__} object>"
        new_seen = seen.copy()
        new_seen.add(id(inner_obj))
        if isinstance(inner_obj, set):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, tuple):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, list):
            return [_serialize_inner(item, new_seen) for item in inner_obj]
        elif isinstance(inner_obj, dict):
            serialized_dict = {}
            for key, value in inner_obj.items():
                serialized_key = str(key) if not isinstance(key, str) else key
                serialized_dict[serialized_key] = _serialize_inner(value, new_seen)
            return _convert_dict_keys_to_snake_case(serialized_dict)
        elif hasattr(inner_obj, "__dict__"):
            try:
                if callable(getattr(inner_obj, "__dict__", None)):
                    dict_data = inner_obj.__dict__()
                else:
                    dict_data = _get_non_empty_attributes(
                        inner_obj
                    )  # ← place breakpoint here
                return _serialize_inner(dict_data, new_seen)
            except Exception:
                return str(inner_obj)
        elif _is_class_instance(inner_obj):
            dict_data = _get_non_empty_attributes(inner_obj)
            return _serialize_inner(dict_data, new_seen)
        else:
            return f"{type(inner_obj).__name__}({str(inner_obj)[:80]})"

    result = _serialize_inner(obj, seen)
    return _convert_dict_keys_to_snake_case(result)
