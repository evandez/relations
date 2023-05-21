"""Utilities for interacting with dataclasses."""
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

T = TypeVar("T")


def create_with_optional_kwargs(cls: type[T], **kwargs: Any) -> T:
    """Create a dataclass instance with all optional fields set to None.

    Provided **kwargs must be sufficient to create the instance. But only those
    expected by the class will actually be passed.
    """
    if not is_dataclass(cls):
        raise TypeError(f"not a dataclass: {cls.__name__}")
    cls_kwargs = {}
    cls_fields = {field.name for field in fields(cls)}
    for key, value in kwargs.items():
        if key in cls_fields:
            cls_kwargs[key] = value
    return cls(**cls_kwargs)
