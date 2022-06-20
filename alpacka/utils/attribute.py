"""Handling attributes helpers."""

import copy
import functools


def recursive_getattr(obj, path, *default):
    """Recursive getattr."""
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def recursive_setattr(obj, path, value):
    """Recursive setattr."""
    pre, _, post = path.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj,
                   post,
                   value)


def deep_copy_without_fields(obj, fields_to_be_omitted):
    """Deep copies obj omitting some fields.

    Args:
        obj: Object to be copied.
        fields_to_be_omitted: Fields which will not be copied.

    Returns:
        Copied object without specified fields.
    """
    values_to_save = [getattr(obj, field_name)
                      for field_name in fields_to_be_omitted]
    for field_name in fields_to_be_omitted:
        setattr(obj, field_name, None)

    new_obj = copy.deepcopy(obj)

    for field_name, val in zip(fields_to_be_omitted, values_to_save):
        setattr(obj, field_name, val)

    return new_obj


def deep_copy_merge(obj_1, obj_2, fields_from_obj_2):
    """Deep copies obj filling some fields from the second object.

    Args:
        obj_1: First object to be merged.
        obj_2: Second object to be merged.
        fields_from_obj_2: Fields to take from the second object.

    Returns:
        Merged object.
    """
    values_to_plug = [getattr(obj_2, field_name)
                      for field_name in fields_from_obj_2]
    new_obj = copy.deepcopy(obj_1)
    for field_name, val in zip(fields_from_obj_2, values_to_plug):
        setattr(new_obj, field_name, val)

    return new_obj
