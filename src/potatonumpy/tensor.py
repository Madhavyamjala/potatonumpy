"""Tensor operations for arbitrary-dimensional PotatoNumPy arrays.

Provides recursive algorithms that operate on tensors of any depth,
including summation, min/max, and axis-based reduction.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from potatonumpy.utils import is_numeric, flatten, compute_shape
from potatonumpy.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from potatonumpy.core import PotatoNumPy


def tensor_sum(a: PotatoNumPy, axis: Optional[int] = None) -> Any:
    """Sum all elements or reduce along a specific axis.

    When axis is None, returns a scalar sum of all elements.
    When axis is specified, collapses that dimension by summing.
    """
    from potatonumpy.core import PotatoNumPy as PP

    if axis is None:
        return _recursive_sum(a.data)

    if axis < 0 or axis >= a.ndim:
        raise InvalidOperationError(f"Axis {axis} out of range for {a.ndim}D array")

    result = _reduce_axis(a.data, axis, 0, _add_elements)
    if is_numeric(result):
        return result
    return PP(result)


def tensor_min(a: PotatoNumPy, axis: Optional[int] = None) -> Any:
    """Find the minimum element, optionally along an axis."""
    from potatonumpy.core import PotatoNumPy as PP

    if axis is None:
        flat = flatten(a.data)
        minimum = flat[0]
        for i in range(1, len(flat)):
            if flat[i] < minimum:
                minimum = flat[i]
        return minimum

    if axis < 0 or axis >= a.ndim:
        raise InvalidOperationError(f"Axis {axis} out of range for {a.ndim}D array")

    result = _reduce_axis(a.data, axis, 0, _min_elements)
    if is_numeric(result):
        return result
    return PP(result)


def tensor_max(a: PotatoNumPy, axis: Optional[int] = None) -> Any:
    """Find the maximum element, optionally along an axis."""
    from potatonumpy.core import PotatoNumPy as PP

    if axis is None:
        flat = flatten(a.data)
        maximum = flat[0]
        for i in range(1, len(flat)):
            if flat[i] > maximum:
                maximum = flat[i]
        return maximum

    if axis < 0 or axis >= a.ndim:
        raise InvalidOperationError(f"Axis {axis} out of range for {a.ndim}D array")

    result = _reduce_axis(a.data, axis, 0, _max_elements)
    if is_numeric(result):
        return result
    return PP(result)


def tensor_mean(a: PotatoNumPy) -> float:
    """Compute the arithmetic mean of all elements."""
    total = _recursive_sum(a.data)
    return total / a.size


def _recursive_sum(data: Any) -> Any:
    """Recursively sum all scalar elements in a nested structure."""
    if is_numeric(data):
        return data

    total = 0
    for i in range(len(data)):
        total = total + _recursive_sum(data[i])
    return total


def _add_elements(a: Any, b: Any) -> Any:
    if is_numeric(a) and is_numeric(b):
        return a + b
    if isinstance(a, list) and isinstance(b, list):
        result = [None] * len(a)
        for i in range(len(a)):
            result[i] = _add_elements(a[i], b[i])
        return result
    raise InvalidOperationError("Incompatible structures in reduction")


def _min_elements(a: Any, b: Any) -> Any:
    if is_numeric(a) and is_numeric(b):
        return a if a < b else b
    if isinstance(a, list) and isinstance(b, list):
        result = [None] * len(a)
        for i in range(len(a)):
            result[i] = _min_elements(a[i], b[i])
        return result
    raise InvalidOperationError("Incompatible structures in reduction")


def _max_elements(a: Any, b: Any) -> Any:
    if is_numeric(a) and is_numeric(b):
        return a if a > b else b
    if isinstance(a, list) and isinstance(b, list):
        result = [None] * len(a)
        for i in range(len(a)):
            result[i] = _max_elements(a[i], b[i])
        return result
    raise InvalidOperationError("Incompatible structures in reduction")


def _reduce_axis(data: Any, target_axis: int, current_axis: int, combine_func) -> Any:
    """Recursively reduce along a target axis.

    Walks down the nesting until reaching the target axis depth,
    then combines all slices along that axis using combine_func.
    """
    if current_axis == target_axis:
        if not isinstance(data, list) or len(data) == 0:
            raise InvalidOperationError("Cannot reduce along empty axis")

        accumulator = data[0]
        for i in range(1, len(data)):
            accumulator = combine_func(accumulator, data[i])
        return accumulator

    if not isinstance(data, list):
        return data

    result = [None] * len(data)
    for i in range(len(data)):
        result[i] = _reduce_axis(data[i], target_axis, current_axis + 1, combine_func)
    return result
