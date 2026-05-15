"""Utility functions for recursive tensor operations and validation."""

from typing import Any, Callable, Tuple, Union, List

from potatonumpy.exceptions import InvalidTensorError

Numeric = Union[int, float, complex]


def is_numeric(value: Any) -> bool:
    """Check if a value is a supported numeric type."""
    return isinstance(value, (int, float, complex))


def validate_tensor(data: Any) -> None:
    """Recursively validate that a tensor has consistent dimensions.

    Walks the nested list structure and ensures all sub-lists at the same
    depth have identical lengths. Raises InvalidTensorError on mismatch.
    """
    if is_numeric(data):
        return

    if not isinstance(data, list):
        raise InvalidTensorError(f"Expected list or numeric, got {type(data).__name__}")

    if len(data) == 0:
        raise InvalidTensorError("Empty arrays are not supported")

    first_is_list = isinstance(data[0], list)
    first_len = len(data[0]) if first_is_list else None

    for i in range(len(data)):
        element = data[i]
        current_is_list = isinstance(element, list)

        if current_is_list != first_is_list:
            raise InvalidTensorError(
                "Mixed types at same depth: some elements are lists, others are scalars"
            )

        if current_is_list:
            if len(element) != first_len:
                raise InvalidTensorError(
                    f"Inconsistent dimension at depth: lengths {first_len} and {len(element)}"
                )
            validate_tensor(element)
        elif not is_numeric(element):
            raise InvalidTensorError(
                f"Unsupported element type: {type(element).__name__}"
            )


def compute_shape(data: Any) -> Tuple[int, ...]:
    """Recursively compute the shape of a nested list structure."""
    if is_numeric(data):
        return ()

    if not isinstance(data, list):
        raise InvalidTensorError(f"Expected list or numeric, got {type(data).__name__}")

    if len(data) == 0:
        return (0,)

    inner_shape = compute_shape(data[0])
    return (len(data),) + inner_shape


def deep_copy(data: Any) -> Any:
    """Recursively deep copy a nested list structure."""
    if is_numeric(data):
        return data

    result = [None] * len(data)
    for i in range(len(data)):
        result[i] = deep_copy(data[i])
    return result


def recursive_apply(data: Any, func: Callable[[Numeric], Numeric]) -> Any:
    """Apply a function to every scalar element in a nested list."""
    if is_numeric(data):
        return func(data)

    result = [None] * len(data)
    for i in range(len(data)):
        result[i] = recursive_apply(data[i], func)
    return result


def recursive_elementwise(
    a: Any, b: Any, func: Callable[[Numeric, Numeric], Numeric]
) -> Any:
    """Apply a binary function elementwise to two identically-shaped nested lists."""
    if is_numeric(a) and is_numeric(b):
        return func(a, b)

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise InvalidTensorError(
                f"Dimension mismatch in elementwise operation: {len(a)} vs {len(b)}"
            )
        result = [None] * len(a)
        for i in range(len(a)):
            result[i] = recursive_elementwise(a[i], b[i], func)
        return result

    raise InvalidTensorError("Incompatible types for elementwise operation")


def flatten(data: Any) -> List[Numeric]:
    """Recursively flatten a nested list into a 1D list."""
    if is_numeric(data):
        return [data]

    result = []
    for i in range(len(data)):
        result.extend(flatten(data[i]))
    return result


def reshape_flat(flat: List[Numeric], shape: Tuple[int, ...]) -> Any:
    """Reshape a flat list into a nested list with the given shape."""
    if len(shape) == 0:
        if len(flat) != 1:
            raise InvalidTensorError("Cannot reshape: size mismatch")
        return flat[0]

    if len(shape) == 1:
        if len(flat) != shape[0]:
            raise InvalidTensorError(
                f"Cannot reshape: {len(flat)} elements into shape {shape}"
            )
        return list(flat)

    total = 1
    for dim in shape:
        total = total * dim

    if len(flat) != total:
        raise InvalidTensorError(
            f"Cannot reshape: {len(flat)} elements into shape {shape}"
        )

    outer_size = shape[0]
    inner_shape = shape[1:]
    inner_total = total // outer_size

    result = [None] * outer_size
    for i in range(outer_size):
        start = i * inner_total
        end = start + inner_total
        result[i] = reshape_flat(flat[start:end], inner_shape)
    return result


def recursive_equals(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Recursively compare two nested structures for near-equality."""
    if is_numeric(a) and is_numeric(b):
        return abs(a - b) < tol

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not recursive_equals(a[i], b[i], tol):
                return False
        return True

    return False
